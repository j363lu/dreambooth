"""Train DreamBooth weights for Stable Diffusion.

Example:
    accelerate launch code/train_dreambooth.py \
        --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
        --instance_data_dir data/dreambooth_original/cat2 \
        --class_data_dir data/generated/cat-stable-diffusion-1_5 \
        --instance_prompt "a sks cat" \
        --class_prompt "a cat" \
        --with_prior_preservation \
        --output_dir outputs/cat_lora

This script follows the DreamBooth objective:
    loss = reconstruction_loss + prior_loss_weight * prior_preservation_loss

By default, only LoRA adapters on the UNet attention layers are trained. Pass
--no_lora to fine-tune the full UNet instead. The VAE and text encoder are kept
frozen in both modes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with DreamBooth.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, help="HF model id or local model path.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--variant", default=None, help="Optional model variant, for example fp16.")

    parser.add_argument("--instance_data_dir", required=True, type=Path, help="Directory with subject images.")
    parser.add_argument("--class_data_dir", type=Path, default=None, help="Directory with class images for prior preservation.")
    parser.add_argument("--output_dir", default="lora-dreambooth-model", type=Path, help="Where to save LoRA weights.")

    parser.add_argument("--instance_prompt", required=True, help='Prompt with the unique token, e.g. "a sks dog".')
    parser.add_argument("--class_prompt", default=None, help='Class prompt, e.g. "a dog". Required with prior preservation.')
    parser.add_argument("--with_prior_preservation", action="store_true", help="Use DreamBooth class prior loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss.")
    parser.add_argument("--num_class_images", type=int, default=100, help="Target number of generated class images.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size when generating class images.")
    parser.add_argument(
        "--use_lora",
        dest="use_lora",
        action="store_true",
        default=True,
        help="Train LoRA adapters on the UNet attention layers.",
    )
    parser.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        help="Fine-tune the full UNet instead of training LoRA adapters.",
    )

    parser.add_argument("--resolution", type=int, default=512, help="Training image resolution.")
    parser.add_argument("--center_crop", action="store_true", help="Center crop instead of random crop.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save LoRA weights every N steps; 0 disables.")

    args = parser.parse_args()

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            parser.error("--class_data_dir is required when --with_prior_preservation is set.")
        if args.class_prompt is None:
            parser.error("--class_prompt is required when --with_prior_preservation is set.")
    if not args.instance_data_dir.exists():
        parser.error(f"--instance_data_dir does not exist: {args.instance_data_dir}")

    return args


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def image_grid(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    """Tile same-sized PIL images into a single rows-by-cols preview image."""
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image, box=(index % cols * width, index // cols * height))
    return grid


def generate_class_images(args: argparse.Namespace, accelerator: Accelerator) -> None:
    """Generate missing class images used by the DreamBooth prior loss."""
    if not args.with_prior_preservation:
        return

    args.class_data_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list_images(args.class_data_dir)
    images_to_generate = args.num_class_images - len(existing_images)
    if images_to_generate <= 0:
        return

    LOGGER.info("Generating %d class images for prior preservation.", images_to_generate)
    torch_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipeline.set_progress_bar_config(disable=not accelerator.is_local_main_process)
    pipeline.to(accelerator.device)

    generated = 0
    while generated < images_to_generate:
        batch_size = min(args.sample_batch_size, images_to_generate - generated)
        prompts = [args.class_prompt] * batch_size
        images = pipeline(prompts, num_inference_steps=50).images
        for image in images:
            digest = hashlib.sha1(image.tobytes()).hexdigest()
            image.save(args.class_data_dir / f"{len(existing_images) + generated:05d}-{digest}.jpg")
            generated += 1

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_dir: Path,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        size: int,
        center_crop: bool,
        class_data_dir: Path | None = None,
        class_prompt: str | None = None,
    ) -> None:
        self.instance_images = list_images(instance_data_dir)
        if not self.instance_images:
            raise ValueError(f"No training images found in {instance_data_dir}")

        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer

        self.class_images = list_images(class_data_dir) if class_data_dir is not None else []
        self.class_prompt = class_prompt
        self.length = max(len(self.instance_images), len(self.class_images) or 0)

        crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        instance_image = self.load_image(self.instance_images[index % len(self.instance_images)])
        example = {
            "instance_pixel_values": self.image_transforms(instance_image),
            "instance_input_ids": self.tokenize(self.instance_prompt),
        }

        if self.class_images:
            class_image = self.load_image(self.class_images[index % len(self.class_images)])
            example["class_pixel_values"] = self.image_transforms(class_image)
            example["class_input_ids"] = self.tokenize(self.class_prompt or "")

        return example

    def tokenize(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return inputs.input_ids[0]

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image


def collate_fn(examples: list[dict[str, torch.Tensor]], with_prior_preservation: bool) -> dict[str, torch.Tensor]:
    pixel_values = [example["instance_pixel_values"] for example in examples]
    input_ids = [example["instance_input_ids"] for example in examples]

    if with_prior_preservation:
        pixel_values += [example["class_pixel_values"] for example in examples]
        input_ids += [example["class_input_ids"] for example in examples]

    return {
        "pixel_values": torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
        "input_ids": torch.stack(input_ids),
    }


def freeze_parameters(parameters: Iterable[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.requires_grad = False


def save_lora_weights(unet: UNet2DConditionModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_state_dict = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )


def save_full_pipeline(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=True)


def main() -> None:
    args = parse_args()
    logging_dir = args.output_dir / "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        generate_class_images(args, accelerator)
    accelerator.wait_for_everyone()

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    freeze_parameters(vae.parameters())
    freeze_parameters(text_encoder.parameters())

    if args.use_lora:
        freeze_parameters(unet.parameters())
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        class_data_dir=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    trainable_params = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * updates_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / updates_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    LOGGER.info(
        "Training %d %s parameters.",
        sum(parameter.numel() for parameter in trainable_params),
        "LoRA" if args.use_lora else "UNet",
    )
    LOGGER.info(
        "Training setup: epochs=%d, max_train_steps=%d, updates_per_epoch=%d, batches_per_epoch=%d, "
        "dataset_size=%d, batch_size=%d, gradient_accumulation_steps=%d, learning_rate=%g, "
        "with_prior_preservation=%s, use_lora=%s.",
        args.num_train_epochs,
        args.max_train_steps,
        updates_per_epoch,
        len(train_dataloader),
        len(train_dataset),
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.with_prior_preservation,
        args.use_lora,
    )
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        LOGGER.info("Starting epoch %d/%d at global step %d.", epoch + 1, args.num_train_epochs, global_step)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_preservation_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = reconstruction_loss + args.prior_loss_weight * prior_preservation_loss
                else:
                    reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_preservation_loss = torch.tensor(0.0, device=accelerator.device)
                    loss = reconstruction_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "reconstruction_loss": reconstruction_loss.detach().item(),
                    "prior_preservation_loss": prior_preservation_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                        if args.use_lora:
                            save_lora_weights(accelerator.unwrap_model(unet), checkpoint_dir)
                        else:
                            accelerator.unwrap_model(unet).save_pretrained(
                                checkpoint_dir / "unet",
                                safe_serialization=True,
                            )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            LOGGER.info(
                "Finished epoch %d/%d at global step %d/%d; reached max_train_steps.",
                epoch + 1,
                args.num_train_epochs,
                global_step,
                args.max_train_steps,
            )
            break
        LOGGER.info(
            "Finished epoch %d/%d at global step %d/%d.",
            epoch + 1,
            args.num_train_epochs,
            global_step,
            args.max_train_steps,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            save_lora_weights(accelerator.unwrap_model(unet), args.output_dir)
            LOGGER.info("Saved LoRA weights to %s", args.output_dir)
        else:
            save_full_pipeline(
                accelerator.unwrap_model(unet),
                vae,
                text_encoder,
                tokenizer,
                noise_scheduler,
                args.output_dir,
            )
            LOGGER.info("Saved full DreamBooth pipeline to %s", args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
