"""Generate images from a DreamBooth checkpoint.

Examples:
    # LoRA checkpoint, requires the base model.
    python code/inference_dreambooth.py \
        --checkpoint_dir outputs/cat_lora \
        --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
        --prompt "a sks cat sitting on a chair" \
        --output_dir outputs/cat_samples

    # Full DreamBooth pipeline checkpoint saved with --no_lora.
    python code/inference_dreambooth.py \
        --checkpoint_dir outputs/cat_full \
        --prompt "a sks cat sitting on a chair" \
        --output_dir outputs/cat_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


LORA_WEIGHT_NAMES = {
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a DreamBooth checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True, type=Path, help="Saved LoRA, UNet, or full pipeline checkpoint.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Base model id or path. Required for LoRA and UNet-only checkpoints.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to generate images from.")
    parser.add_argument("--negative_prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--output_dir", default=Path("outputs/inference"), type=Path, help="Where to save generated images.")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to generate at once.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--checkpoint_type",
        choices=["auto", "lora", "full_pipeline", "unet"],
        default="auto",
        help="Checkpoint format. Auto detects checkpoints saved by code/train_dreambooth.py.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)

    args = parser.parse_args()
    if not args.checkpoint_dir.exists():
        parser.error(f"--checkpoint_dir does not exist: {args.checkpoint_dir}")
    if args.num_images < 1:
        parser.error("--num_images must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")
    return args


def get_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.device.startswith("cuda") and args.mixed_precision == "fp16":
        return torch.float16
    if args.device.startswith("cuda") and args.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def detect_checkpoint_type(checkpoint_dir: Path) -> str:
    if (checkpoint_dir / "model_index.json").exists():
        return "full_pipeline"
    if any((checkpoint_dir / weight_name).exists() for weight_name in LORA_WEIGHT_NAMES):
        return "lora"
    if (checkpoint_dir / "unet" / "config.json").exists() or (checkpoint_dir / "config.json").exists():
        return "unet"
    raise ValueError(
        "Could not auto-detect checkpoint type. Expected a full pipeline with model_index.json, "
        "a LoRA checkpoint with pytorch_lora_weights.*, or a UNet checkpoint with config.json. "
        "Pass --checkpoint_type explicitly if needed."
    )


def load_pipeline(args: argparse.Namespace) -> StableDiffusionPipeline:
    torch_dtype = get_torch_dtype(args)
    checkpoint_type = args.checkpoint_type
    if checkpoint_type == "auto":
        checkpoint_type = detect_checkpoint_type(args.checkpoint_dir)
    print(f"Loading {checkpoint_type} checkpoint from {args.checkpoint_dir}")

    if checkpoint_type == "full_pipeline":
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        if checkpoint_type == "lora":
            pipeline.load_lora_weights(args.checkpoint_dir)
        elif checkpoint_type == "unet":
            unet_dir = args.checkpoint_dir / "unet" if (args.checkpoint_dir / "unet").exists() else args.checkpoint_dir
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=torch_dtype,
            )
        else:
            raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")

    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    pipeline = load_pipeline(args)
    generated = 0
    while generated < args.num_images:
        batch_size = min(args.batch_size, args.num_images - generated)
        images = pipeline(
            prompt=[args.prompt] * batch_size,
            negative_prompt=[args.negative_prompt] * batch_size if args.negative_prompt else None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images

        for image in images:
            output_path = args.output_dir / f"sample_{generated:03d}.png"
            image.save(output_path)
            print(f"Saved {output_path}")
            generated += 1


if __name__ == "__main__":
    main()
