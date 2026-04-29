import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, CLIPImageProcessor, CLIPModel, CLIPTokenizer, Dinov2Model


logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images from a DreamBooth-like model using CLIP-I, CLIP-T, and DINO metrics.")
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="data/dreambooth_original/cat2",
        help="Directory containing the reference images.",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default="results/cat2",
        help="Directory containing the generated images.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file containing prompts, one per line. If not provided, prompts will be derived from generated image filenames.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where metric files will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model used for CLIP-I and CLIP-T.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="facebook/dinov2-base",
        help="DINO model used for the DINO similarity metric.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_image_files(directory: Path) -> list[Path]:
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in allowed_suffixes])


def load_images(directory: Path) -> tuple[list[Path], list[Image.Image]]:
    image_paths = get_image_files(directory)
    if not image_paths:
        raise ValueError(f"No images were found in {directory}.")

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return image_paths, images


def load_prompts(prompts_file: str) -> list[str]:
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=-1)


def _to_embedding_tensor(features):
    if isinstance(features, torch.Tensor):
        return features

    # Some transformer versions may return model outputs instead of plain tensors.
    for attr in ["image_embeds", "text_embeds", "pooler_output", "last_hidden_state"]:
        if hasattr(features, attr):
            value = getattr(features, attr)
            if isinstance(value, torch.Tensor):
                if attr == "last_hidden_state":
                    return value[:, 0]
                return value

    if isinstance(features, (tuple, list)) and len(features) > 0 and isinstance(features[0], torch.Tensor):
        return features[0]

    raise TypeError(f"Unsupported embedding output type: {type(features)}")


@torch.inference_mode()
def encode_clip_images(
    images: list[Image.Image],
    processor: CLIPImageProcessor,
    model: CLIPModel,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    image_embeddings = _to_embedding_tensor(model.get_image_features(pixel_values=pixel_values))
    return normalize_embeddings(image_embeddings.float())


@torch.inference_mode()
def encode_clip_texts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    model: CLIPModel,
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    text_embeddings = _to_embedding_tensor(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
    return normalize_embeddings(text_embeddings.float())


@torch.inference_mode()
def encode_dino_images(
    images: list[Image.Image],
    processor: AutoImageProcessor,
    model: Dinov2Model,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    outputs = model(pixel_values=pixel_values)
    embeddings = outputs.last_hidden_state[:, 0]
    return normalize_embeddings(embeddings.float())


def cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs = normalize_embeddings(lhs.float())
    rhs = normalize_embeddings(rhs.float())
    return lhs @ rhs.T


def evaluate_generated_images(
    reference_images: list[Image.Image],
    generated_images: list[Image.Image],
    prompts: list[str],
    clip_model: CLIPModel,
    clip_image_processor: CLIPImageProcessor,
    clip_tokenizer: CLIPTokenizer,
    dino_model: Dinov2Model,
    dino_processor: AutoImageProcessor,
    device: torch.device,
) -> dict:
    # Encode reference images
    reference_clip_embeddings = encode_clip_images(reference_images, clip_image_processor, clip_model, device)
    reference_dino_embeddings = encode_dino_images(reference_images, dino_processor, dino_model, device)
    reference_clip_embedding = normalize_embeddings(reference_clip_embeddings.mean(dim=0, keepdim=True))
    reference_dino_embedding = normalize_embeddings(reference_dino_embeddings.mean(dim=0, keepdim=True))

    # Encode generated images
    generated_clip_embeddings = encode_clip_images(generated_images, clip_image_processor, clip_model, device)
    generated_dino_embeddings = encode_dino_images(generated_images, dino_processor, dino_model, device)

    # Encode prompts
    text_embeddings = encode_clip_texts(prompts, clip_tokenizer, clip_model, device)

    # Calculate similarities
    clip_i_scores = cosine_similarity(generated_clip_embeddings, reference_clip_embedding).squeeze(-1)
    clip_t_scores = cosine_similarity(generated_clip_embeddings, text_embeddings).diagonal()
    dino_scores = cosine_similarity(generated_dino_embeddings, reference_dino_embedding).squeeze(-1)

    results = {
        "num_reference_images": len(reference_images),
        "num_generated_images": len(generated_images),
        "num_prompts": len(prompts),
        "clip_i": float(clip_i_scores.mean().item()),
        "clip_t": float(clip_t_scores.mean().item()),
        "dino": float(dino_scores.mean().item()),
        "individual_scores": {
            "clip_i": clip_i_scores.tolist(),
            "clip_t": clip_t_scores.tolist(),
            "dino": dino_scores.tolist(),
        }
    }
    return results


def save_results(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"eval_log_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved to {json_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    args = parse_args()
    device = resolve_device(args.device)

    reference_dir = Path(args.reference_dir).expanduser()
    generated_dir = Path(args.generated_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not reference_dir.exists():
        raise FileNotFoundError(f"The reference directory does not exist: {reference_dir}")
    if not generated_dir.exists():
        raise FileNotFoundError(f"The generated directory does not exist: {generated_dir}")

    logger.info("Loading reference images from %s", reference_dir)
    reference_image_paths, reference_images = load_images(reference_dir)

    logger.info("Loading generated images from %s", generated_dir)
    generated_image_paths, generated_images = load_images(generated_dir)

    if args.prompts_file:
        prompts_file = Path(args.prompts_file).expanduser()
        logger.info("Loading prompts from %s", prompts_file)
        prompts = load_prompts(str(prompts_file))
    else:
        logger.info("Deriving prompts from generated image filenames")
        prompts = [path.stem for path in generated_image_paths]

    # Ensure we have prompts for each generated image
    if len(prompts) != len(generated_images):
        raise ValueError(f"Number of prompts ({len(prompts)}) must match number of generated images ({len(generated_images)})")

    logger.info("Loading CLIP model %s", args.clip_model_name)
    clip_dtype = torch.float16 if device.type == "cuda" else torch.float32
    clip_model = CLIPModel.from_pretrained(args.clip_model_name, torch_dtype=clip_dtype).eval().to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)

    logger.info("Loading DINO model %s", args.dino_model_name)
    dino_model = Dinov2Model.from_pretrained(args.dino_model_name, torch_dtype=clip_dtype).eval().to(device)
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_name)

    results = evaluate_generated_images(
        reference_images=reference_images,
        generated_images=generated_images,
        prompts=prompts,
        clip_model=clip_model,
        clip_image_processor=clip_image_processor,
        clip_tokenizer=clip_tokenizer,
        dino_model=dino_model,
        dino_processor=dino_processor,
        device=device,
    )

    save_results(output_dir, results)

    logger.info(
        "Finished evaluation: CLIP-I=%.4f CLIP-T=%.4f DINO=%.4f",
        results["clip_i"],
        results["clip_t"],
        results["dino"],
    )


if __name__ == "__main__":
    main()