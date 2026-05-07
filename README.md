# DreamBooth Reproduction

![DreamBooth generated examples](results/dreambooth.png)

CS 5782 final project reproducing
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242).
This repository fine-tunes Stable Diffusion on a small set of subject images,
generates subject-driven images from the adapted model, and evaluates the
outputs with DINO, CLIP-I, and CLIP-T similarity metrics.

## Overview

DreamBooth adapts a text-to-image diffusion model to a specific subject using a
small number of reference images and a unique identifier token. This
implementation trains Stable Diffusion with the DreamBooth objective:

```text
loss = reconstruction_loss + prior_loss_weight * prior_preservation_loss
```

The reconstruction loss teaches the model the target subject, while the prior
preservation loss helps retain the general class concept, such as `cat`,
`dog`, or `toy`.

## Repository Layout

```text
code/
  train_dreambooth.py       DreamBooth training entry point
  inference_dreambooth.py   Image generation from trained checkpoints
  eval.py                   DINO, CLIP-I, and CLIP-T evaluation
data/
  dreambooth_original/      Official DreamBooth subject images
  generated/                Generated class images for prior preservation
  our_data/                 Custom subject images collected for this project
results/                    Generated examples, comparisons, and metrics
```

See `data/README.md` for dataset details and `results/README.md` for qualitative
examples and quantitative metric comparisons.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Training and inference use Stable Diffusion v1.5 by default:

```text
runwayml/stable-diffusion-v1-5
```

GPU acceleration is strongly recommended for training.

## Training

The training entry point is `code/train_dreambooth.py`. A typical run with prior
preservation is:

```bash
accelerate launch code/train_dreambooth.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --instance_data_dir data/dreambooth_original/cat2 \
  --class_data_dir data/generated/cat-stable-diffusion-1_5 \
  --instance_prompt "a sks cat" \
  --class_prompt "a cat" \
  --with_prior_preservation \
  --num_class_images 100 \
  --max_train_steps 800 \
  --output_dir outputs/cat
```

`--instance_data_dir` points to the few-shot subject images, such as
`data/dreambooth_original/cat2` or `data/our_data/bottle`. These images are
paired with the instance prompt, such as `"a sks cat"`, and contribute the
reconstruction loss.

`--class_data_dir` points to generic class images used for prior preservation.
If the directory has fewer than `--num_class_images` images, the training script
can generate the missing images from the base model using `--class_prompt`.

The trained weights are saved to `--output_dir`.

## Inference

Use `code/inference_dreambooth.py` to generate images from a trained checkpoint:

```bash
python code/inference_dreambooth.py \
  --checkpoint_dir outputs/cat \
  --prompt "a sks cat in a garden" \
  --output_dir outputs/inference/cat_garden \
  --num_images 4 \
  --seed 0
```

The script supports LoRA checkpoints, full Stable Diffusion pipelines, and UNet
checkpoints through `--checkpoint_type`.

## Evaluation

The evaluation entry point is `code/eval.py`. It compares generated images with
reference images and prompts using:

- **DINO**: image-to-image similarity using self-supervised visual features.
- **CLIP-I**: image-to-image similarity in CLIP embedding space.
- **CLIP-T**: image-to-text similarity in CLIP embedding space.

Run evaluation with the default example directories:

```bash
python code/eval.py
```

Use the simplified dataset option to evaluate a named dataset from the repository structure:

```bash
python code/eval.py --dataset cat2
```

This sets:

- reference images: `data/dreambooth_original/<dataset>`
- generated images: `results/<dataset>`

Short options are also supported:

- `-r` / `--reference_dir`
- `-g` / `--generated_dir`
- `-d` / `--dataset`

Or provide custom paths:

```bash
python code/eval.py \
  -r path/to/reference/images \
  -g path/to/generated/images \
  --prompts_file path/to/prompts.txt \
  --output_dir path/to/output
```

When `--output_dir` is omitted, the script saves metric files by default to `results/metrics/<generated_dir_name>`.

Evaluation logs are written as timestamped JSON files, for example
`eval_log_20260428_143022.json`.

## Results

The `results/` directory contains generated image grids, prior-preservation
comparisons, and a metric comparison between reported DreamBooth baselines and
this reproduction. See `results/README.md` for the full result summary.

### Expression Modification

![Expression modification results](results/expression_modification.png)

### Viewpoint Modification

![Viewpoint modification results](results/viewpoint_modification.png)

### Background Modification

![Background modification results](results/background_modification.png)

### Outfit Modification

![Outfit modification results](results/outfit_modification.png)

### With and Without Prior Preservation

![With and without prior preservation](results/with_and_without_prior_preservation.png)

### Quantitative Metrics

![DreamBooth metric comparison](results/metrics.png)
