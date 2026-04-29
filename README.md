# dreambooth

CS 5782 Final Project, reproduces the paper DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

## Training DreamBooth LoRA

The training entry point is `code/train_dreambooth.py`. It fine-tunes Stable
Diffusion with LoRA adapters on the UNet and uses the DreamBooth objective:

```text
loss = reconstruction_loss + prior_loss_weight * prior_preservation_loss
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run training with `accelerate`:

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
  --output_dir outputs/cat_lora
```

`instance_data_dir` should point to the specific subject images, such as
`data/dreambooth_original/cat2`. These images are trained with the instance prompt, such as
`"a sks cat"`, and contribute the reconstruction loss.

`class_data_dir` should point to generic images from the same class, such as
`data/class_cat`. If the directory has fewer than `--num_class_images` images,
the script generates the missing class images from the base Stable Diffusion
model using `--class_prompt`. These images contribute the prior preservation
loss, which helps the model keep the general meaning of prompts like `"a cat"`.

The script saves LoRA weights to `--output_dir`.

## Evaluating Generated Images

The evaluation entry point is `code/eval.py`. It evaluates generated images using CLIP-I, CLIP-T, and DINO similarity metrics to assess image quality and fidelity.

Run evaluation with default parameters (uses `data/dreambooth_original/cat2` as reference and `results/cat2` as generated images, derives prompts from filenames):

```bash
python code/eval.py
```

For custom directories and prompts:

```bash
python code/eval.py \
  --reference_dir path/to/reference/images \
  --generated_dir path/to/generated/images \
  --prompts_file path/to/prompts.txt \
  --output_dir path/to/output
```

The script computes CLIP-I (image-to-reference similarity), CLIP-T (image-to-text similarity), and DINO (feature similarity) scores. Results are saved as a JSON file in the output directory with a timestamped filename, e.g., `eval_log_20260428_143022.json`.
