# Data

This directory contains the image data used for DreamBooth training,
prior preservation, and evaluation.

## Directory Layout

- `dreambooth_original/`: Subject images from the official Google DreamBooth
  dataset. Each subdirectory contains the few-shot instance images for one
  subject, such as `cat2`, `dog8`, `duck_toy`, or `robot_toy`.
- `generated/`: Class images generated from the base Stable Diffusion model for
  prior preservation. These folders are passed as `--class_data_dir` during
  training, for example `generated/cat-stable-diffusion-1_5`.
- `our_data/`: Custom subject images collected for this project. These can be
  used as `--instance_data_dir` in the same format as the original DreamBooth
  subjects.

## Training Usage

Use a subject folder as the instance data directory:

```bash
--instance_data_dir data/dreambooth_original/cat2
```

or, for custom data:

```bash
--instance_data_dir data/our_data/bottle
```

Use a matching generated class-image folder for prior preservation:

```bash
--class_data_dir data/generated/cat-stable-diffusion-1_5
```

If the class-image directory has fewer than the requested number of class
images, `code/train_dreambooth.py` can generate the missing images from the base
model using `--class_prompt`.

## Notes

The upstream DreamBooth data includes its own
`dreambooth_original/README.md`, `prompts_and_classes.txt`, and
`references_and_licenses.txt`. Check those files for the original dataset
source, subject prompts, class labels, and license information.
