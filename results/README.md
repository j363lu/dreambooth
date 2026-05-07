# DreamBooth Results

![DreamBooth generated examples](dreambooth.png)

This folder summarizes generated examples from the DreamBooth fine-tuned models.

## Expression Modification

![Expression modification results](expression_modification.png)

`expression_modification.png` shows generated images where the prompt modifies
the subject expression.

## Viewpoint Modification

![Viewpoint modification results](viewpoint_modification.png)

`viewpoint_modification.png` shows generated images where the prompt changes
the subject viewpoint.

## Background Modification

![Background modification results](background_modification.png)

`background_modification.png` shows generated images where the prompt places
the fine-tuned subject in different backgrounds.

## Outfit Modification

![Outfit modification results](outfit_modification.png)

`outfit_modification.png` shows generated images where the prompt changes the
subject outfit.

## With and Without Prior Preservation

![With and without prior preservation](with_and_without_prior_preservation.png)

`with_and_without_prior_preservation.png` compares generated images for the
prompt `"a robot toy"`. With prior preservation, the model can generate general
and diverse robot toy results. Without prior preservation, the model tends to
generate the fine-tuned instance.

## Quantitative Metrics

![DreamBooth metric comparison](metrics.png)

`metrics.png` compares the reported DreamBooth baseline metrics with our
reproduction results on DINO, CLIP-I, and CLIP-T.

- **DINO** measures image-to-image similarity using self-supervised visual
  features. Higher DINO scores indicate that generated images better preserve
  the visual identity and structure of the reference subject.
- **CLIP-I** measures image-to-image similarity in CLIP embedding space. Higher
  CLIP-I scores indicate stronger visual alignment between the generated images
  and the subject reference images.
- **CLIP-T** measures image-to-text similarity in CLIP embedding space. Higher
  CLIP-T scores indicate that generated images better match the text prompts.
