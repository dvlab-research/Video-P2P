from __future__ import annotations

import pathlib


def find_exp_dirs() -> list[str]:
    repo_dir = pathlib.Path(__file__).parent
    exp_root_dir = repo_dir / 'experiments'
    if not exp_root_dir.exists():
        return []
    exp_dirs = sorted(exp_root_dir.glob('*'))
    exp_dirs = [
        exp_dir for exp_dir in exp_dirs
        if (exp_dir / 'model_index.json').exists()
    ]
    return [path.relative_to(repo_dir).as_posix() for path in exp_dirs]


def save_model_card(
    save_dir: pathlib.Path,
    base_model: str,
    training_prompt: str,
    test_prompt: str = '',
    test_image_dir: str = '',
) -> None:
    image_str = ''
    if test_prompt and test_image_dir:
        image_paths = sorted((save_dir / test_image_dir).glob('*.gif'))
        if image_paths:
            image_path = image_paths[-1]
            rel_path = image_path.relative_to(save_dir)
            image_str = f'''## Samples
Test prompt: {test_prompt}

![{image_path.stem}]({rel_path})'''

    model_card = f'''---
license: creativeml-openrail-m
base_model: {base_model}
training_prompt: {training_prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- text-to-video
- tune-a-video
- video-p2p
inference: false
---

# Video-P2P - {save_dir.name}

## Model description
- Base model: [{base_model}](https://huggingface.co/{base_model})
- Training prompt: {training_prompt}

{image_str}

## Related papers:
- [Video-P2P](https://arxiv.org/abs/2303.04761): Video editing with cross-attention control
- [Tune-A-Video](https://arxiv.org/abs/2212.11565): One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation
- [Stable-Diffusion](https://arxiv.org/abs/2112.10752): High-Resolution Image Synthesis with Latent Diffusion Models
'''

    with open(save_dir / 'README.md', 'w') as f:
        f.write(model_card)
