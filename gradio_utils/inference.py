from __future__ import annotations

import gc
import pathlib
import sys
import tempfile

import gradio as gr
import imageio
import PIL.Image
import torch
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from huggingface_hub import ModelCard

from ..tuneavideo.models.unet import UNet3DConditionModel
from ..tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline


class InferencePipeline:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.pipe = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_id = None

    def clear(self) -> None:
        self.model_id = None
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def check_if_model_is_local(model_id: str) -> bool:
        return pathlib.Path(model_id).exists()

    @staticmethod
    def get_model_card(model_id: str,
                       hf_token: str | None = None) -> ModelCard:
        if InferencePipeline.check_if_model_is_local(model_id):
            card_path = (pathlib.Path(model_id) / 'README.md').as_posix()
        else:
            card_path = model_id
        return ModelCard.load(card_path, token=hf_token)

    @staticmethod
    def get_base_model_info(model_id: str, hf_token: str | None = None) -> str:
        card = InferencePipeline.get_model_card(model_id, hf_token)
        return card.data.base_model

    def load_pipe(self, model_id: str) -> None:
        if model_id == self.model_id:
            return
        base_model_id = self.get_base_model_info(model_id, self.hf_token)
        unet = UNet3DConditionModel.from_pretrained(
            model_id,
            subfolder='unet',
            torch_dtype=torch.float16,
            use_auth_token=self.hf_token)
        pipe = TuneAVideoPipeline.from_pretrained(base_model_id,
                                                  unet=unet,
                                                  torch_dtype=torch.float16,
                                                  use_auth_token=self.hf_token)
        pipe = pipe.to(self.device)
        if is_xformers_available():
            pipe.unet.enable_xformers_memory_efficient_attention()
        self.pipe = pipe
        self.model_id = model_id  # type: ignore

    def run(
        self,
        model_id: str,
        prompt: str,
        video_length: int,
        fps: int,
        seed: int,
        n_steps: int,
        guidance_scale: float,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        self.load_pipe(model_id)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt,
            video_length=video_length,
            width=512,
            height=512,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )  # type: ignore

        frames = rearrange(out.videos[0], 'c t h w -> t h w c')
        frames = (frames * 255).to(torch.uint8).numpy()

        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        writer = imageio.get_writer(out_file.name, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return out_file.name
