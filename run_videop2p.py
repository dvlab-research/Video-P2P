# Adapted from https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
from einops import rearrange

from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

import cv2
import argparse
from omegaconf import OmegaConf

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# need to adjust sometimes
mask_th = (.3, .3)

def main(
    pretrained_model_path: str,
    image_path: str,
    prompt: str,
    prompts: Tuple[str],
    eq_params: Dict,
    save_name: str,
    is_word_swap: bool,
    blend_word: Tuple[str] = None,
    cross_replace_steps: float = 0.2,
    self_replace_steps: float = 0.5,
    video_len: int = 8,
    fast: bool = False,
    mixed_precision: str = 'fp32',
):
    output_folder = os.path.join(pretrained_model_path, 'results')
    if fast:
        save_name_1 = os.path.join(output_folder, 'inversion_fast.gif')
        save_name_2 = os.path.join(output_folder, '{}_fast.gif'.format(save_name))
    else:
        save_name_1 = os.path.join(output_folder, 'inversion.gif')
        save_name_2 = os.path.join(output_folder, '{}.gif'.format(save_name))
    if blend_word:
        blend_word = (((blend_word[0],), (blend_word[1],)))
    eq_params = dict(eq_params)
    prompts = list(prompts)
    cross_replace_steps = {'default_': cross_replace_steps,}

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    ).to(device, dtype=weight_dtype)
    unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet"
    ).to(device)
    ldm_stable = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer # Tokenizer of class: [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
    # A tokenizer breaks a stream of text into tokens, usually by looking for whitespace (tabs, spaces, new lines).

    class LocalBlend:
        
        def get_mask(self, maps, alpha, use_pool):
            k = 1
            maps = (maps * alpha).sum(-1).mean(2)
            if use_pool:
                maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
            mask = nnf.interpolate(maps, size=(x_t.shape[3:]))
            mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
            mask = mask.gt(self.th[1-int(use_pool)])
            mask = mask[:1] + mask
            return mask
        
        def __call__(self, x_t, attention_store, step):
            self.counter += 1
            if self.counter > self.start_blend:
                maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
                maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, MAX_NUM_WORDS) for item in maps]
                maps = torch.cat(maps, dim=2)
                mask = self.get_mask(maps, self.alpha_layers, True)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            return x_t
        
        def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
            alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    alpha_layers[i, :, :, :, :, ind] = 1
            
            if substruct_words is not None:
                substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
                for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                    if type(words_) is str:
                        words_ = [words_]
                    for word in words_:
                        ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                        substruct_layers[i, :, :, :, :, ind] = 1
                self.substruct_layers = substruct_layers.to(device)
            else:
                self.substruct_layers = None
            self.alpha_layers = alpha_layers.to(device)
            self.start_blend = int(start_blend * NUM_DDIM_STEPS)
            self.counter = 0 
            self.th=th
            
            
    class EmptyControl:
        
        
        def step_callback(self, x_t):
            return x_t
        
        def between_steps(self):
            return
        
        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            return attn

        
    class AttentionControl(abc.ABC):
        
        def step_callback(self, x_t):
            return x_t
        
        def between_steps(self):
            return
        
        @property
        def num_uncond_att_layers(self):
            return self.num_att_layers if LOW_RESOURCE else 0
        
        @abc.abstractmethod
        def forward (self, attn, is_cross: bool, place_in_unet: str):
            raise NotImplementedError

        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if LOW_RESOURCE:
                    attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return attn
        
        def reset(self):
            self.cur_step = 0
            self.cur_att_layer = 0

        def __init__(self):
            self.cur_step = 0
            self.num_att_layers = -1
            self.cur_att_layer = 0

    class SpatialReplace(EmptyControl):
        
        def step_callback(self, x_t):
            if self.cur_step < self.stop_inject:
                b = x_t.shape[0]
                x_t = x_t[:1].expand(b, *x_t.shape[1:])
            return x_t

        def __init__(self, stop_inject: float):
            super(SpatialReplace, self).__init__()
            self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
            

    class AttentionStore(AttentionControl):

        @staticmethod
        def get_empty_store():
            return {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [],  "mid_self": [],  "up_self": []}

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:
                self.step_store[key].append(attn)
            return attn

        def between_steps(self):
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

        def get_average_attention(self):
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
            return average_attention


        def reset(self):
            super(AttentionStore, self).reset()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

        def __init__(self):
            super(AttentionStore, self).__init__()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

            
    class AttentionControlEdit(AttentionStore, abc.ABC):
        
        def step_callback(self, x_t):
            if self.local_blend is not None:
                x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
            return x_t
            
        def replace_self_attention(self, attn_base, att_replace, place_in_unet):
            if att_replace.shape[2] <= 32 ** 2:
                attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
                return attn_base
            else:
                return att_replace
        
        @abc.abstractmethod
        def replace_cross_attention(self, attn_base, att_replace):
            raise NotImplementedError
        
        def forward(self, attn, is_cross: bool, place_in_unet: str):
            super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
            if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                h = attn.shape[0] // (self.batch_size)
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                attn_base, attn_repalce = attn[0], attn[1:]
                if is_cross:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                    attn[1:] = attn_repalce_new
                else:
                    attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            return attn
        
        def __init__(self, prompts, num_steps: int,
                    cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                    self_replace_steps: Union[float, Tuple[float, float]],
                    local_blend: Optional[LocalBlend]):
            super(AttentionControlEdit, self).__init__()
            self.batch_size = len(prompts)
            self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
            if type(self_replace_steps) is float:
                self_replace_steps = 0, self_replace_steps
            self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
            self.local_blend = local_blend

    class AttentionReplace(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        
        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[LocalBlend] = None):
            super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
            

    class AttentionRefine(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
            attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
            return attn_replace

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[LocalBlend] = None):
            super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
            self.mapper, alphas = self.mapper.to(device), alphas.to(device)
            self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


    class AttentionReweight(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            if self.prev_controller is not None:
                attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
            attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
            return attn_replace

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                    local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
            super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.equalizer = equalizer.to(device)
            self.prev_controller = controller


    def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                    Tuple[float, ...]]):
        if type(word_select) is int or type(word_select) is str:
            word_select = (word_select,)
        equalizer = torch.ones(1, 77)
        
        for word, val in zip(word_select, values):
            inds = ptp_utils.get_word_inds(text, word, tokenizer)
            equalizer[:, inds] = val
        return equalizer

    def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(8, 8, res, res, item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        out = out.sum(1) / out.shape[1]
        return out.cpu()


    def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None, mask_th=(.3,.3)) -> AttentionControlEdit:
        if blend_words is None:
            lb = None
        else:
            lb = LocalBlend(prompts, blend_word, th=mask_th)
        if is_replace_controller:
            controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
        else:
            controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
        if equilizer_params is not None:
            eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
            controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
        return controller


    def load_512_seq(image_path, left=0, right=0, top=0, bottom=0, n_sample_frame=video_len, sampling_rate=1):
        images = []
        for file in sorted(os.listdir(image_path)):
            images.append(file)
        n_images = len(images)
        sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if n_images < sequence_length:
            raise ValueError
        frames = []
        for index in range(n_sample_frame):
            p = os.path.join(image_path, images[index])
            image = np.array(Image.open(p).convert("RGB"))
            h, w, c = image.shape
            left = min(left, w-1)
            right = min(right, w - left - 1)
            top = min(top, h - left - 1)
            bottom = min(bottom, h - top - 1)
            image = image[top:h-bottom, left:w-right]
            h, w, c = image.shape
            if h < w:
                offset = (w - h) // 2
                image = image[:, offset:offset + h]
            elif w < h:
                offset = (h - w) // 2
                image = image[offset:offset + w]
            image = np.array(Image.fromarray(image).resize((512, 512)))
            frames.append(image)
        return np.stack(frames)


    class NullInversion:
        
        def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
            prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            return prev_sample
        
        def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
            timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
            alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
            beta_prod_t = 1 - alpha_prod_t
            next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
            next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            return next_sample
        
        def get_noise_pred_single(self, latents, t, context):
            noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
            return noise_pred

        def get_noise_pred(self, latents, t, is_forward=True, context=None):
            latents_input = torch.cat([latents] * 2)
            if context is None:
                context = self.context
            guidance_scale = 1 if is_forward else GUIDANCE_SCALE
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            if is_forward:
                latents = self.next_step(noise_pred, t, latents)
            else:
                latents = self.prev_step(noise_pred, t, latents)
            return latents

        @torch.no_grad()
        def latent2image(self, latents, return_type='np'):
            latents = 1 / 0.18215 * latents.detach()
            image = self.model.vae.decode(latents)['sample']
            if return_type == 'np':
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).astype(np.uint8)
            return image

        @torch.no_grad()
        def latent2image_video(self, latents, return_type='np'):
            latents = 1 / 0.18215 * latents.detach()
            latents = latents[0].permute(1, 0, 2, 3)
            image = self.model.vae.decode(latents)['sample']
            if return_type == 'np':
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).astype(np.uint8)
            return image

        @torch.no_grad()
        def image2latent(self, image):
            with torch.no_grad():
                if type(image) is Image:
                    image = np.array(image)
                if type(image) is torch.Tensor and image.dim() == 4:
                    latents = image
                else:
                    image = torch.from_numpy(image).float() / 127.5 - 1
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)
                    latents = self.model.vae.encode(image)['latent_dist'].mean
                    latents = latents * 0.18215
            return latents

        @torch.no_grad()
        def image2latent_video(self, image):
            with torch.no_grad():
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(device).to(device, dtype=weight_dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)
                latents = latents * 0.18215
            return latents

        @torch.no_grad()
        def init_prompt(self, prompt: str):
            uncond_input = self.model.tokenizer(
                [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt"
            )
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
            text_input = self.model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
            self.context = torch.cat([uncond_embeddings, text_embeddings])
            self.prompt = prompt

        @torch.no_grad()
        def ddim_loop(self, latent):
            uncond_embeddings, cond_embeddings = self.context.chunk(2)
            all_latent = [latent]
            latent = latent.clone().detach()
            for i in range(NUM_DDIM_STEPS):
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
                latent = self.next_step(noise_pred, t, latent)
                all_latent.append(latent)
            return all_latent

        @property
        def scheduler(self):
            return self.model.scheduler

        @torch.no_grad()
        def ddim_inversion(self, image):
            latent = self.image2latent_video(image)
            image_rec = self.latent2image_video(latent)
            ddim_latents = self.ddim_loop(latent)
            return image_rec, ddim_latents

        def null_optimization(self, latents, num_inner_steps, epsilon):
            uncond_embeddings, cond_embeddings = self.context.chunk(2)
            uncond_embeddings_list = []
            latent_cur = latents[-1]
            # bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
            for i in range(NUM_DDIM_STEPS):
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                t = self.model.scheduler.timesteps[i]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    # bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
                # for j in range(j + 1, num_inner_steps):
                #     bar.update()
                uncond_embeddings_list.append(uncond_embeddings[:1].detach())
                with torch.no_grad():
                    context = torch.cat([uncond_embeddings, cond_embeddings])
                    latent_cur = self.get_noise_pred(latent_cur, t, False, context)
            # bar.close()
            return uncond_embeddings_list
        
        def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
            self.init_prompt(prompt)
            ptp_utils.register_attention_control(self.model, None)
            image_gt = load_512_seq(image_path, *offsets)
            if verbose:
                print("DDIM inversion...")
            image_rec, ddim_latents = self.ddim_inversion(image_gt)
            if verbose:
                print("Null-text optimization...")
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
            return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

        def invert_(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
            self.init_prompt(prompt)
            ptp_utils.register_attention_control(self.model, None)
            image_gt = load_512_seq(image_path, *offsets)
            if verbose:
                print("DDIM inversion...")
            image_rec, ddim_latents = self.ddim_inversion(image_gt)
            if verbose:
                print("Null-text optimization...")
            return (image_gt, image_rec), ddim_latents[-1], None
        
        def __init__(self, model):
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                    set_alpha_to_one=False)
            self.model = model
            self.tokenizer = self.model.tokenizer
            self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
            self.prompt = None
            self.context = None

    null_inversion = NullInversion(ldm_stable)

    ###############
    # Custom APIs:

    ldm_stable.enable_xformers_memory_efficient_attention()

    if fast:
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert_(image_path, prompt, offsets=(0,0,0,0), verbose=True)
    else:
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

    ##### load uncond #####
    # uncond_embeddings_load = np.load(uncond_embeddings_path)
    # uncond_embeddings = []
    # for i in range(uncond_embeddings_load.shape[0]):
    #     uncond_embeddings.append(torch.from_numpy(uncond_embeddings_load[i]).to(device))
    #######################

    ##### save uncond #####
    # uncond_embeddings = torch.cat(uncond_embeddings)
    # uncond_embeddings = uncond_embeddings.cpu().numpy()
    #######################

    print("Start Video-P2P!")
    controller = make_controller(prompts, is_word_swap, cross_replace_steps, self_replace_steps, blend_word, eq_params, mask_th=mask_th)
    ptp_utils.register_attention_control(ldm_stable, controller)
    generator = torch.Generator(device=device)
    with torch.no_grad():
        sequence = ldm_stable(
            prompts,
            generator=generator,
            latents=x_t,
            uncond_embeddings_pre=uncond_embeddings,
            controller = controller,
            video_length=video_len,
            fast=fast,
        ).videos
    sequence1 = rearrange(sequence[0], "c t h w -> t h w c")
    sequence2 = rearrange(sequence[1], "c t h w -> t h w c")
    inversion = []
    videop2p = []
    for i in range(sequence1.shape[0]):
        inversion.append( Image.fromarray((sequence1[i] * 255).numpy().astype(np.uint8)) )
        videop2p.append( Image.fromarray((sequence2[i] * 255).numpy().astype(np.uint8)) )

    inversion[0].save(save_name_1, save_all=True, append_images=inversion[1:], optimize=False, loop=0, duration=250)
    videop2p[0].save(save_name_2, save_all=True, append_images=videop2p[1:], optimize=False, loop=0, duration=250)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/videop2p.yaml")
    parser.add_argument("--fast", action='store_true')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config), fast=args.fast)
