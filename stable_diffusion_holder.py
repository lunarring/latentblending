# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import warnings
import torch
from PIL import Image
import torch
from typing import Optional
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import repeat, rearrange
from utils import interpolate_spherical


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def make_batch_superres(
        image,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device),
                         "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


class StableDiffusionHolder:
    def __init__(self,
                 fp_ckpt: str = None,
                 fp_config: str = None,
                 num_inference_steps: int = 30,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 device: str = None,
                 precision: str = 'autocast',
                 ):
        r"""
        Initializes the stable diffusion holder, which contains the models and sampler.
        Args:
            fp_ckpt: File pointer to the .ckpt model file
            fp_config: File pointer to the .yaml config file
            num_inference_steps: Number of diffusion iterations. Will be overwritten by latent blending.
            height: Height of the resulting image.
            width: Width of the resulting image.
            device: Device to run the model on.
            precision: Precision to run the model on.
        """
        self.seed = 42
        self.guidance_scale = 5.0

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.precision = precision
        self.init_model(fp_ckpt, fp_config)

        self.f = 8  # downsampling factor, most often 8 or 16"
        self.C = 4
        self.ddim_eta = 0
        self.num_inference_steps = num_inference_steps

        if height is None and width is None:
            self.init_auto_res()
        else:
            assert height is not None, "specify both width and height"
            assert width is not None, "specify both width and height"
            self.height = height
            self.width = width

        self.negative_prompt = [""]

    def init_model(self, fp_ckpt, fp_config):
        r"""Loads the models and sampler.
        """

        assert os.path.isfile(fp_ckpt), f"Your model checkpoint file does not exist: {fp_ckpt}"
        self.fp_ckpt = fp_ckpt

        # Auto init the config?
        if fp_config is None:
            fn_ckpt = os.path.basename(fp_ckpt)
            if 'depth' in fn_ckpt:
                fp_config = 'configs/v2-midas-inference.yaml'
            elif 'upscaler' in fn_ckpt:
                fp_config = 'configs/x4-upscaling.yaml'
            elif '512' in fn_ckpt:
                fp_config = 'configs/v2-inference.yaml'
            elif '768' in fn_ckpt:
                fp_config = 'configs/v2-inference-v.yaml'
            elif 'v1-5' in fn_ckpt:
                fp_config = 'configs/v1-inference.yaml'
            else:
                raise ValueError("auto detect of config failed. please specify fp_config manually!")

            assert os.path.isfile(fp_config), "Auto-init of the config file failed. Please specify manually."

        assert os.path.isfile(fp_config), f"Your config file does not exist: {fp_config}"

        config = OmegaConf.load(fp_config)

        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(torch.load(fp_ckpt)["state_dict"], strict=False)

        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)

    def init_auto_res(self):
        r"""Automatically set the resolution to the one used in training.
        """
        if '768' in self.fp_ckpt:
            self.height = 768
            self.width = 768
        else:
            self.height = 512
            self.width = 512
            
    def get_noise(self, seed, mode='standard'):
        r"""
        Helper function to get noise given seed.
        Args:
            seed: int
        """
        
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if mode == 'standard':
            shape_latents = [self.C, self.height // self.f, self.width // self.f]
            C, H, W = shape_latents
        elif mode == 'upscale':
            w = self.image1_lowres.size[0]
            h = self.image1_lowres.size[1]
            shape_latents = [self.model.channels, h, w]
            C, H, W = shape_latents
        return torch.randn((1, C, H, W), generator=generator, device=self.device)

    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported
        """

        if isinstance(negative_prompt, str):
            self.negative_prompt = [negative_prompt]
        else:
            self.negative_prompt = negative_prompt

        if len(self.negative_prompt) > 1:
            self.negative_prompt = [self.negative_prompt[0]]

    def get_text_embedding(self, prompt):
        c = self.model.get_learned_conditioning(prompt)
        return c

    @torch.no_grad()
    def get_cond_upscaling(self, image, text_embedding, noise_level):
        r"""
        Initializes the conditioning for the x4 upscaling model.
        """
        image = pad_image(image)  # resize to integer multiple of 32
        w, h = image.size
        noise_level = torch.Tensor(1 * [noise_level]).to(self.sampler.model.device).long()
        batch = make_batch_superres(image, txt="placeholder", device=self.device, num_samples=1)

        x_augment, noise_level = make_noise_augmentation(self.model, batch, noise_level)

        cond = {"c_concat": [x_augment], "c_crossattn": [text_embedding], "c_adm": noise_level}
        # uncond cond
        uc_cross = self.model.get_unconditional_conditioning(1, "")
        uc_full = {"c_concat": [x_augment], "c_crossattn": [uc_cross], "c_adm": noise_level}
        return cond, uc_full

    @torch.no_grad()
    def run_diffusion_standard(
            self,
            text_embeddings: torch.FloatTensor,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            spatial_mask=None,
            return_image: Optional[bool] = False):
        r"""
        Diffusion standard version.
        Args:
            text_embeddings: torch.FloatTensor
                Text embeddings used for diffusion
            latents_for_injection: torch.FloatTensor or list
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            mixing_coeff:
                mixing coefficients for latent blending
            spatial_mask:
                experimental feature for enforcing pixels from list_latents_mixing
            return_image: Optional[bool]
                Optionally return image directly
        """
        # Asserts
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = self.num_inference_steps * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps

        precision_scope = autocast if self.precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            with self.model.ema_scope():
                if self.guidance_scale != 1.0:
                    uc = self.model.get_learned_conditioning(self.negative_prompt)
                else:
                    uc = None
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps - 1, ddim_eta=self.ddim_eta, verbose=False)
                latents = latents_start.clone()
                timesteps = self.sampler.ddim_timesteps
                time_range = np.flip(timesteps)
                total_steps = timesteps.shape[0]
                # Collect latents
                list_latents_out = []
                for i, step in enumerate(time_range):
                    # Set the right starting latents
                    if i < idx_start:
                        list_latents_out.append(None)
                        continue
                    elif i == idx_start:
                        latents = latents_start.clone()
                    # Mix latents
                    if i > 0 and list_mixing_coeffs[i] > 0:
                        latents_mixtarget = list_latents_mixing[i - 1].clone()
                        latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])

                    if spatial_mask is not None and list_latents_mixing is not None:
                        latents = interpolate_spherical(latents, list_latents_mixing[i - 1], 1 - spatial_mask)

                    index = total_steps - i - 1
                    ts = torch.full((1,), step, device=self.device, dtype=torch.long)
                    outs = self.sampler.p_sample_ddim(latents, text_embeddings, ts, index=index, use_original_steps=False,
                                                      quantize_denoised=False, temperature=1.0,
                                                      noise_dropout=0.0, score_corrector=None,
                                                      corrector_kwargs=None,
                                                      unconditional_guidance_scale=self.guidance_scale,
                                                      unconditional_conditioning=uc,
                                                      dynamic_threshold=None)
                    latents, pred_x0 = outs
                    list_latents_out.append(latents.clone())
                if return_image:
                    return self.latent2image(latents)
                else:
                    return list_latents_out

    @torch.no_grad()
    def run_diffusion_upscaling(
            self,
            cond,
            uc_full,
            latents_start: torch.FloatTensor,
            idx_start: int = -1,
            list_latents_mixing: list = None,
            mixing_coeffs: float = 0.0,
            return_image: Optional[bool] = False):
        r"""
        Diffusion upscaling version.
        """

        # Asserts
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = self.num_inference_steps * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps

        precision_scope = autocast if self.precision == "autocast" else nullcontext
        h = uc_full['c_concat'][0].shape[2]
        w = uc_full['c_concat'][0].shape[3]
        with precision_scope("cuda"):
            with self.model.ema_scope():

                shape_latents = [self.model.channels, h, w]
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps - 1, ddim_eta=self.ddim_eta, verbose=False)
                C, H, W = shape_latents
                size = (1, C, H, W)
                b = size[0]
                latents = latents_start.clone()
                timesteps = self.sampler.ddim_timesteps
                time_range = np.flip(timesteps)
                total_steps = timesteps.shape[0]
                # collect latents
                list_latents_out = []
                for i, step in enumerate(time_range):
                    # Set the right starting latents
                    if i < idx_start:
                        list_latents_out.append(None)
                        continue
                    elif i == idx_start:
                        latents = latents_start.clone()
                    # Mix the latents.
                    if i > 0 and list_mixing_coeffs[i] > 0:
                        latents_mixtarget = list_latents_mixing[i - 1].clone()
                        latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])
                    # print(f"diffusion iter {i}")
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=self.device, dtype=torch.long)
                    outs = self.sampler.p_sample_ddim(latents, cond, ts, index=index, use_original_steps=False,
                                                      quantize_denoised=False, temperature=1.0,
                                                      noise_dropout=0.0, score_corrector=None,
                                                      corrector_kwargs=None,
                                                      unconditional_guidance_scale=self.guidance_scale,
                                                      unconditional_conditioning=uc_full,
                                                      dynamic_threshold=None)
                    latents, pred_x0 = outs
                    list_latents_out.append(latents.clone())

                if return_image:
                    return self.latent2image(latents)
                else:
                    return list_latents_out

    @torch.no_grad()
    def latent2image(
            self,
            latents: torch.FloatTensor):
        r"""
        Returns an image provided a latent representation from diffusion.
        Args:
            latents: torch.FloatTensor
                Result of the diffusion process.
        """
        x_sample = self.model.decode_first_stage(latents)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255 * x_sample[0, :, :].permute([1, 2, 0]).cpu().numpy()
        image = x_sample.astype(np.uint8)
        return image
