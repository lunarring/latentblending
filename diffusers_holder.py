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
from torch import autocast
from contextlib import nullcontext
from utils import interpolate_spherical
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)


class DiffusersHolder():
    def __init__(self, pipe):
        # Base settings
        self.negative_prompt = ""
        self.guidance_scale = 5.0
        self.num_inference_steps = 30
        
        # Check if valid pipe
        self.pipe = pipe
        self.device = str(pipe._execution_device)
        self.init_type_pipe()
        self.init_dtype()
        
        self.width_latent = self.pipe.unet.config.sample_size
        self.height_latent = self.pipe.unet.config.sample_size
        
        

    def init_type_pipe(self):
        self.type_pipe = "StableDiffusionXLPipeline"
        if self.type_pipe == "StableDiffusionXLPipeline":
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            self.use_sd_xl = True
        else:
            self.use_sd_xl = False

    def init_dtype(self):
        if self.type_pipe == "StableDiffusionXLPipeline":
            prompt_embeds, _, _, _ = self.pipe.encode_prompt("test")
            self.dtype = prompt_embeds.dtype

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        if self.use_sd_xl:
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)


    def set_dimensions(self, width, height):
        s = self.pipe.vae_scale_factor
        if width is None:
            self.width_latent = self.pipe.unet.config.sample_size 
            self.width_img = self.width_latent * self.pipe.vae_scale_factor
        else:
            self.width_img = int(round(width / s) * s)
            self.width_latent = int(self.width_img / s)
            
        if height is None:
            self.height_latent = self.pipe.unet.config.sample_size 
            self.height_img = self.width_latent * self.pipe.vae_scale_factor
        else:
            self.height_img = int(round(height / s) * s)
            self.height_latent = int(self.height_img / s)
        

    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported
        """
        if isinstance(negative_prompt, str):
            self.negative_prompt = [negative_prompt]
        else:
            self.negative_prompt = negative_prompt

        if len(self.negative_prompt) > 1:
            self.negative_prompt = [self.negative_prompt[0]]

    def get_text_embedding(self, prompt, do_classifier_free_guidance=True):
        if self.use_sd_xl:
            pr_encoder = self.pipe.encode_prompt
        else:
            pr_encoder = self.pipe._encode_prompt

        prompt_embeds = pr_encoder(
            prompt,
            self.device,
            1,
            do_classifier_free_guidance,
            negative_prompt=self.negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
        )
        return prompt_embeds


    def get_noise(self, seed=420, mode=None):
        H = self.height_latent
        W = self.width_latent
        C = self.pipe.unet.config.in_channels
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = torch.randn((1, C, H, W), generator=generator, dtype=self.dtype, device=self.device)
        if self.use_sd_xl:
            latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

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
        if self.use_sd_xl:
            # make sure the VAE is in float32 mode, as it overflows in float16
            self.pipe.vae.to(dtype=torch.float32)

            use_torch_2_0_or_xformers = isinstance(
                self.pipe.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    LoRAXFormersAttnProcessor,
                    LoRAAttnProcessor2_0,
                ),
            )
            # if xformers or torch_2_0 is used attention block does not need
            # to be in float32 which can save lots of memory
            if use_torch_2_0_or_xformers:
                self.pipe.vae.post_quant_conv.to(latents.dtype)
                self.pipe.vae.decoder.conv_in.to(latents.dtype)
                self.pipe.vae.decoder.mid_block.to(latents.dtype)
            else:
                latents = latents.float()

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])
        
        
        
        return np.asarray(image[0])

    @torch.no_grad()
    def run_diffusion_standard(
            self,
            text_embeddings: torch.FloatTensor,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):
        
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = (1+self.num_inference_steps) * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps, f"len(mixing_coeffs) {len(mixing_coeffs)} != self.num_inference_steps {self.num_inference_steps}"
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps, f"len(list_latents_mixing) {len(list_latents_mixing)} != self.num_inference_steps {self.num_inference_steps}"
            
        do_classifier_free_guidance = self.guidance_scale > 1.0
        
        # diffusers bit wiggly
        self.pipe.scheduler.set_timesteps(self.num_inference_steps-1, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        if len(timesteps) != self.num_inference_steps:
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.pipe.scheduler.timesteps
        
        latents = latents_start.clone()
        list_latents_out = []

        num_warmup_steps = len(timesteps) - self.num_inference_steps * self.pipe.scheduler.order
        for i, t in enumerate(timesteps):
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
                
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
        
            # predict the noise residual
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            list_latents_out.append(latents.clone())
        
        if return_image:
            return self.latent2image(latents)
        else:
            return list_latents_out

    @torch.no_grad()
    def run_diffusion_sd_xl(
            self,
            text_embeddings: list,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):
        
        # prompt = "photo of a house"
        # self.num_inference_steps = 50
        # mixing_coeffs= 0.0
        # idx_start= 0
        # latents_start = self.get_noise()
        # text_embeddings = self.pipe.encode_prompt(
        #     prompt,
        #     self.device,
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=True,
        #     negative_prompt="",
        #     prompt_embeds=None,
        #     negative_prompt_embeds=None,
        #     pooled_prompt_embeds=None,
        #     negative_pooled_prompt_embeds=None,
        #     lora_scale=None,
        # )
        

        # 0. Default height and width to unet
        original_size = (1024, 1024)  # FIXME
        crops_coords_top_left = (0, 0) # FIXME
        target_size = original_size
        batch_size = 1
        eta = 0.0
        num_images_per_prompt = 1
        cross_attention_kwargs = None
        generator = torch.Generator(device=self.device)  # dummy generator
        do_classifier_free_guidance = self.guidance_scale > 1.0
        
        # 1. Check inputs. Raise error if not correct & 2. Define call parameters
        # FIXME see if check_inputs use
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = (1+self.num_inference_steps) * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps, f"len(mixing_coeffs) {len(mixing_coeffs)} != self.num_inference_steps {self.num_inference_steps}"
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps, f"len(list_latents_mixing) {len(list_latents_mixing)} != self.num_inference_steps {self.num_inference_steps}"

        # 3. Encode input prompt (already encoded outside bc of mixing, just split here)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = text_embeddings

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        latents = latents_start.clone()
        list_latents_out = []

        # 6. Prepare extra step kwargs. usedummy generator
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)  # dummy

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        for i, t in enumerate(timesteps):
            # Set the right starting latents
            if i < idx_start:
                list_latents_out.append(None)
                continue
            elif i == idx_start:
                latents = latents_start.clone()



            # Mix latents for crossfeeding
            if i > 0 and list_mixing_coeffs[i] > 0:
                latents_mixtarget = list_latents_mixing[i - 1].clone()
                latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])


            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # Always scale latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # FIXME guidance_rescale disabled

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # Append latents
            list_latents_out.append(latents.clone())
        
        if return_image:
            return self.latent2image(latents)
        else:
            return list_latents_out




#%%

if __name__ == "__main__":
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-0.9"
    pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipe.to('cuda')
    # xxx
    self = DiffusersHolder(pipe)
    # xxx
    self.set_num_inference_steps(50)
    self.set_dimensions(1536, 1024)
    prompt = "photo of a beautiful cherry forest covered in white flowers, ambient light, very detailed, magic"
    text_embeddings = self.get_text_embedding(prompt)
    generator = torch.Generator(device=self.device).manual_seed(int(420))
    latents_start = self.get_noise()
    list_latents_1 = self.run_diffusion_sd_xl(text_embeddings, latents_start)
    img_orig = self.latent2image(list_latents_1[-1])
    
    
    
    # %%
    
    """
    OPEN
        - other examples
        - kill upscaling? or keep?
        - cleanup
            - ldh
            - sdh class
            - diffusion holder
            - check linting
            - check docstrings
        - fix readme
    """    
    
    
   
