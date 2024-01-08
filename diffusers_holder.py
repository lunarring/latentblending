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

import torch
import numpy as np
import warnings

from typing import Optional
from utils import interpolate_spherical
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)


class DiffusersHolder():
    def __init__(self, pipe):
        # Base settings
        self.negative_prompt = ""
        self.guidance_scale = 5.0
        self.num_inference_steps = 30

        # Check if valid pipe
        self.pipe = pipe
        self.device = str(pipe._execution_device)
        self.init_types()

        self.width_latent = self.pipe.unet.config.sample_size
        self.height_latent = self.pipe.unet.config.sample_size
        self.width_img = self.width_latent  * self.pipe.vae_scale_factor
        self.height_img = self.height_latent  * self.pipe.vae_scale_factor

    def init_types(self):
        assert hasattr(self.pipe, "__class__"), "No valid diffusers pipeline found."
        assert hasattr(self.pipe.__class__, "__name__"), "No valid diffusers pipeline found."
        if self.pipe.__class__.__name__ == 'StableDiffusionXLPipeline':
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            self.use_sd_xl = True
            prompt_embeds, _, _, _ = self.pipe.encode_prompt("test")
        else:
            self.use_sd_xl = False
            prompt_embeds = self.pipe._encode_prompt("test", self.device, 1, True)
        self.dtype = prompt_embeds.dtype

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        if self.use_sd_xl:
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    def set_dimensions(self, size_output):
        s = self.pipe.vae_scale_factor
        if size_output is None:
            width = self.pipe.unet.config.sample_size
            height = self.pipe.unet.config.sample_size
        else:
            width, height = size_output
        self.width_img = int(round(width / s) * s)
        self.width_latent = int(self.width_img / s)
        self.height_img = int(round(height / s) * s)
        self.height_latent = int(self.height_img / s)
        print(f"set_dimensions to width={width} and height={height}")

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
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=self.negative_prompt,
            negative_prompt_2=self.negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=False,
        )
        return prompt_embeds

    def get_noise(self, seed=420):
        
        latents = self.pipe.prepare_latents(
            1,
            self.pipe.unet.config.in_channels,
            self.height_img,
            self.width_img,
            torch.float16,
            self.pipe._execution_device,
            torch.Generator(device=self.device).manual_seed(int(seed)),
            None,
        )
        
        return latents
        
        
        # H = self.height_latent
        # W = self.width_latent
        # C = self.pipe.unet.config.in_channels
        # generator = torch.Generator(device=self.device).manual_seed(int(seed))
        # latents = torch.randn((1, C, H, W), generator=generator, dtype=self.dtype, device=self.device)
        # if self.use_sd_xl:
        #     latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def latent2image(
            self,
            latents: torch.FloatTensor,
            output_type="pil"):
        r"""
        Returns an image provided a latent representation from diffusion.
        Args:
            latents: torch.FloatTensor
                Result of the diffusion process.
            output_type: "pil" or "np"
        """
        assert output_type in ["pil", "np"]
            
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
    
        if needs_upcasting:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)
    
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
    
        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)
    
        image = self.pipe.image_processor.postprocess(image, output_type=output_type)[0]
        
        return image
        
        # if output_type == "np":
        #     return np.asarray(image)
        # else:
        #     return image
    
    
        # # xxx
        # if self.use_sd_xl:
        #     # make sure the VAE is in float32 mode, as it overflows in float16
        #     self.pipe.vae.to(dtype=torch.float32)

        #     use_torch_2_0_or_xformers = isinstance(
        #         self.pipe.vae.decoder.mid_block.attentions[0].processor,
        #         (
        #             AttnProcessor2_0,
        #             XFormersAttnProcessor,
        #             LoRAXFormersAttnProcessor,
        #             LoRAAttnProcessor2_0,
        #         ),
        #     )
        #     # if xformers or torch_2_0 is used attention block does not need
        #     # to be in float32 which can save lots of memory
        #     if use_torch_2_0_or_xformers:
        #         self.pipe.vae.post_quant_conv.to(latents.dtype)
        #         self.pipe.vae.decoder.conv_in.to(latents.dtype)
        #         self.pipe.vae.decoder.mid_block.to(latents.dtype)
        #     else:
        #         latents = latents.float()

        # image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        # image = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]
        # if output_type == "np":
        #     return np.asarray(image)
        # else:
        #     return image

    def prepare_mixing(self, mixing_coeffs, list_latents_mixing):
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = (1 + self.num_inference_steps) * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps, f"len(mixing_coeffs) {len(mixing_coeffs)} != self.num_inference_steps {self.num_inference_steps}"
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")
        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps, f"len(list_latents_mixing) {len(list_latents_mixing)} != self.num_inference_steps {self.num_inference_steps}"
        return list_mixing_coeffs

    @torch.no_grad()
    def run_diffusion(
            self,
            text_embeddings: torch.FloatTensor,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):

        if self.pipe.__class__.__name__ == 'StableDiffusionXLPipeline':
            return self.run_diffusion_sd_xl(text_embeddings, latents_start, idx_start, list_latents_mixing, mixing_coeffs, return_image)
        elif self.pipe.__class__.__name__ == 'StableDiffusionPipeline':
            return self.run_diffusion_sd12x(text_embeddings, latents_start, idx_start, list_latents_mixing, mixing_coeffs, return_image)
        elif self.pipe.__class__.__name__ == 'StableDiffusionControlNetPipeline':
            pass

    @torch.no_grad()
    def run_diffusion_sd12x(
            self,
            text_embeddings: torch.FloatTensor,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):

        list_mixing_coeffs = self.prepare_mixing()

        do_classifier_free_guidance = self.guidance_scale > 1.0

        # accomodate different sd model types
        self.pipe.scheduler.set_timesteps(self.num_inference_steps - 1, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        if len(timesteps) != self.num_inference_steps:
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.pipe.scheduler.timesteps

        latents = latents_start.clone()
        list_latents_out = []

        for i, t in enumerate(timesteps):
            # Set the right starting latents
            if i == idx_start:
                latents = latents_start.clone()
            # Mix latents
            if i > 0 and list_mixing_coeffs[i] > 0:
                latents_mixtarget = list_latents_mixing[i - 1].clone()
                latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])

            if i < idx_start:
                list_latents_out.append(latents)
                
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
    def run_diffusion_controlnet(
            self,
            conditioning: list,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):

        prompt_embeds = conditioning[0]
        image = conditioning[1]
        list_mixing_coeffs = self.prepare_mixing()

        controlnet = self.pipe.controlnet
        control_guidance_start = [0.0]
        control_guidance_end = [1.0]
        guess_mode = False
        num_images_per_prompt = 1
        batch_size = 1
        eta = 0.0
        controlnet_conditioning_scale = 1.0
       
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]

        # 2. Define call parameters
        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = self.guidance_scale > 1.0

        # 4. Prepare image
        image = self.pipe.prepare_image(
            image=image,
            width=None,
            height=None,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=self.device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]

        # 5. Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # 6. Prepare latent variables
        generator = torch.Generator(device=self.device).manual_seed(int(420))
        latents = latents_start.clone()
        list_latents_out = []

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

        # 8. Denoising loop
        for i, t in enumerate(timesteps):
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
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            # predict the noise residual
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # Append latents
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
        return_image: Optional[bool] = False,
        seed=420,
        **kwargs,
    ):
        
        timesteps = None
        denoising_end = None
        guidance_scale = 0.0
        negative_prompt = None
        negative_prompt_2 = None
        num_images_per_prompt = 1
        eta = 0.0
        generator = None
        latents = None
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        ip_adapter_image = None
        output_type = "pil"
        return_dict = True
        cross_attention_kwargs = None
        guidance_rescale = 0.0
        original_size = None
        crops_coords_top_left = (0, 0)
        target_size = None
        negative_original_size = None
        negative_crops_coords_top_left = (0, 0)
        negative_target_size = None
        clip_skip = None
        callback_on_step_end = None
        callback_on_step_end_tensor_inputs = ["latents"]
        

        # 0. Default height and width to unet
        height = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        width = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        list_mixing_coeffs = self.prepare_mixing(mixing_coeffs, list_latents_mixing)

        original_size = (height, width)
        target_size = (height, width)

        # 1. (skipped) Check inputs. Raise error if not correct
        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._denoising_end = denoising_end
        self.pipe._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = self.pipe._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = text_embeddings

        # 4. Prepare timesteps
        timesteps, self.num_inference_steps = retrieve_timesteps(self.pipe.scheduler, self.num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        latents = latents_start.clone()
        list_latents_out = []
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(torch.Generator(device=self.device).manual_seed(int(0)), eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.pipe.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.pipe.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.pipe.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - self.num_inference_steps * self.pipe.scheduler.order, 0)



        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.pipe._num_timesteps = len(timesteps)
        
        
        for i, t in enumerate(timesteps):
            # Set the right starting latents
            # Write latents out and skip
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
            latent_model_input = torch.cat([latents] * 2) if self.pipe.do_classifier_free_guidance else latents

            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None:
                added_cond_kwargs["image_embeds"] = image_embeds
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.pipe.do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

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
    from PIL import Image
    #%% 
    from diffusers import AutoencoderTiny
    # pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_model_name_or_path = "stabilityai/sdxl-turbo"
    pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipe.to('cuda')    # xxx
    
    #%
    pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    pipe.vae = pipe.vae.cuda()
    #%%

    # # xxx
    # self.set_dimensions((512, 512))
    # self.set_num_inference_steps(4)
    # self.guidance_scale = 2
    # # self.set_dimensions(1536, 1024)
    # latents_start = torch.randn((1,4,64//1,64)).half().cuda()
    # # latents_start = self.get_noise()
    # list_latents_1 = self.run_diffusion_sd_xl(text_embeddings, latents_start)
    # img_orig = self.latent2image(list_latents_1[-1])
    
    #%%
    
    self = DiffusersHolder(pipe)
    num_inference_steps = 4
    self.set_num_inference_steps(num_inference_steps)
    latents_start = self.get_noise()
    guidance_scale = 0
    self.guidance_scale = 0
    
    #% get embeddings1
    prompt1 = "Photo of a colorful landscape with a blue sky with clouds"
    text_embeddings1 = self.get_text_embedding(prompt1)
    prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = text_embeddings1
    
    #% get embeddings2
    prompt2 = "Photo of a tree"
    text_embeddings2 = self.get_text_embedding(prompt2)
    prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = text_embeddings2
    
    latents1 = self.run_diffusion_sd_xl(text_embeddings1, latents_start, idx_start=0, return_image=False)
    
    img1 = self.run_diffusion_sd_xl(text_embeddings1, latents_start, idx_start=0, return_image=True)
    img1B = self.run_diffusion_sd_xl(text_embeddings1, latents_start, idx_start=0, return_image=True)
    
    
    
    # latents2 = self.run_diffusion_sd_xl(text_embeddings2, latents_start, idx_start=0, return_image=False)
    
    
    # # check if brings same image if restarted
    # img1_return = self.run_diffusion_sd_xl(text_embeddings1, latents1[idx_mix-1], idx_start=idx_start, return_image=True)
    
    # mix latents
    #%%
    idx_mix = 2
    fract=0.8
    latents_start_mixed = interpolate_spherical(latents1[idx_mix-1], latents2[idx_mix-1], fract)
    prompt_embeds = interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
    pooled_prompt_embeds = interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
    negative_prompt_embeds = negative_prompt_embeds1
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds1
    text_embeddings_mix = [prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds]
    
    self.run_diffusion_sd_xl(text_embeddings_mix, latents_start_mixed, idx_start=idx_start, return_image=True)
    

    

