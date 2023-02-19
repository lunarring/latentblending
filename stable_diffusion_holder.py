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

import os, sys
dp_git = "/home/lugo/git/"
sys.path.append(os.path.join(dp_git,'garden4'))
sys.path.append('util')
import torch
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import subprocess
import warnings
import torch
from tqdm.auto import tqdm
from PIL import Image
# import matplotlib.pyplot as plt
import torch
from movie_util import MovieSaver
import datetime
from typing import Callable, List, Optional, Union
import inspect
from threading import Thread
torch.set_grad_enabled(False)
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import repeat, rearrange
#%%


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded



def make_batch_inpaint(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def make_batch_superres(
        image,
        txt,
        device,
        num_samples=1,
    ):
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
                 precision: str='autocast',
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
        
        self.f = 8 #downsampling factor, most often 8 or 16",
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
            
        # Inpainting inits
        self.mask_empty = Image.fromarray(255*np.ones([self.width, self.height], dtype=np.uint8))
        self.image_empty = Image.fromarray(np.zeros([self.width, self.height, 3], dtype=np.uint8))
        
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
            elif 'inpain' in fn_ckpt:
                fp_config = 'configs/v2-inpainting-inference.yaml'
            elif 'upscaler' in fn_ckpt:
                fp_config = 'configs/x4-upscaling.yaml' 
            elif '512' in fn_ckpt:
                fp_config = 'configs/v2-inference.yaml' 
            elif '768'in fn_ckpt:
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
            list_latents_mixing = None, 
            mixing_coeffs = 0.0,
            spatial_mask = None,
            return_image: Optional[bool] = False,
        ):
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
                # FIXME spatial_mask
            return_image: Optional[bool]
                Optionally return image directly
            
        """
 
        # Asserts
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = self.num_inference_steps*[mixing_coeffs]
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
    
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps-1, ddim_eta=self.ddim_eta, verbose=False)
                
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
                    if i > 0 and list_mixing_coeffs[i]>0:
                        latents_mixtarget = list_latents_mixing[i-1].clone()
                        latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])
                        
                    if spatial_mask is not None and list_latents_mixing is not None:
                        latents = interpolate_spherical(latents, list_latents_mixing[i-1], 1-spatial_mask)
                        # latents[:,:,-15:,:] = latents_mixtarget[:,:,-15:,:]
                    
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
            list_latents_mixing = None, 
            mixing_coeffs = 0.0,
            return_image: Optional[bool] = False
        ):
        r"""
        Diffusion upscaling version. 
        # FIXME
        Args:
            ??
            latents_for_injection: torch.FloatTensor 
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            return_image: Optional[bool]
                Optionally return image directly
        """
 
        # Asserts
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = self.num_inference_steps*[mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")
        
        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps
        
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        generator = torch.Generator(device=self.device).manual_seed(int(self.seed))
        
        h = uc_full['c_concat'][0].shape[2]        
        w = uc_full['c_concat'][0].shape[3]  
        
        with precision_scope("cuda"):
            with self.model.ema_scope():

                shape_latents = [self.model.channels, h, w]
    
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps-1, ddim_eta=self.ddim_eta, verbose=False)
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
                    if i > 0 and list_mixing_coeffs[i]>0:
                        latents_mixtarget = list_latents_mixing[i-1].clone()
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
    def run_diffusion_inpaint(
            self, 
            text_embeddings: torch.FloatTensor, 
            latents_for_injection: torch.FloatTensor = None, 
            idx_start: int = -1, 
            idx_stop: int = -1, 
            return_image: Optional[bool] = False
        ):
        r"""
        Runs inpaint-based diffusion. Returns a list of latents that were computed.
        Adaptations allow to supply 
        a) starting index for diffusion
        b) stopping index for diffusion
        c) latent representations that are injected at the starting index
        Furthermore the intermittent latents are collected and returned.
        
        Adapted from diffusers (https://github.com/huggingface/diffusers)
        Args:
            text_embeddings: torch.FloatTensor
                Text embeddings used for diffusion
            latents_for_injection: torch.FloatTensor 
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            idx_stop: int
                Index of the diffusion process end.
            return_image: Optional[bool]
                Optionally return image directly
                
        """ 
        
        if latents_for_injection is None:
            do_inject_latents = False
        else:
            do_inject_latents = True
        
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        generator = torch.Generator(device=self.device).manual_seed(int(self.seed))

        with precision_scope("cuda"):
            with self.model.ema_scope():
                    
                batch = make_batch_inpaint(self.image_source, self.mask_image, txt="willbereplaced", device=self.device, num_samples=1)
                c = text_embeddings
                c_cat = list()
                for ck in self.model.concat_keys:
                    cc = batch[ck].float()
                    if ck != self.model.masked_image_key:
                        bchw = [1, 4, self.height // 8, self.width // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = self.model.get_first_stage_encoding(self.model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)
    
                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    
                # uncond cond
                uc_cross = self.model.get_unconditional_conditioning(1, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    
                shape_latents = [self.model.channels, self.height // 8, self.width // 8]
                
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps-1, ddim_eta=0., verbose=False)
                # sampling
                C, H, W = shape_latents
                size = (1, C, H, W)
                
                device = self.model.betas.device
                b = size[0]
                latents = torch.randn(size, generator=generator, device=device)
    
                timesteps = self.sampler.ddim_timesteps
    
                time_range = np.flip(timesteps)
                total_steps = timesteps.shape[0]
                
                # collect latents
                list_latents_out = []
                for i, step in enumerate(time_range):
                    if do_inject_latents:
                        # Inject latent at right place
                        if i < idx_start:
                            continue
                        elif i == idx_start:
                            latents = latents_for_injection.clone()
                    
                    if i == idx_stop:
                        return list_latents_out
                    
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
    
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
            latents: torch.FloatTensor
        ):
        r"""
        Returns an image provided a latent representation from diffusion.
        Args:
            latents: torch.FloatTensor
                Result of the diffusion process. 
        """
        x_sample = self.model.decode_first_stage(latents)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255 * x_sample[0,:,:].permute([1,2,0]).cpu().numpy()
        image = x_sample.astype(np.uint8)
        return image

@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Slerp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
    
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
        
    return interp


if __name__ == "__main__":
    
            




    num_inference_steps = 20 # Number of diffusion interations
    
    # fp_ckpt = "../stable_diffusion_models/ckpt/768-v-ema.ckpt"
    # fp_config = '../stablediffusion/configs/stable-diffusion/v2-inference-v.yaml'
    
    # fp_ckpt= "../stable_diffusion_models/ckpt/512-inpainting-ema.ckpt"
    # fp_config = '../stablediffusion/configs//stable-diffusion/v2-inpainting-inference.yaml'
    
    fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_768-ema-pruned.ckpt"
    # fp_config = 'configs/v2-inference-v.yaml'

    
    self = StableDiffusionHolder(fp_ckpt, num_inference_steps=num_inference_steps)
    
    xxx
    
    #%%
    self.width = 1536
    self.height = 768
    prompt = "360 degree equirectangular, a huge rocky hill full of pianos and keyboards, musical instruments, cinematic, masterpiece 8 k, artstation"
    self.set_negative_prompt("out of frame, faces, rendering, blurry")
    te = self.get_text_embedding(prompt)
    
    img = self.run_diffusion_standard(te, return_image=True)
    Image.fromarray(img).show()
    
