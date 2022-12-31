# Copyright 2022 Lunar Ring. All rights reserved.
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
import matplotlib.pyplot as plt
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
from einops import repeat


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def make_batch_sd(
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

class StableDiffusionHolder:
    def __init__(self, 
                 fp_ckpt: str = None, 
                 fp_config: str = None,
                 device: str = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 30, 
                 precision: str='autocast',
                 ):
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
        
        
    def init_model(self, fp_ckpt, fp_config):
        assert os.path.isfile(fp_ckpt), f"Your model checkpoint file does not exist: {fp_ckpt}"
        assert os.path.isfile(fp_config), f"Your config file does not exist: {fp_config}"
        config = OmegaConf.load(fp_config)
        self.model = load_model_from_config(config, fp_ckpt)

        
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)
        self.fp_ckpt = fp_ckpt
        

            
    def init_auto_res(self):
        r"""Automatically set the resolution to the one used in training.
        """
        if '768' in self.fp_ckpt:
            self.height = 768
            self.width = 768
        else:
            self.height = 512
            self.width = 512
        

    def init_inpainting(
            self, 
            image_source: Union[Image.Image, np.ndarray] = None, 
            mask_image: Union[Image.Image, np.ndarray] = None, 
            init_empty: Optional[bool] = False,
        ):
        r"""
        Initializes inpainting with a source and maks image.
        Args:
            image_source: Union[Image.Image, np.ndarray]
                Source image onto which the mask will be applied.
            mask_image: Union[Image.Image, np.ndarray]
                Mask image, value = 0 will stay untouched, value = 255 subjet to diffusion
            init_empty: Optional[bool]:
                Initialize inpainting with an empty image and mask, effectively disabling inpainting,
                useful for generating a first image for transitions using diffusion.
        """
        if not init_empty:
            assert image_source is not None, "init_inpainting: you need to provide image_source"
            assert mask_image is not None, "init_inpainting: you need to provide mask_image"
            if type(image_source) == np.ndarray:
                image_source = Image.fromarray(image_source)
            self.image_source = image_source
            
            if type(mask_image) == np.ndarray:
                mask_image = Image.fromarray(mask_image)
            self.mask_image = mask_image
        else:
            self.mask_image  = self.mask_empty
            self.image_source  = self.image_empty


    def get_text_embedding(self, prompt):
        c = self.model.get_learned_conditioning(prompt)
        return c

    @torch.no_grad()
    def run_diffusion_standard(
            self, 
            text_embeddings: torch.FloatTensor, 
            latents_for_injection: torch.FloatTensor = None, 
            idx_start: int = -1, 
            idx_stop: int = -1, 
            return_image: Optional[bool] = False
        ):
        r"""
        Wrapper function for run_diffusion_standard and run_diffusion_inpaint.
        Depending on the mode, the correct one will be executed.
        
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
                if self.guidance_scale != 1.0:
                    uc = self.model.get_learned_conditioning([""])
                else:
                    uc = None
                shape_latents = [self.C, self.height // self.f, self.width // self.f]
    
                self.sampler.make_schedule(ddim_num_steps=self.num_inference_steps-1, ddim_eta=self.ddim_eta, verbose=False)
                C, H, W = shape_latents
                size = (1, C, H, W)
                b = size[0]
                
                latents = torch.randn(size, generator=generator, device=self.device)
    
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
                    
                    # print(f"diffusion iter {i}")
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=self.device, dtype=torch.long)
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
                    
                batch = make_batch_sd(self.image_source, self.mask_image, txt="willbereplaced", device=self.device, num_samples=1)
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


if __name__ == "__main__":
    
    num_inference_steps = 20 # Number of diffusion interations
    
    # fp_ckpt = "../stable_diffusion_models/ckpt/768-v-ema.ckpt"
    # fp_config = '../stablediffusion/configs/stable-diffusion/v2-inference-v.yaml'
    
    fp_ckpt= "../stable_diffusion_models/ckpt/512-inpainting-ema.ckpt"
    fp_config = '../stablediffusion/configs//stable-diffusion/v2-inpainting-inference.yaml'
    
    sdh = StableDiffusionHolder(fp_ckpt, fp_config, num_inference_steps)
    # fp_ckpt= "../stable_diffusion_models/ckpt/512-base-ema.ckpt"
    # fp_config = '../stablediffusion/configs//stable-diffusion/v2-inference.yaml'
    
    
    
    #%% INPAINT PREPS
    image_source = Image.fromarray((255*np.random.rand(512,512,3)).astype(np.uint8))
    mask = 255*np.ones([512,512], dtype=np.uint8)
    mask[0:50, 0:50] = 0
    mask = Image.fromarray(mask)
    
    sdh.init_inpainting(image_source, mask)
    text_embedding = sdh.get_text_embedding("photo of a strange house, surreal painting")
    list_latents = sdh.run_diffusion_inpaint(text_embedding)
    
    #%%
    idx_inject = 3
    img_orig = sdh.latent2image(list_latents[-1])
    list_inject = sdh.run_diffusion_inpaint(text_embedding, list_latents[idx_inject], idx_start=idx_inject+1)
    img_inject = sdh.latent2image(list_inject[-1])
    
    img_diff = img_orig - img_inject
    import matplotlib.pyplot as plt
    plt.imshow(np.concatenate((img_orig, img_inject, img_diff), axis=1))



#%%


"""
next steps:
    incorporate into lb
    incorporate into outpaint
"""
