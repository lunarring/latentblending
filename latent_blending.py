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
sys.path.append('util')
# sys.path.append('../stablediffusion/ldm')
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

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentInpaintDiffusion
from stable_diffusion_holder import StableDiffusionHolder
import yaml
import lpips
#%% 
class LatentBlending():
    def __init__(
            self, 
            sdh: None,
            guidance_scale: float = 4,
            guidance_scale_mid_damper: float = 0.5,
            mid_compression_scaler: float = 1.2,
        ):
        r"""
        Initializes the latent blending class.
        Args:
            guidance_scale: float
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            guidance_scale_mid_damper: float = 0.5
                Reduces the guidance scale towards the middle of the transition.
                A value of 0.5 would decrease the guidance_scale towards the middle linearly by 0.5.
            mid_compression_scaler: float = 2.0
                Increases the sampling density in the middle (where most changes happen). Higher value
                imply more values in the middle. However the inflection point can occur outside the middle,
                thus high values can give rough transitions. Values around 2 should be fine.
            
        """
        assert guidance_scale_mid_damper>0 and guidance_scale_mid_damper<=1.0, f"guidance_scale_mid_damper neees to be in interval (0,1], you provided {guidance_scale_mid_damper}"

        self.sdh = sdh
        self.device = self.sdh.device
        self.width = self.sdh.width
        self.height = self.sdh.height
        self.guidance_scale_mid_damper = guidance_scale_mid_damper
        self.mid_compression_scaler = mid_compression_scaler
        self.seed1 = 0 
        self.seed2 = 0
    
        # Initialize vars
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""
        
        self.tree_latents = [None, None]
        self.tree_fracts = None
        self.idx_injection = []
        self.tree_status = None
        self.tree_final_imgs = []
        
        self.list_nmb_branches_prev = []
        self.list_injection_idx_prev = []
        self.text_embedding1 = None
        self.text_embedding2 = None
        self.image1_lowres = None
        self.image2_lowres = None
        self.negative_prompt = None
        self.num_inference_steps = self.sdh.num_inference_steps
        self.noise_level_upscaling = 20
        self.list_injection_idx = None
        self.list_nmb_branches = None
        
        # Mixing parameters
        self.branch1_crossfeed_power = 0.1
        self.branch1_crossfeed_range = 0.6
        self.branch1_crossfeed_decay = 0.8
        
        self.parental_crossfeed_power = 0.1
        self.parental_crossfeed_range = 0.8
        self.parental_crossfeed_power_decay = 0.8    
        
        self.set_guidance_scale(guidance_scale)
        self.init_mode()
        self.multi_transition_img_first = None
        self.multi_transition_img_last = None
        self.dt_per_diff = 0
        self.spatial_mask = None
        
        self.lpips = lpips.LPIPS(net='alex').cuda(self.device)
        

    def init_mode(self):
        r"""
        Sets the operational mode. Currently supported are standard, inpainting and x4 upscaling.
        """
        if isinstance(self.sdh.model, LatentUpscaleDiffusion):
            self.mode = 'upscale'
        elif isinstance(self.sdh.model, LatentInpaintDiffusion):
            self.sdh.image_source = None
            self.sdh.mask_image = None
            self.mode = 'inpaint'
        else:
            self.mode = 'standard'
            
    def set_guidance_scale(self, guidance_scale):
        r"""
        sets the guidance scale.
        """
        self.guidance_scale_base = guidance_scale
        self.guidance_scale = guidance_scale
        self.sdh.guidance_scale = guidance_scale
        
    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported
        """
        self.negative_prompt = negative_prompt
        self.sdh.set_negative_prompt(negative_prompt)
        
    def set_guidance_mid_dampening(self, fract_mixing):
        r"""
        Tunes the guidance scale down as a linear function of fract_mixing, 
        towards 0.5 the minimum will be reached.
        """
        mid_factor = 1 - np.abs(fract_mixing - 0.5)/ 0.5
        max_guidance_reduction = self.guidance_scale_base * (1-self.guidance_scale_mid_damper) - 1
        guidance_scale_effective = self.guidance_scale_base - max_guidance_reduction*mid_factor
        self.guidance_scale = guidance_scale_effective
        self.sdh.guidance_scale = guidance_scale_effective

    
    def set_branch1_crossfeed(self, crossfeed_power, crossfeed_range, crossfeed_decay):
        r"""
        Sets the crossfeed parameters for the first branch to the last branch.
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding between the first and last image branch.
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        self.branch1_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.branch1_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.branch1_crossfeed_decay = np.clip(crossfeed_decay, 0, 1)
        
    
    def set_parental_crossfeed(self, crossfeed_power, crossfeed_range, crossfeed_decay):
        r"""
        Sets the crossfeed parameters for all transition images (within the first and last branch).
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding from the parental branches  
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        self.parental_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.parental_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.parental_crossfeed_power_decay = np.clip(crossfeed_decay, 0, 1)


    def set_prompt1(self, prompt: str):
        r"""
        Sets the first prompt (for the first keyframe) including text embeddings.
        Args:
            prompt: str
                ABC trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt1 = prompt
        self.text_embedding1 = self.get_text_embeddings(self.prompt1)
        
    
    def set_prompt2(self, prompt: str):
        r"""
        Sets the second prompt (for the second keyframe) including text embeddings.
        Args:
            prompt: str
                XYZ trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt2 = prompt
        self.text_embedding2 = self.get_text_embeddings(self.prompt2)
        
    def set_image1(self, image: Image):
        r"""
        Sets the first image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image1_lowres = image
        
    def set_image2(self, image: Image):
        r"""
        Sets the second image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image2_lowres = image
    
    def run_transition(
            self,
            recycle_img1: Optional[bool] = False, 
            recycle_img2: Optional[bool] = False, 
            num_inference_steps: Optional[int] = 30,
            depth_strength: Optional[float] = 0.3,
            t_compute_max_allowed: Optional[float] = None,
            nmb_max_branches: Optional[int] = None,
            fixed_seeds: Optional[List[int]] = None,
        ):
        r"""
        Function for computing transitions.
        Returns a list of transition images using spherical latent blending.
        Args:
            recycle_img1: Optional[bool]:
                Don't recompute the latents for the first keyframe (purely prompt1). Saves compute.
            recycle_img2: Optional[bool]:
                Don't recompute the latents for the second keyframe (purely prompt2). Saves compute.
            num_inference_steps:
                Number of diffusion steps. Higher values will take more compute time.
            depth_strength:
                Determines how deep the first injection will happen. 
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed:
                Either provide t_compute_max_allowed or nmb_max_branches. 
                The maximum time allowed for computation. Higher values give better results but take longer. 
            nmb_max_branches: int
                Either provide t_compute_max_allowed or nmb_max_branches. The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent 
                of your computer.
            fixed_seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
            
        """
        
        # Sanity checks first
        assert self.text_embedding1 is not None, 'Set the first text embedding with .set_prompt1(...) before'
        assert self.text_embedding2 is not None, 'Set the second text embedding with .set_prompt2(...) before'
        
        # Random seeds
        if fixed_seeds is not None:
            if fixed_seeds == 'randomize':
                fixed_seeds = list(np.random.randint(0, 1000000, 2).astype(np.int32))
            else:
                assert len(fixed_seeds)==2, "Supply a list with len = 2"
        
            self.seed1 = fixed_seeds[0]
            self.seed2 = fixed_seeds[1]
        
        # Ensure correct num_inference_steps in holder
        self.num_inference_steps = num_inference_steps
        self.sdh.num_inference_steps = num_inference_steps
        
        # Compute / Recycle first image
        if not recycle_img1 or len(self.tree_latents[0]) != self.num_inference_steps:
            list_latents1 = self.compute_latents1()
        else:
            list_latents1 = self.tree_latents[0]
            
        # Compute / Recycle first image
        if not recycle_img2 or len(self.tree_latents[-1]) != self.num_inference_steps:
            list_latents2 = self.compute_latents2()
        else:
            list_latents2 = self.tree_latents[-1]
            
        # Reset the tree, injecting the edge latents1/2 we just generated/recycled
        self.tree_latents = [list_latents1, list_latents2]    
        self.tree_fracts = [0.0, 1.0]
        self.tree_final_imgs = [self.sdh.latent2image((self.tree_latents[0][-1])), self.sdh.latent2image((self.tree_latents[-1][-1]))]
        self.tree_idx_injection = [0, 0]
        
        # Hard-fix. Apply spatial mask only for list_latents2 but not for transition. WIP...
        self.spatial_mask = None
        
        # Set up branching scheme (dependent on provided compute time)
        list_idx_injection, list_nmb_stems = self.get_time_based_branching(depth_strength, t_compute_max_allowed, nmb_max_branches)

        # Run iteratively, starting with the longest trajectory. 
        # Always inserting new branches where they are needed most according to image similarity
        for s_idx in tqdm(range(len(list_idx_injection))):
            nmb_stems = list_nmb_stems[s_idx]
            idx_injection = list_idx_injection[s_idx]
            
            for i in range(nmb_stems):
                fract_mixing, b_parent1, b_parent2 = self.get_mixing_parameters(idx_injection)
                self.set_guidance_mid_dampening(fract_mixing)
                list_latents = self.compute_latents_mix(fract_mixing, b_parent1, b_parent2, idx_injection)
                self.insert_into_tree(fract_mixing, idx_injection, list_latents)
                # print(f"fract_mixing: {fract_mixing} idx_injection {idx_injection}")
            
        return self.tree_final_imgs
                

    def compute_latents1(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the first image
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        print("starting compute_latents1")
        list_conditionings = self.get_mixed_conditioning(0)
        t0 = time.time()
        latents_start = self.get_noise(self.seed1)
        list_latents1 = self.run_diffusion(
            list_conditionings, 
            latents_start = latents_start,
            idx_start = 0
            )
        t1 = time.time()
        self.dt_per_diff = (t1-t0) / self.num_inference_steps
        self.tree_latents[0] = list_latents1
        if return_image:
            return self.sdh.latent2image(list_latents1[-1])
        else:
            return list_latents1
    
    def compute_latents2(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the last image, which may be affected by the first image's trajectory.
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        print("starting compute_latents2")
        list_conditionings = self.get_mixed_conditioning(1)
        latents_start = self.get_noise(self.seed2)
        # Influence from branch1
        if self.branch1_crossfeed_power > 0.0:
            # Set up the mixing_coeffs
            idx_mixing_stop = int(round(self.num_inference_steps*self.branch1_crossfeed_range))
            mixing_coeffs = list(np.linspace(self.branch1_crossfeed_power, self.branch1_crossfeed_power*self.branch1_crossfeed_decay, idx_mixing_stop))     
            mixing_coeffs.extend((self.num_inference_steps-idx_mixing_stop)*[0])
            list_latents_mixing = self.tree_latents[0]
            list_latents2 = self.run_diffusion(
                list_conditionings, 
                latents_start = latents_start,
                idx_start = 0,
                list_latents_mixing = list_latents_mixing,
                mixing_coeffs = mixing_coeffs
                )
        else:
            list_latents2 = self.run_diffusion(list_conditionings, latents_start)
        self.tree_latents[-1] = list_latents2
        
        if return_image:
            return self.sdh.latent2image(list_latents2[-1])
        else:
            return list_latents2            


    def compute_latents_mix(self, fract_mixing, b_parent1, b_parent2, idx_injection):    
        r"""
        Runs a diffusion trajectory, using the latents from the respective parents
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            b_parent1: int
                index of parent1 to be used
            b_parent2: int
                index of parent2 to be used
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        list_conditionings = self.get_mixed_conditioning(fract_mixing)
        fract_mixing_parental = (fract_mixing - self.tree_fracts[b_parent1]) / (self.tree_fracts[b_parent2] - self.tree_fracts[b_parent1]) 
        # idx_reversed = self.num_inference_steps - idx_injection
        
        list_latents_parental_mix = []
        for i in range(self.num_inference_steps):
            latents_p1 = self.tree_latents[b_parent1][i]
            latents_p2 = self.tree_latents[b_parent2][i]
            if latents_p1 is None or latents_p2 is None:
                latents_parental = None
            else:
                latents_parental = interpolate_spherical(latents_p1, latents_p2, fract_mixing_parental)
            list_latents_parental_mix.append(latents_parental)

        idx_mixing_stop = int(round(self.num_inference_steps*self.parental_crossfeed_range))
        mixing_coeffs = idx_injection*[self.parental_crossfeed_power]
        nmb_mixing = idx_mixing_stop - idx_injection
        if nmb_mixing > 0:
            mixing_coeffs.extend(list(np.linspace(self.parental_crossfeed_power, self.parental_crossfeed_power*self.parental_crossfeed_power_decay, nmb_mixing)))     
        mixing_coeffs.extend((self.num_inference_steps-len(mixing_coeffs))*[0])
        
        latents_start = list_latents_parental_mix[idx_injection-1]
        list_latents = self.run_diffusion(
            list_conditionings, 
            latents_start = latents_start,
            idx_start = idx_injection,
            list_latents_mixing = list_latents_parental_mix,
            mixing_coeffs = mixing_coeffs
            )
        
        return list_latents

    def get_time_based_branching(self, depth_strength, t_compute_max_allowed=None, nmb_max_branches=None):
        r"""
        Sets up the branching scheme dependent on the time that is granted for compute.
        The scheme uses an estimation derived from the first image's computation speed.
        Either provide t_compute_max_allowed or nmb_max_branches
        Args:
            depth_strength:
                Determines how deep the first injection will happen. 
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed: float
                The maximum time allowed for computation. Higher values give better results
                but take longer. Use this if you want to fix your waiting time for the results. 
            nmb_max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent 
                of your computer.
        """
        idx_injection_base = int(round(self.num_inference_steps*depth_strength))
        list_idx_injection = np.arange(idx_injection_base, self.num_inference_steps-1, 3)
        list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
        t_compute = 0
        
        if nmb_max_branches is None:
            assert t_compute_max_allowed is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "t_compute_max_allowed"
        elif t_compute_max_allowed is None:
            assert nmb_max_branches is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "nmb_max_branches"
            nmb_max_branches -= 2 # discounting the outer frames
        else:
            raise ValueError("Either specify t_compute_max_allowed or nmb_max_branches")
            
        stop_criterion_reached = False
        is_first_iteration = True
        
        while not stop_criterion_reached:
            list_compute_steps = self.num_inference_steps - list_idx_injection
            list_compute_steps *= list_nmb_stems
            t_compute = np.sum(list_compute_steps) * self.dt_per_diff  + 0.15*np.sum(list_nmb_stems)
            increase_done = False
            for s_idx in range(len(list_nmb_stems)-1):
                if list_nmb_stems[s_idx+1] / list_nmb_stems[s_idx] >= 2:
                    list_nmb_stems[s_idx] += 1
                    increase_done = True
                    break
            if not increase_done:
                list_nmb_stems[-1] += 1
            
            if stop_criterion == "t_compute_max_allowed" and t_compute > t_compute_max_allowed:
                stop_criterion_reached = True
            elif stop_criterion == "nmb_max_branches" and np.sum(list_nmb_stems) >= nmb_max_branches:
                stop_criterion_reached = True
                if is_first_iteration:
                    # Need to undersample.
                    list_idx_injection = np.linspace(list_idx_injection[0], list_idx_injection[-1], nmb_max_branches).astype(np.int32)
                    list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
            else:
                is_first_iteration = False
                
            # print(f"t_compute {t_compute} list_nmb_stems {list_nmb_stems}")
        return list_idx_injection, list_nmb_stems

    def get_mixing_parameters(self, idx_injection):
        r"""
        Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place
        where the metric is maximal.
        Args:
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        # get_lpips_similarity
        similarities = []
        for i in range(len(self.tree_final_imgs)-1):
            similarities.append(self.get_lpips_similarity(self.tree_final_imgs[i], self.tree_final_imgs[i+1]))
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1+1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]
        
        # Ensure that the parents are indeed older!
        b_parent1 = b_closest1
        while True:
            if self.tree_idx_injection[b_parent1] < idx_injection:
                break
            else:
                b_parent1 -= 1
            
        b_parent2 = b_closest2
        while True:
            if self.tree_idx_injection[b_parent2] < idx_injection:
                break
            else:
                b_parent2 += 1
                
        # print(f"\n\nb_closest: {b_closest1} {b_closest2} fract_closest1 {fract_closest1} fract_closest2 {fract_closest2}")
        # print(f"b_parent: {b_parent1} {b_parent2}")
        # print(f"similarities {similarities}")
        # print(f"idx_injection {idx_injection} tree_idx_injection {self.tree_idx_injection}")
        
        fract_mixing = (fract_closest1 + fract_closest2) /2 
        return fract_mixing, b_parent1, b_parent2
        
     
    def insert_into_tree(self, fract_mixing, idx_injection, list_latents):
        r"""
        Inserts all necessary parameters into the trajectory tree.
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
            list_latents: list
                list of the latents to be inserted
        """
        b_parent1, b_parent2 = get_closest_idx(fract_mixing, self.tree_fracts)
        self.tree_latents.insert(b_parent1+1, list_latents)
        self.tree_final_imgs.insert(b_parent1+1, self.sdh.latent2image(list_latents[-1]))
        self.tree_fracts.insert(b_parent1+1, fract_mixing)
        self.tree_idx_injection.insert(b_parent1+1, idx_injection)
            
    
    def get_spatial_mask_template(self):   
        r"""
        Experimental helper function to get a spatial mask template. 
        """
        shape_latents = [self.sdh.C, self.sdh.height // self.sdh.f, self.sdh.width // self.sdh.f]
        C, H, W = shape_latents
        return np.ones((H, W))
    
    def set_spatial_mask(self, img_mask):
        r"""
        Experimental helper function to set a spatial mask. 
        The mask forces latents to be overwritten.
        Args:
            img_mask: 
                mask image [0,1]. You can get a template using get_spatial_mask_template
            
        """
        
        shape_latents = [self.sdh.C, self.sdh.height // self.sdh.f, self.sdh.width // self.sdh.f]
        C, H, W = shape_latents
        img_mask = np.asarray(img_mask)
        assert len(img_mask.shape) == 2, "Currently, only 2D images are supported as mask"
        img_mask = np.clip(img_mask, 0, 1)
        assert img_mask.shape[0] == H, f"Your mask needs to be of dimension {H} x {W}"
        assert img_mask.shape[1] == W, f"Your mask needs to be of dimension {H} x {W}"
        spatial_mask = torch.from_numpy(img_mask).to(device=self.device)
        spatial_mask = torch.unsqueeze(spatial_mask, 0)
        spatial_mask = spatial_mask.repeat((C,1,1))
        spatial_mask = torch.unsqueeze(spatial_mask, 0)
        
        self.spatial_mask = spatial_mask
        
        
    def get_noise(self, seed):
        r"""
        Helper function to get noise given seed.
        Args:
            seed: int
            
        """
        generator = torch.Generator(device=self.sdh.device).manual_seed(int(seed))
        if self.mode == 'standard':
            shape_latents = [self.sdh.C, self.sdh.height // self.sdh.f, self.sdh.width // self.sdh.f]
            C, H, W = shape_latents
        elif self.mode == 'upscale':
            w = self.image1_lowres.size[0]
            h = self.image1_lowres.size[1]
            shape_latents = [self.sdh.model.channels, h, w]
            C, H, W = shape_latents
        
        return torch.randn((1, C, H, W), generator=generator, device=self.sdh.device)


    @torch.no_grad()
    def run_diffusion(
            self, 
            list_conditionings, 
            latents_start: torch.FloatTensor = None, 
            idx_start: int = 0, 
            list_latents_mixing = None, 
            mixing_coeffs = 0.0,
            return_image: Optional[bool] = False
        ):
        
        r"""
        Wrapper function for diffusion runners.
        Depending on the mode, the correct one will be executed.
        
        Args:
            list_conditionings: list
                List of all conditionings for the diffusion model.
            latents_start: torch.FloatTensor 
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            list_latents_mixing: torch.FloatTensor 
                List of latents (latent trajectories) that are used for mixing
            mixing_coeffs: float or list
                Coefficients, how strong each element of list_latents_mixing will be mixed in.
            return_image: Optional[bool]
                Optionally return image directly
        """
        
        # Ensure correct num_inference_steps in Holder
        self.sdh.num_inference_steps = self.num_inference_steps
        assert type(list_conditionings) is list, "list_conditionings need to be a list"
        
        if self.mode == 'standard':
            text_embeddings = list_conditionings[0]
            return self.sdh.run_diffusion_standard(
                text_embeddings = text_embeddings,
                latents_start = latents_start,
                idx_start = idx_start,
                list_latents_mixing = list_latents_mixing,
                mixing_coeffs = mixing_coeffs,
                spatial_mask =  self.spatial_mask,
                return_image = return_image,
                )
        
        elif self.mode == 'upscale':
            cond = list_conditionings[0]
            uc_full = list_conditionings[1]
            return self.sdh.run_diffusion_upscaling(
                cond, 
                uc_full, 
                latents_start=latents_start, 
                idx_start=idx_start, 
                list_latents_mixing = list_latents_mixing,
                mixing_coeffs = mixing_coeffs,
                return_image=return_image)

        
    def run_upscaling(
            self, 
            dp_img: str,
            depth_strength: float = 0.65,
            num_inference_steps: int = 100,
            nmb_max_branches_highres: int = 5,
            nmb_max_branches_lowres: int = 6,
            duration_single_segment = 3,
            fixed_seeds: Optional[List[int]] = None,
            ):
        r"""
        Runs upscaling with the x4 model. Requires that you run a transition before with a low-res model and save the results using write_imgs_transition.
        
        Args:
            dp_img: str
                Path to the low-res transition path (as saved in write_imgs_transition)
            depth_strength:
                Determines how deep the first injection will happen. 
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            num_inference_steps:
                Number of diffusion steps. Higher values will take more compute time.
            nmb_max_branches_highres: int
                Number of final branches of the upscaling transition pass. Note this is the number
                of branches between each pair of low-res images.
            nmb_max_branches_lowres: int
                Number of input low-res images, subsampling all transition images written in the low-res pass.
                Setting this number lower (e.g. 6) will decrease the compute time but not affect the results too much.
            duration_single_segment: float
                The duration of each high-res movie segment. You will have nmb_max_branches_lowres-1 segments in total.
            fixed_seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
        """
        fp_yml = os.path.join(dp_img, "lowres.yaml")
        fp_movie = os.path.join(dp_img, "movie_highres.mp4")
        fps = 24
        ms = MovieSaver(fp_movie, fps=fps)
        assert os.path.isfile(fp_yml), "lowres.yaml does not exist. did you forget run_upscaling_step1?"
        dict_stuff = yml_load(fp_yml)
        
        # load lowres images
        nmb_images_lowres = dict_stuff['nmb_images']
        prompt1 = dict_stuff['prompt1']
        prompt2 = dict_stuff['prompt2']
        idx_img_lowres = np.round(np.linspace(0, nmb_images_lowres-1, nmb_max_branches_lowres)).astype(np.int32)
        imgs_lowres = []
        for i in idx_img_lowres:
            fp_img_lowres = os.path.join(dp_img, f"lowres_img_{str(i).zfill(4)}.jpg")
            assert os.path.isfile(fp_img_lowres), f"{fp_img_lowres} does not exist. did you forget run_upscaling_step1?"
            imgs_lowres.append(Image.open(fp_img_lowres))
        

        # set up upscaling
        text_embeddingA = self.sdh.get_text_embedding(prompt1)
        text_embeddingB = self.sdh.get_text_embedding(prompt2)
        
        list_fract_mixing = np.linspace(0, 1, nmb_max_branches_lowres-1)
        
        for i in range(nmb_max_branches_lowres-1):
            print(f"Starting movie segment {i+1}/{nmb_max_branches_lowres-1}")
            
            self.text_embedding1 = interpolate_linear(text_embeddingA, text_embeddingB, list_fract_mixing[i])
            self.text_embedding2 = interpolate_linear(text_embeddingA, text_embeddingB, 1-list_fract_mixing[i])
            
            if i==0:
                recycle_img1 = False    
            else:
                self.swap_forward()
                recycle_img1 = True    
            
            self.set_image1(imgs_lowres[i])
            self.set_image2(imgs_lowres[i+1])
            
            list_imgs = self.run_transition(
                recycle_img1 = recycle_img1,
                recycle_img2 = False,
                num_inference_steps = num_inference_steps, 
                depth_strength = depth_strength,
                nmb_max_branches = nmb_max_branches_highres,
                )
            
            list_imgs_interp = add_frames_linear_interp(list_imgs, fps, duration_single_segment)
            
            # Save movie frame
            for img in list_imgs_interp:
                ms.write_frame(img)
                
        ms.finalize()
        

   
    @torch.no_grad()
    def get_mixed_conditioning(self, fract_mixing):
        if self.mode == 'standard':
            text_embeddings_mix = interpolate_linear(self.text_embedding1, self.text_embedding2, fract_mixing)
            list_conditionings = [text_embeddings_mix]
        elif self.mode == 'inpaint':
            text_embeddings_mix = interpolate_linear(self.text_embedding1, self.text_embedding2, fract_mixing)
            list_conditionings = [text_embeddings_mix]
        elif self.mode == 'upscale':
            text_embeddings_mix = interpolate_linear(self.text_embedding1, self.text_embedding2, fract_mixing)
            cond, uc_full = self.sdh.get_cond_upscaling(self.image1_lowres, text_embeddings_mix, self.noise_level_upscaling)
            condB, uc_fullB = self.sdh.get_cond_upscaling(self.image2_lowres, text_embeddings_mix, self.noise_level_upscaling)
            cond['c_concat'][0] = interpolate_spherical(cond['c_concat'][0], condB['c_concat'][0], fract_mixing)
            uc_full['c_concat'][0] = interpolate_spherical(uc_full['c_concat'][0], uc_fullB['c_concat'][0], fract_mixing)
            list_conditionings = [cond, uc_full]
        else:
            raise ValueError(f"mix_conditioning: unknown mode {self.mode}")
        return list_conditionings

    @torch.no_grad()
    def get_text_embeddings(
            self, 
            prompt: str
        ):
        r"""
        Computes the text embeddings provided a string with a prompts.
        Adapted from stable diffusion repo
        Args:
            prompt: str
                ABC trending on artstation painted by Old Greg.
        """
        
        return self.sdh.get_text_embedding(prompt)
    

    def write_imgs_transition(self, dp_img):
        r"""
        Writes the transition images into the folder dp_img.
        Requires run_transition to be completed.
        Args:
            dp_img: str
                Directory, into which the transition images, yaml file and latents are written.
        """
        imgs_transition = self.tree_final_imgs
        os.makedirs(dp_img, exist_ok=True)
        for i, img in enumerate(imgs_transition):
            img_leaf = Image.fromarray(img)
            img_leaf.save(os.path.join(dp_img, f"lowres_img_{str(i).zfill(4)}.jpg"))
        
        fp_yml = os.path.join(dp_img, "lowres.yaml") 
        self.save_statedict(fp_yml)
        
    def write_movie_transition(self, fp_movie, duration_transition, fps=30):
        r"""
        Writes the transition movie to fp_movie, using the given duration and fps..
        The missing frames are linearly interpolated.
        Args:
            fp_movie: str
                file pointer to the final movie.
            duration_transition: float
                duration of the movie in seonds
            fps: int
                fps of the movie
                
        """
        
        # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
        imgs_transition_ext = add_frames_linear_interp(self.tree_final_imgs, duration_transition, fps)

        # Save as MP4
        if os.path.isfile(fp_movie):
            os.remove(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps, shape_hw=[self.sdh.height, self.sdh.width])
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()

        
        
    def save_statedict(self, fp_yml):
        # Dump everything relevant into yaml
        imgs_transition = self.tree_final_imgs
        state_dict = self.get_state_dict()
        state_dict['nmb_images'] = len(imgs_transition)
        yml_save(fp_yml, state_dict)
        
    def get_state_dict(self):
        state_dict = {}
        grab_vars = ['prompt1', 'prompt2', 'seed1', 'seed2', 'height', 'width',
                     'num_inference_steps', 'depth_strength', 'guidance_scale',
                     'guidance_scale_mid_damper', 'mid_compression_scaler', 'negative_prompt',
                     'branch1_crossfeed_power', 'branch1_crossfeed_range', 'branch1_crossfeed_decay'
                     'parental_crossfeed_power', 'parental_crossfeed_range', 'parental_crossfeed_power_decay']
        for v in grab_vars:
            if hasattr(self, v):
                if v == 'seed1' or v == 'seed2':
                    state_dict[v] = int(getattr(self, v))
                elif v == 'guidance_scale':
                    state_dict[v] = float(getattr(self, v))
                    
                else:
                    try:
                        state_dict[v] = getattr(self, v)
                    except Exception as e:
                        pass
                
        return state_dict
        
    def randomize_seed(self):
        r"""
        Set a random seed for a fresh start.
        """ 
        seed = np.random.randint(999999999)
        self.set_seed(seed)
    
    def set_seed(self, seed: int):
        r"""
        Set a the seed for a fresh start.
        """ 
        self.seed = seed
        self.sdh.seed = seed
        
    def set_width(self, width):
        r"""
        Set the width of the resulting image.
        """ 
        assert np.mod(width, 64) == 0, "set_width: value needs to be divisible by 64"
        self.width = width
        self.sdh.width = width
        
    def set_height(self, height):
        r"""
        Set the height of the resulting image.
        """ 
        assert np.mod(height, 64) == 0, "set_height: value needs to be divisible by 64"
        self.height = height
        self.sdh.height = height
        

    def swap_forward(self):
        r"""
        Moves over keyframe two -> keyframe one. Useful for making a sequence of transitions
        as in run_multi_transition()
        """ 
        # Move over all latents
        self.tree_latents[0] = self.tree_latents[-1]
        
        # Move over prompts and text embeddings
        self.prompt1 = self.prompt2
        self.text_embedding1 = self.text_embedding2
        
        # Final cleanup for extra sanity
        self.tree_final_imgs = [] 
        
        
    def get_lpips_similarity(self, imgA, imgB):
        r"""
        Computes the image similarity between two images imgA and imgB. 
        Used to determine the optimal point of insertion to create smooth transitions.
        High values indicate low similarity.
        """ 
        tensorA = torch.from_numpy(imgA).float().cuda(self.device)
        tensorA = 2*tensorA/255.0 - 1
        tensorA = tensorA.permute([2,0,1]).unsqueeze(0)
        
        tensorB = torch.from_numpy(imgB).float().cuda(self.device)
        tensorB = 2*tensorB/255.0 - 1
        tensorB = tensorB.permute([2,0,1]).unsqueeze(0)
        lploss = self.lpips(tensorA, tensorB)
        lploss = float(lploss[0][0][0][0])
        
        return lploss
        
        
# Auxiliary functions
def get_closest_idx(
        fract_mixing: float, 
        list_fract_mixing_prev: List[float],
    ):
    r"""
    Helper function to retrieve the parents for any given mixing.
    Example: fract_mixing = 0.4 and list_fract_mixing_prev = [0, 0.3, 0.6, 1.0]
    Will return the two closest values from list_fract_mixing_prev, i.e. [1, 2]
    """ 
        
    pdist = fract_mixing - np.asarray(list_fract_mixing_prev)
    pdist_pos = pdist.copy()
    pdist_pos[pdist_pos<0] = np.inf
    b_parent1 = np.argmin(pdist_pos)
    pdist_neg = -pdist.copy()
    pdist_neg[pdist_neg<=0] = np.inf
    b_parent2= np.argmin(pdist_neg)
    
    if b_parent1 > b_parent2:
        tmp = b_parent2
        b_parent2 = b_parent1
        b_parent1 = tmp
    
    return b_parent1, b_parent2

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


def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0: 
            First tensor / np.ndarray for interpolation
        p1: 
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """ 
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)
        
    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)
    
    interp = (1-fract_mixing) * p0 + fract_mixing * p1
    
    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)
        
    return interp


def add_frames_linear_interp(
        list_imgs: List[np.ndarray], 
        fps_target: Union[float, int] = None, 
        duration_target: Union[float, int] = None,
        nmb_frames_target: int=None,
    ):
    r"""
    Helper function to cheaply increase the number of frames given a list of images, 
    by virtue of standard linear interpolation.
    The number of inserted frames will be automatically adjusted so that the total of number
    of frames can be fixed precisely, using a random shuffling technique.
    The function allows 1:1 comparisons between transitions as videos.
    
    Args:
        list_imgs: List[np.ndarray)
            List of images, between each image new frames will be inserted via linear interpolation.
        fps_target: 
            OptionA: specify here the desired frames per second.
        duration_target: 
            OptionA: specify here the desired duration of the transition in seconds.
        nmb_frames_target: 
            OptionB: directly fix the total number of frames of the output.
    """ 
    
    # Sanity
    if nmb_frames_target is not None and fps_target is not None:
        raise ValueError("You cannot specify both fps_target and nmb_frames_target")
    if fps_target is None:
        assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
    if nmb_frames_target is None:
        assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        nmb_frames_target = fps_target*duration_target
    
    # Get number of frames that are missing
    nmb_frames_diff = len(list_imgs)-1
    nmb_frames_missing = nmb_frames_target - nmb_frames_diff - 1
    
    if nmb_frames_missing < 1:
        return list_imgs
    
    list_imgs_float = [img.astype(np.float32) for img in list_imgs]
    # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
    mean_nmb_frames_insert = nmb_frames_missing/nmb_frames_diff
    constfact = np.floor(mean_nmb_frames_insert)
    remainder_x = 1-(mean_nmb_frames_insert - constfact)
    
    nmb_iter = 0
    while True:
        nmb_frames_to_insert = np.random.rand(nmb_frames_diff)
        nmb_frames_to_insert[nmb_frames_to_insert<=remainder_x] = 0
        nmb_frames_to_insert[nmb_frames_to_insert>remainder_x] = 1
        nmb_frames_to_insert += constfact
        if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
            break
        nmb_iter += 1
        if nmb_iter > 100000:
            print("add_frames_linear_interp: issue with inserting the right number of frames")
            break
        
    nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
    list_imgs_interp = []
    for i in range(len(list_imgs_float)-1):#, desc="STAGE linear interp"):
        img0 = list_imgs_float[i]
        img1 = list_imgs_float[i+1]
        list_imgs_interp.append(img0.astype(np.uint8))
        list_fracts_linblend = np.linspace(0, 1, nmb_frames_to_insert[i]+2)[1:-1]
        for fract_linblend in list_fracts_linblend:
            img_blend = interpolate_linear(img0, img1, fract_linblend).astype(np.uint8)
            list_imgs_interp.append(img_blend.astype(np.uint8))
        
        if i==len(list_imgs_float)-2:
            list_imgs_interp.append(img1.astype(np.uint8))
    
    return list_imgs_interp


def get_spacing(nmb_points: int, scaling: float):
    """
    Helper function for getting nonlinear spacing between 0 and 1, symmetric around 0.5
    Args:
        nmb_points: int
            Number of points between [0, 1]
        scaling: float
            Higher values will return higher sampling density around 0.5
            
    """
    if scaling < 1.7:
        return np.linspace(0, 1, nmb_points)
    nmb_points_per_side = nmb_points//2 + 1
    if np.mod(nmb_points, 2) != 0: # uneven case
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side)**scaling / 2 - 0.5)
        right_side = 1-left_side[::-1][1:]
    else:
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side)**scaling / 2 - 0.5)[0:-1]
        right_side = 1-left_side[::-1]
    all_fracts = np.hstack([left_side, right_side])
    return all_fracts


def get_time(resolution=None):
    """
    Helper function returning an nicely formatted time string, e.g. 221117_1620
    """
    if resolution==None:
        resolution="second"
    if resolution == "day":
        t = time.strftime('%y%m%d', time.localtime())
    elif resolution == "minute":
        t = time.strftime('%y%m%d_%H%M', time.localtime())
    elif resolution == "second":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
    elif resolution == "millisecond":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
        t += "_"
        t += str("{:03d}".format(int(int(datetime.utcnow().strftime('%f'))/1000)))
    else:
        raise ValueError("bad resolution provided: %s" %resolution)
    return t

def compare_dicts(a, b):
    """
    Compares two dictionaries a and b and returns a dictionary c, with all 
    keys,values that have shared keys in a and b but same values in a and b.
    The values of a and b are stacked together in the output.
    Example:
        a = {}; a['bobo'] = 4
        b = {}; b['bobo'] = 5
        c = dict_compare(a,b)
        c = {"bobo",[4,5]}
    """
    c = {}
    for key in a.keys():
        if key in b.keys():
          val_a = a[key]  
          val_b = b[key]  
          if val_a != val_b:
              c[key] = [val_a, val_b]
    return c

def yml_load(fp_yml, print_fields=False):
    """
    Helper function for loading yaml files
    """
    with open(fp_yml) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    dict_data = dict(data)
    print("load: loaded {}".format(fp_yml))
    return dict_data

def yml_save(fp_yml, dict_stuff):
    """
    Helper function for saving yaml files
    """
    with open(fp_yml, 'w') as f:
        data = yaml.dump(dict_stuff, f, sort_keys=False, default_flow_style=False)
    print("yml_save: saved {}".format(fp_yml))


#%% le main
if __name__ == "__main__":
    # xxxx
    
    #%% First let us spawn a stable diffusion holder
    device = "cuda" 
    fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 
    
    sdh = StableDiffusionHolder(fp_ckpt)
    
    xxx
    
        
    #%% Next let's set up all parameters
    depth_strength = 0.3 # Specifies how deep (in terms of diffusion iterations the first branching happens)
    fixed_seeds = [697164, 430214]
        
    prompt1 = "photo of a desert and a sky"
    prompt2 = "photo of a tree with a lake"
    
    duration_transition = 12 # In seconds
    fps = 30
    
    # Spawn latent blending
    self = LatentBlending(sdh)
    
    self.set_prompt1(prompt1)
    self.set_prompt2(prompt2)
    
    # Run latent blending
    self.branch1_crossfeed_power = 0.3
    self.branch1_crossfeed_range = 0.4
    # self.run_transition(depth_strength=depth_strength, fixed_seeds=fixed_seeds)
    self.seed1=21312
    img1 =self.compute_latents1(True)
    #%
    self.seed2=1234121
    self.branch1_crossfeed_power = 0.7
    self.branch1_crossfeed_range = 0.3
    self.branch1_crossfeed_decay = 0.3
    img2 =self.compute_latents2(True)
    # Image.fromarray(np.concatenate((img1, img2), axis=1))
    
    #%%
    t0  = time.time()
    self.t_compute_max_allowed = 30
    self.parental_crossfeed_range = 1.0
    self.parental_crossfeed_power = 0.0
    self.parental_crossfeed_power_decay = 1.0    
    imgs_transition = self.run_transition(recycle_img1=True, recycle_img2=True)
    t1 = time.time()
    print(f"took: {t1-t0}s")