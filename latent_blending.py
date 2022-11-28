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
sys.path.append('../stablediffusion/ldm')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion_holder import StableDiffusionHolder
#%% 
class LatentBlending():
    def __init__(
            self, 
            sdh: None,
            guidance_scale: float = 4,
            guidance_scale_mid_damper: float = 0.5,
            mid_compression_scaler: float = 2.0,
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
        self.sdh = sdh
        self.device = self.sdh.device
        self.width = self.sdh.width
        self.height = self.sdh.height
        assert guidance_scale_mid_damper>0 and guidance_scale_mid_damper<=1.0, f"guidance_scale_mid_damper neees to be in interval (0,1], you provided {guidance_scale_mid_damper}"
        self.guidance_scale_mid_damper = guidance_scale_mid_damper
        self.mid_compression_scaler = mid_compression_scaler
        self.seed = 420 # Run self.set_seed or fixed_seeds argument in run_transition
    
        # Initialize vars
        self.prompt1 = ""
        self.prompt2 = ""
        self.tree_latents = []
        self.tree_fracts = []
        self.tree_status = []
        self.tree_final_imgs = []
        self.list_nmb_branches_prev = []
        self.list_injection_idx_prev = []
        self.text_embedding1 = None
        self.text_embedding2 = None
        self.stop_diffusion = False
        self.negative_prompt = None
        self.num_inference_steps = -1
        self.list_injection_idx = None
        self.list_nmb_branches = None
        self.set_guidance_scale(guidance_scale)
        self.init_mode()
        

    def init_mode(self, mode='standard'):
        r"""
        Automatically sets the mode of this class, depending on the supplied pipeline.
        FIXME XXX
        """
        if mode == 'inpaint':
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
        
    def set_guidance_mid_dampening(self, fract_mixing):
        r"""
        Tunes the guidance scale down as a linear function of fract_mixing, 
        towards 0.5 the minimum will be reached.
        """
        mid_factor = 1 - np.abs(fract_mixing - 0.5)/ 0.5
        max_guidance_reduction = self.guidance_scale_base * (1-self.guidance_scale_mid_damper)
        guidance_scale_effective = self.guidance_scale_base - max_guidance_reduction*mid_factor
        self.guidance_scale = guidance_scale_effective
        self.sdh.guidance_scale = guidance_scale_effective

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
        
    def autosetup_branching(
            self, 
            quality: str = 'medium',
            deepth_strength: float = 0.65,
            nmb_frames: int = 360,
            nmb_mindist: int = 3,
        ):
        r"""
        Helper function to set up the branching structure automatically.
        
        Args:
            quality: str 
                Determines how many diffusion steps are being made + how many branches in total.
                Tradeoff between quality and speed of computation.
                Choose: lowest, low, medium, high, ultra
            deepth_strength: float = 0.65,
                Determines how deep the first injection will happen. 
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            nmb_frames: int = 360,
                total number of frames
            nmb_mindist: int = 3 
                minimum distance in terms of diffusion iteratinos between subsequent injections

        """ 
        
        if quality == 'lowest':
            num_inference_steps = 12
            nmb_branches_final = 5
        elif quality == 'low':
            num_inference_steps = 15
            nmb_branches_final = nmb_frames//16
        elif quality == 'medium':
            num_inference_steps = 30
            nmb_branches_final = nmb_frames//8
        elif quality == 'high':
            num_inference_steps = 60
            nmb_branches_final = nmb_frames//4
        elif quality == 'ultra':
            num_inference_steps = 100
            nmb_branches_final = nmb_frames//2
        else: 
            raise ValueError("quality = '{quality}' not supported")
        
        idx_injection_first = int(np.round(num_inference_steps*deepth_strength))
        idx_injection_last = num_inference_steps - 3
        nmb_injections = int(np.floor(num_inference_steps/5)) - 1
        
        list_injection_idx = [0]
        list_injection_idx.extend(np.linspace(idx_injection_first, idx_injection_last, nmb_injections).astype(int))
        list_nmb_branches = np.round(np.logspace(np.log10(2), np.log10(nmb_branches_final), nmb_injections+1)).astype(int)
        
        # Cleanup. There should be at least 3 diffusion steps between each injection
        list_injection_idx_clean = [list_injection_idx[0]]
        list_nmb_branches_clean = [list_nmb_branches[0]]
        idx_last_check = 0
        for i in range(len(list_injection_idx)-1):
            if list_injection_idx[i+1] - list_injection_idx_clean[idx_last_check] >= nmb_mindist:
                list_injection_idx_clean.append(list_injection_idx[i+1])
                list_nmb_branches_clean.append(list_nmb_branches[i+1])
                idx_last_check +=1 
        list_injection_idx_clean = [int(l) for l in list_injection_idx_clean]
        list_nmb_branches_clean = [int(l) for l in list_nmb_branches_clean]
        
        list_injection_idx = list_injection_idx_clean
        list_nmb_branches = list_nmb_branches_clean

        # print(f"num_inference_steps: {num_inference_steps}")
        # print(f"list_injection_idx: {list_injection_idx}")
        # print(f"list_nmb_branches: {list_nmb_branches}")
        
        list_nmb_branches = list_nmb_branches
        list_injection_idx = list_injection_idx
        self.setup_branching(num_inference_steps, list_nmb_branches=list_nmb_branches, list_injection_idx=list_injection_idx)

    
    def setup_branching(self,
                        num_inference_steps: int =30,
                        list_nmb_branches: List[int] = None, 
                        list_injection_strength: List[float] = None, 
                        list_injection_idx: List[int] = None, 
                      ):
            r""" 
            Sets the branching structure for making transitions.
            num_inference_steps: int
                Number of diffusion steps. Larger values will take more compute time.
            list_nmb_branches: List[int]:
                list of the number of branches for each injection.
            list_injection_strength: List[float]:
                list of injection strengths within interval [0, 1), values need to be increasing.
                Alternatively you can direclty specify the list_injection_idx.
            list_injection_idx: List[int]:
                list of injection strengths within interval [0, 1), values need to be increasing.
                Alternatively you can specify the list_injection_strength.
                
            """
            # Assert
            assert not((list_injection_strength is not None) and (list_injection_idx is not None)), "suppyl either list_injection_strength or list_injection_idx"
            
            if list_injection_strength is None:
                assert list_injection_idx is not None, "Supply either list_injection_idx or list_injection_strength"
                assert isinstance(list_injection_idx[0], int) or isinstance(list_injection_idx[0], np.int) , "Need to supply integers for list_injection_idx"
                
            if list_injection_idx is None:
                assert list_injection_strength is not None, "Supply either list_injection_idx or list_injection_strength"
                # Create the injection indexes
                list_injection_idx = [int(round(x*num_inference_steps)) for x in list_injection_strength]
                assert min(np.diff(list_injection_idx)) > 0, 'Injection idx needs to be increasing'
                if min(np.diff(list_injection_idx)) < 2:
                    print("Warning: your injection spacing is very tight. consider increasing the distances")
                assert isinstance(list_injection_strength[1], np.floating) or isinstance(list_injection_strength[1], float), "Need to supply floats for list_injection_strength"
                # we are checking element 1 in list_injection_strength because "0" is an int... [0, 0.5]
            
            assert max(list_injection_idx) < num_inference_steps, "Decrease the injection index or strength"
            assert len(list_injection_idx) == len(list_nmb_branches), "Need to have same length"
            assert max(list_injection_idx) < num_inference_steps,"Injection index cannot happen after last diffusion step! Decrease list_injection_idx or list_injection_strength[-1]"
                
            
            # Set attributes
            self.num_inference_steps = num_inference_steps
            self.sdh.num_inference_steps = num_inference_steps
            self.list_nmb_branches = list_nmb_branches
            self.list_injection_idx = list_injection_idx
            
           
    
    def run_transition(
            self, 
            recycle_img1: Optional[bool] = False, 
            recycle_img2: Optional[bool] = False, 
            fixed_seeds: Optional[List[int]] = None,
        ):
        r"""
        Returns a list of transition images using spherical latent blending.
        Args:
            recycle_img1: Optional[bool]:
                Don't recompute the latents for the first keyframe (purely prompt1). Saves compute.
            recycle_img2: Optional[bool]:
                Don't recompute the latents for the second keyframe (purely prompt2). Saves compute.
            fixed_seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
            
        """
        # Sanity checks first
        assert self.text_embedding1 is not None, 'Set the first text embedding with .set_prompt1(...) before'
        assert self.text_embedding2 is not None, 'Set the second text embedding with .set_prompt2(...) before'
        assert self.list_injection_idx is not None, 'Set the branching structure before, by calling autosetup_branching or setup_branching'
        
        if fixed_seeds is not None:
            if fixed_seeds == 'randomize':
                fixed_seeds = list(np.random.randint(0, 1000000, 2).astype(np.int32))
            else:
                assert len(fixed_seeds)==2, "Supply a list with len = 2"
                
        # Process interruption variable
        self.stop_diffusion = False
        
        # Ensure correct num_inference_steps in holder
        self.sdh.num_inference_steps = self.num_inference_steps
        
        # Recycling? There are requirements
        if recycle_img1 or recycle_img2:
            if self.list_nmb_branches_prev == []:
                print("Warning. You want to recycle but there is nothing here. Disabling recycling.")
                recycle_img1 = False
                recycle_img2 = False
            elif self.list_nmb_branches_prev != self.list_nmb_branches:
                print("Warning. Cannot change list_nmb_branches if recycling latent. Disabling recycling.")
                recycle_img1 = False
                recycle_img2 = False
            elif self.list_injection_idx_prev != self.list_injection_idx:
                print("Warning. Cannot change list_nmb_branches if recycling latent. Disabling recycling.")
                recycle_img1 = False
                recycle_img2 = False
        
        # Make a backup for future reference
        self.list_nmb_branches_prev = self.list_nmb_branches[:]
        self.list_injection_idx_prev = self.list_injection_idx[:]
        
        # Auto inits
        list_injection_idx_ext = self.list_injection_idx[:] 
        list_nmb_branches = self.list_nmb_branches[:] 
        list_injection_idx_ext.append(self.num_inference_steps)
        
        # If injection at depth 0 not specified, we will start out with 2 branches
        if list_injection_idx_ext[0] != 0:
            list_injection_idx_ext.insert(0,0)
            list_nmb_branches.insert(0,2)
        assert list_nmb_branches[0] == 2, "Need to start with 2 branches. set list_nmb_branches[0]=2"
        
        # Pre-define entire branching tree structures
        if not recycle_img1 and not recycle_img2:
            self.tree_latents = []
            self.tree_fracts = []
            self.tree_status = []
            self.tree_final_imgs = [None]*list_nmb_branches[-1]
            self.tree_final_imgs_timing = [0]*list_nmb_branches[-1]
            
            nmb_blocks_time = len(list_injection_idx_ext)-1
            for t_block in range(nmb_blocks_time):
                nmb_branches = list_nmb_branches[t_block]
                # list_fract_mixing_current = np.linspace(0, 1, nmb_branches)
                list_fract_mixing_current = get_spacing(nmb_branches, self.mid_compression_scaler)
                self.tree_fracts.append(list_fract_mixing_current)
                self.tree_latents.append([None]*nmb_branches)
                self.tree_status.append(['untouched']*nmb_branches)
        else:
            self.tree_final_imgs = [None]*list_nmb_branches[-1]
            nmb_blocks_time = len(list_injection_idx_ext)-1
            for t_block in range(nmb_blocks_time):
                nmb_branches = list_nmb_branches[t_block]
                for idx_branch in range(nmb_branches):
                    self.tree_status[t_block][idx_branch] = 'untouched'
                if recycle_img1:
                    self.tree_status[t_block][0] = 'computed'
                    self.tree_final_imgs[0] = self.sdh.latent2image(self.tree_latents[-1][0][-1])
                    self.tree_final_imgs_timing[0] = 0
                if recycle_img2:
                    self.tree_status[t_block][-1] = 'computed'
                    self.tree_final_imgs[-1] = self.sdh.latent2image(self.tree_latents[-1][-1][-1])
                    self.tree_final_imgs_timing[-1] = 0
                    
        # setup compute order: goal: try to get last branch computed asap. 
        # first compute the right keyframe. needs to be there in any case
        list_compute = []
        list_local_stem = []
        for t_block in range(nmb_blocks_time - 1, -1, -1):
            if self.tree_status[t_block][0] == 'untouched':
                self.tree_status[t_block][0] = 'prefetched'
                list_local_stem.append([t_block, 0])
        list_compute.extend(list_local_stem[::-1]) 
        
        # setup compute order: start from last leafs (the final transition images) and work way down. what parents do they need?
        for idx_leaf in range(1, list_nmb_branches[-1]):
            list_local_stem = []
            t_block = nmb_blocks_time - 1
            t_block_prev = t_block - 1
            self.tree_status[t_block][idx_leaf] = 'prefetched'
            list_local_stem.append([t_block, idx_leaf])
            idx_leaf_deep = idx_leaf
            
            for t_block in range(nmb_blocks_time-1, 0, -1):
                t_block_prev = t_block - 1
                fract_mixing = self.tree_fracts[t_block][idx_leaf_deep]
                list_fract_mixing_prev = self.tree_fracts[t_block_prev]
                b_parent1, b_parent2 = get_closest_idx(fract_mixing, list_fract_mixing_prev)
                assert self.tree_status[t_block_prev][b_parent1] != 'untouched', 'Branch destruction??? This should never happen!'
                if self.tree_status[t_block_prev][b_parent2] == 'untouched':
                    self.tree_status[t_block_prev][b_parent2] = 'prefetched'
                    list_local_stem.append([t_block_prev, b_parent2])
                idx_leaf_deep = b_parent2
            list_compute.extend(list_local_stem[::-1])        
            
        # Diffusion computations start here
        time_start = time.time()
        for t_block, idx_branch in tqdm(list_compute, desc="computing transition", smoothing=-1):
            if self.stop_diffusion:
                print("run_transition: process interrupted")
                return self.tree_final_imgs
            
            # print(f"computing t_block {t_block} idx_branch {idx_branch}")
            idx_stop = list_injection_idx_ext[t_block+1]
            fract_mixing = self.tree_fracts[t_block][idx_branch]
            text_embeddings_mix = interpolate_linear(self.text_embedding1, self.text_embedding2, fract_mixing)
            self.set_guidance_mid_dampening(fract_mixing)
            # print(f"fract_mixing {fract_mixing} guid {self.sdh.guidance_scale}")
            if t_block == 0:
                if fixed_seeds is not None:
                    if idx_branch == 0:
                        self.set_seed(fixed_seeds[0])
                    elif idx_branch == list_nmb_branches[0] -1:
                        self.set_seed(fixed_seeds[1])
                list_latents = self.run_diffusion(text_embeddings_mix, idx_stop=idx_stop)
            else:
                # find parents latents
                b_parent1, b_parent2 = get_closest_idx(fract_mixing, self.tree_fracts[t_block-1])
                latents1 = self.tree_latents[t_block-1][b_parent1][-1]
                if fract_mixing == 0:
                    latents2 = latents1
                else:
                    latents2 = self.tree_latents[t_block-1][b_parent2][-1]
                idx_start = list_injection_idx_ext[t_block]
                fract_mixing_parental = (fract_mixing - self.tree_fracts[t_block-1][b_parent1]) / (self.tree_fracts[t_block-1][b_parent2] - self.tree_fracts[t_block-1][b_parent1]) 
                latents_for_injection = interpolate_spherical(latents1, latents2, fract_mixing_parental)
                list_latents = self.run_diffusion(text_embeddings_mix, latents_for_injection, idx_start=idx_start, idx_stop=idx_stop)
            
            self.tree_latents[t_block][idx_branch] = list_latents
            self.tree_status[t_block][idx_branch] = 'computed'
            
            # Convert latents to image directly for the last t_block
            if t_block == nmb_blocks_time-1:
                self.tree_final_imgs[idx_branch] = self.sdh.latent2image(list_latents[-1])
                self.tree_final_imgs_timing[idx_branch] = time.time() - time_start
            
        return self.tree_final_imgs
                

    def run_multi_transition(
            self,
            list_prompts: List[str],
            list_seeds: List[int] = None,
            list_nmb_branches: List[int] = None, 
            list_injection_strength: List[float] = None, 
            list_injection_idx: List[int] = None, 
            ms: MovieSaver = None,
            fps: float = 24,
            duration_single_trans: float = 15,
        ):
        r"""
        Runs multiple transitions and stitches them together. You can supply the seeds for each prompt.
        Args:
            list_prompts: List[float]:
                list of the prompts. There will be a transition starting from the first to the last.
            list_seeds: List[int] = None: 
                Random Seeds for each prompt.
            list_nmb_branches: List[int]:
                list of the number of branches for each injection.
            list_injection_strength: List[float]:
                list of injection strengths within interval [0, 1), values need to be increasing.
                Alternatively you can direclty specify the list_injection_idx.
            list_injection_idx: List[int]:
                list of injection strengths within interval [0, 1), values need to be increasing.
                Alternatively you can specify the list_injection_strength.
            ms: MovieSaver
                You need to spawn a moviesaver instance.
            fps: float:
                frames per second
            duration_single_trans: float:
                The duration of a single transition prompt[i] -> prompt[i+1].
                The duration of your movie will be duration_single_trans * len(list_prompts)
            
        """
        
        assert len(list_prompts) == len(list_seeds), "Supply the same number of prompts and seeds"
        
        if list_seeds is None:
            list_seeds = list(np.random.randint(0, 10e10, len(list_prompts)))
            
        
        for i in range(len(list_prompts)-1):
            print(f"Starting movie segment {i+1}/{len(list_prompts)-1}")
            
            if i==0:
                self.set_prompt1(list_prompts[i])
                self.set_prompt2(list_prompts[i+1])
                recycle_img1 = False    
            else:
                self.swap_forward()
                self.set_prompt2(list_prompts[i+1])
                recycle_img1 = True    
            
            local_seeds = [list_seeds[i], list_seeds[i+1]]
            list_imgs = self.run_transition(list_nmb_branches, list_injection_strength=list_injection_strength, list_injection_idx=list_injection_idx, recycle_img1=recycle_img1, fixed_seeds=local_seeds)
            list_imgs_interp = add_frames_linear_interp(list_imgs, fps, duration_single_trans)
            
            # Save movie frame
            for img in list_imgs_interp:
                ms.write_frame(img)
                
        ms.finalize()
        
        print("run_multi_transition: All completed.")


    @torch.no_grad()
    def run_diffusion(
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
        
        # Ensure correct num_inference_steps in Holder
        self.sdh.num_inference_steps = self.num_inference_steps
        
        if self.mode == 'standard':
            return self.sdh.run_diffusion_standard(text_embeddings, latents_for_injection=latents_for_injection, idx_start=idx_start, idx_stop=idx_stop, return_image=return_image)
        
        elif self.mode == 'inpaint':
            assert self.sdh.image_source is not None, "image_source is None. Please run init_inpainting first."
            assert self.sdh.mask_image is not None, "image_source is None. Please run init_inpainting first."
            return self.sdh.run_diffusion_inpaint(text_embeddings, latents_for_injection=latents_for_injection, idx_start=idx_start, idx_stop=idx_stop, return_image=return_image)

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
        self.init_mode('inpaint')
        self.sdh.init_inpainting(image_source, mask_image, init_empty)

   
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
        

    def swap_forward(self):
        r"""
        Moves over keyframe two -> keyframe one. Useful for making a sequence of transitions
        as in run_multi_transition()
        """ 
        # Move over all latents
        for t_block in range(len(self.tree_latents)):
            self.tree_latents[t_block][0] = self.tree_latents[t_block][-1]
        
        # Move over prompts and text embeddings
        self.prompt1 = self.prompt2
        self.text_embedding1 = self.text_embedding2
        
        # Final cleanup for extra sanity
        self.tree_final_imgs = [] 
        
        
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


def get_spacing(nmb_points:int, scaling: float):
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





#%% le main
if __name__ == "__main__":
    pass

#%%
"""
TODO Coding:
    RUNNING WITHOUT PROMPT!
    save value ranges, can it be trashed?
    in the middle: have more branches + lower guidance scale
    
TODO Other:
    github
    write text
    requirements
    make graphic explaining
    make colab
    license
    twitter et al
"""
