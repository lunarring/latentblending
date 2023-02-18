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
import torch
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import warnings
import torch
from tqdm.auto import tqdm
from PIL import Image
import torch
from movie_util import MovieSaver, concatenate_movies
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
from stable_diffusion_holder import StableDiffusionHolder
torch.set_grad_enabled(False)

#%% First let us spawn a stable diffusion holder
fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt"
# fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_768-ema-pruned.ckpt"
sdh = StableDiffusionHolder(fp_ckpt)

    
#%% Let's setup the multi transition
fps = 30
duration_single_trans = 6
depth_strength = 0.55 #Specifies how deep (in terms of diffusion iterations the first branching happens)

# Specify a list of prompts below
list_prompts = []
list_prompts.append("surrealistic statue made of glitter and dirt, standing in a lake, atmospheric light, strange glow")
list_prompts.append("statue of a mix between a tree and human, made of marble, incredibly detailed")
list_prompts.append("weird statue of a frog monkey, many colors, standing next to the ruins of an ancient city")
list_prompts.append("statue of a spider that looked like a human")
list_prompts.append("statue of a bird that looked like a scorpion")
list_prompts.append("statue of an ancient cybernetic messenger annoucing good news, golden, futuristic")

# You can optionally specify the seeds
list_seeds = [954375479, 332539350, 956051013, 408831845, 250009012, 675588737]
t_compute_max_allowed = 12 # per segment
fp_movie = 'movie_example2.mp4'
lb = LatentBlending(sdh)

list_movie_parts = [] #
for i in range(len(list_prompts)-1):
    # For a multi transition we can save some computation time and recycle the latents
    if i==0:
        lb.set_prompt1(list_prompts[i])
        lb.set_prompt2(list_prompts[i+1])
        recycle_img1 = False
    else:
        lb.swap_forward()
        lb.set_prompt2(list_prompts[i+1])
        recycle_img1 = True   
        
    fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
    
    fixed_seeds = list_seeds[i:i+2]
    
    # Run latent blending
    lb.run_transition(
        depth_strength = depth_strength,
        t_compute_max_allowed = t_compute_max_allowed,
        fixed_seeds = fixed_seeds
        )
    
    # Save movie
    lb.write_movie_transition(fp_movie_part, duration_single_trans)
    list_movie_parts.append(fp_movie_part)

# Finally, concatente the result
concatenate_movies(fp_movie, list_movie_parts)