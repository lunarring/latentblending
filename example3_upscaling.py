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
# import matplotlib.pyplot as plt
import torch
from movie_util import MovieSaver
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
from stable_diffusion_holder import StableDiffusionHolder
torch.set_grad_enabled(False)

#%% Define vars for low-resoltion pass
prompt1 = "photo of mount vesuvius erupting a terrifying pyroclastic ash cloud"
prompt2 = "photo of a inside a building full of ash, fire, death, destruction, explosions"
fixed_seeds = [5054613, 1168652]

width = 512
height = 384
num_inference_steps_lores = 40
nmb_max_branches_lores = 10
depth_strength_lores = 0.5

fp_ckpt_lores = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 

#%% Define vars for high-resoltion pass
fp_ckpt_hires = "../stable_diffusion_models/ckpt/x4-upscaler-ema.ckpt"
depth_strength_hires = 0.65
num_inference_steps_hires = 100
nmb_branches_final_hires = 6
dp_imgs = "tmp_transition" # folder for results and intermediate steps


#%% Run low-res pass
sdh = StableDiffusionHolder(fp_ckpt_lores)

#%%
lb = LatentBlending(sdh)
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
lb.set_width(width)
lb.set_height(height)

# Run latent blending
lb.run_transition(
    depth_strength = depth_strength_lores,
    nmb_max_branches = nmb_max_branches_lores,
    fixed_seeds = fixed_seeds
    )

lb.write_imgs_transition(dp_imgs)

#%% Run high-res pass
sdh = StableDiffusionHolder(fp_ckpt_hires)
lb = LatentBlending(sdh) 
lb.run_upscaling(dp_imgs, depth_strength_hires, num_inference_steps_hires, nmb_branches_final_hires)
