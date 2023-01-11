# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer @j_stelzer
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
dp_img = "upscaling_bleding" # the results will be saved in this folder
prompt1 = "photo of mount vesuvius erupting a terrifying pyroclastic ash cloud"
prompt2 = "photo of a inside a building full of ash, fire, death, destruction, explosions"
fixed_seeds = [5054613, 1168652]

width = 512
height = 384
num_inference_steps_lores = 40
nmb_branches_final_lores = 10
depth_strength_lores = 0.5

device = "cuda" 
fp_ckpt_lores = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 
fp_config_lores = 'configs/v2-inference.yaml'

#%% Define vars for high-resoltion pass
fp_ckpt_hires = "../stable_diffusion_models/ckpt/x4-upscaler-ema.ckpt"
fp_config_hires = 'configs/x4-upscaling.yaml'
depth_strength_hires = 0.65
num_inference_steps_hires = 100
nmb_branches_final_hires = 6

#%% Run low-res pass
sdh = StableDiffusionHolder(fp_ckpt_lores, fp_config_lores, device)
lb = LatentBlending(sdh)
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
lb.set_width(width)
lb.set_height(height)
lb.run_upscaling_step1(dp_img, depth_strength_lores, num_inference_steps_lores, nmb_branches_final_lores, fixed_seeds)

#%% Run high-res pass
sdh = StableDiffusionHolder(fp_ckpt_hires, fp_config_hires)
lb = LatentBlending(sdh) 
lb.run_upscaling_step2(dp_img, depth_strength_hires, num_inference_steps_hires, nmb_branches_final_hires)