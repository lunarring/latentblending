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
import torch
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import warnings
import torch
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
from movie_util import MovieSaver
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
from stable_diffusion_holder import StableDiffusionHolder
torch.set_grad_enabled(False)


#%% First let us spawn a stable diffusion holder
device = "cuda:0"
num_inference_steps = 20 # Number of diffusion interations
fp_ckpt = "../stable_diffusion_models/ckpt/768-v-ema.ckpt"
fp_config = '../stablediffusion/configs/stable-diffusion/v2-inference-v.yaml'

sdh = StableDiffusionHolder(fp_ckpt, fp_config, device, num_inference_steps=num_inference_steps)
    
#%% Next let's set up all parameters
num_inference_steps = 30 # Number of diffusion interations
list_nmb_branches = [2, 3, 10, 24]#, 50] # Branching structure: how many branches
list_injection_strength = [0.0, 0.6, 0.8, 0.9]#, 0.95] # Branching structure: how deep is the blending

guidance_scale = 5
fps = 30
duration_target = 10
width = 512
height = 512

lb = LatentBlending(sdh, num_inference_steps, guidance_scale)

list_prompts = []
list_prompts.append("photo of a beautiful forest covered in white flowers, ambient light, very detailed, magic")
list_prompts.append("photo of an golden statue with a funny hat, surrounded by ferns and vines, grainy analog photograph, mystical ambience, incredible detail")


for k, prompt in enumerate(list_prompts):
    # k = 6
    
    # prompt = list_prompts[k]
    for i in range(10):
        lb.set_prompt1(prompt)
        
        seed = np.random.randint(999999999)
        lb.set_seed(seed)
        plt.imshow(lb.run_diffusion(lb.text_embedding1, return_image=True))
        plt.title(f"prompt {k}, seed {i} {seed}")
        plt.show()
        print(f"prompt {k} seed {seed} trial {i}")
        
#%%

"""
69731932, 504430820
"""