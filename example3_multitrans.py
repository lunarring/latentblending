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
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import matplotlib.pyplot as plt
import torch
from movie_util import MovieSaver
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
torch.set_grad_enabled(False)

#%% First let us spawn a diffusers pipe using DDIMScheduler
device = "cuda:0"
model_path = "../stable_diffusion_models/stable-diffusion-v1-5"

scheduler = DDIMScheduler(beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False)
            
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    scheduler=scheduler,
    use_auth_token=True
)
pipe = pipe.to(device)
    
#%% MULTITRANS
# XXX FIXME AssertionError: Need to supply floats for list_injection_strength
# GO AS DEEP AS POSSIBLE WITHOUT CAUSING MOTION

num_inference_steps = 100 # Number of diffusion interations
#list_nmb_branches = [2, 12, 24, 55, 77] # Branching structure: how many branches
#list_injection_strength = [0.0, 0.35, 0.5, 0.65, 0.95] # Branching structure: how deep is the blending
list_nmb_branches = list(np.linspace(2, 600, 15).astype(int)) #
list_injection_strength = list(np.linspace(0.45, 0.97, 14).astype(np.float32)) # Branching structure: how deep is the blending
list_injection_strength = [float(x) for x in list_injection_strength]
list_injection_strength.insert(0,0.0)

width = 512
height = 512
guidance_scale = 5
fps = 30
duration_target = 20
width = 512
height = 512

lb = LatentBlending(pipe, device, height, width, num_inference_steps, guidance_scale)

#list_nmb_branches = [2, 3, 10, 24] # Branching structure: how many branches
#list_injection_strength = [0.0, 0.6, 0.8, 0.9] # 

list_prompts = []
list_prompts.append("surrealistic statue made of glitter and dirt, standing in a lake, atmospheric light, strange glow")
list_prompts.append("statue of a mix between a tree and human, made of marble, incredibly detailed")
list_prompts.append("weird statue of a frog monkey, many colors, standing next to the ruins of an ancient city")
list_prompts.append("statue made of hot metal, bizzarre, dark clouds in the sky")
list_prompts.append("statue of a spider that looked like a human")
list_prompts.append("statue of a bird that looked like a scorpion")
list_prompts.append("statue of an ancient cybernetic messenger annoucing good news, golden, futuristic")


list_seeds = [234187386, 422209351, 241845736, 28652396, 783279867, 831049796, 234903931]


fp_movie = "/home/lugo/tmp/latentblending/bubu.mp4"
ms = MovieSaver(fp_movie, fps=fps)

for i in range(len(list_prompts)-1):
    print(f"Starting movie segment {i+1}/{len(list_prompts)-1}")
    
    if i==0:
        lb.set_prompt1(list_prompts[i])
        lb.set_prompt2(list_prompts[i+1])
        recycle_img1 = False    
    else:
        lb.swap_forward()
        lb.set_prompt2(list_prompts[i+1])
        recycle_img1 = True    
    
    local_seeds = [list_seeds[i], list_seeds[i+1]]
    list_imgs = lb.run_transition(list_nmb_branches, list_injection_strength, recycle_img1=recycle_img1, fixed_seeds=local_seeds)
    list_imgs_interp = add_frames_linear_interp(list_imgs, fps, duration_target)
    
    # Save movie frame
    for img in list_imgs_interp:
        ms.write_frame(img)
        
ms.finalize()

#%%
#for img in lb.tree_final_imgs:
#    if img is not None:
#        ms.write_frame(img)
#        
#ms.finalize()      

