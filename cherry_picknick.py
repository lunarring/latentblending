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
use_inpaint = True

device = "cuda"
fp_ckpt= "../stable_diffusion_models/ckpt/512-inpainting-ema.ckpt"
fp_config = '../stablediffusion/configs//stable-diffusion/v2-inpainting-inference.yaml'
    
    
# fp_ckpt = "../stable_diffusion_models/ckpt/768-v-ema.ckpt"
# fp_config = '../stablediffusion/configs/stable-diffusion/v2-inference-v.yaml'

sdh = StableDiffusionHolder(fp_ckpt, fp_config, device)



#%% Next let's set up all parameters
num_inference_steps = 30 # Number of diffusion interations

guidance_scale = 5

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
#%% Let's make a source image and mask.
k=0
for i in range(10):
    seed = 190791709# np.random.randint(999999999)
# seed0 = 629575320

    lb = LatentBlending(sdh)
    lb.autosetup_branching(quality='medium', depth_strength=0.65)
    
    prompt1 = "photo of a futuristic alien temple in a desert, mystic, glowing, organic, intricate, sci-fi movie, mesmerizing, scary"
    lb.set_prompt1(prompt1)
    lb.init_inpainting(init_empty=True)
    lb.set_seed(seed)
    plt.imshow(lb.run_diffusion(lb.text_embedding1, return_image=True))
    plt.title(f"prompt1 {k}, seed {i} {seed}")
    plt.show()
    print(f"prompt1 {k} seed {seed} trial {i}")
    
    xx

#%%
mask_image = 255*np.ones([512,512], dtype=np.uint8)
mask_image[340:420, 170:280, ] = 0
mask_image = Image.fromarray(mask_image)
        
#%%

"""
69731932, 504430820
"""