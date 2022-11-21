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
import time
import subprocess
import warnings
import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib.pyplot as plt
import torch
from movie_man import MovieSaver
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
torch.set_grad_enabled(False)

#%% First let us spawn a diffusers pipe using DDIMScheduler
device = "cuda:0"
model_path = "../stable_diffusion_models/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to(device)


#%% Let's make a source image and mask.
height = 512
width = 512
num_inference_steps = 30
guidance_scale = 5
fixed_seeds = [629575320, 670154945]

lb = LatentBlending(pipe, device, height, width, num_inference_steps, guidance_scale)
prompt1 = "photo of a futuristic alien temple in a desert, mystic, glowing, organic, intricate, sci-fi movie, mesmerizing, scary"
lb.set_prompt1(prompt1)
lb.init_inpainting(init_empty=True)
lb.set_seed(fixed_seeds[0])
image_source = lb.run_diffusion(lb.text_embedding1, return_image=True)
mask_image = 255*np.ones([512,512], dtype=np.uint8)
mask_image[160:250, 200:320] = 0
mask_image = Image.fromarray(mask_image)


#%% Next let's set up all parameters
# FIXME below fix numbers
# We want 20 diffusion steps, begin with 2 branches, have 3 branches at step 12 (=0.6*20)
# 10 branches at step 16 (=0.8*20) and 24 branches at step 18 (=0.9*20)
# Furthermore we want seed 993621550 for keyframeA and seed 54878562 for keyframeB ()

num_inference_steps = 20 # Number of diffusion interations
list_nmb_branches = [2, 3, 10, 24] # Branching structure: how many branches
list_injection_strength = [0.0, 0.6, 0.8, 0.9] # Branching structure: how deep is the blending
width = 512 
height = 512
guidance_scale = 5
fixed_seeds = [993621550, 280335986]
    
lb = LatentBlending(pipe, device, height, width, num_inference_steps, guidance_scale)
prompt1 = "photo of a futuristic alien temple in a desert, mystic, glowing, organic, intricate, sci-fi movie, mesmerizing, scary"
prompt2 = "aerial photo of a futuristic alien temple in a coastal area, waves clashing"
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
lb.init_inpainting(image_source, mask_image)

imgs_transition = lb.run_transition(list_nmb_branches, list_injection_strength, fixed_seeds=fixed_seeds)

# let's get more cheap frames via linear interpolation
duration_transition = 12
fps = 60
imgs_transition_ext = add_frames_linear_interp(imgs_transition, duration_transition, fps)

# movie saving
fp_movie = f"/home/lugo/tmp/latentblending/bobo_incoming.mp4"
if os.path.isfile(fp_movie):
    os.remove(fp_movie)
ms = MovieSaver(fp_movie, fps=fps, profile='save')
for img in tqdm(imgs_transition_ext):
    ms.write_frame(img)
ms.finalize()


