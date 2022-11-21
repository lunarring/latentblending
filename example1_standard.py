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
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import matplotlib.pyplot as plt
import torch
from movie_man import MovieSaver
import datetime
from typing import Callable, List, Optional, Union
import inspect
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
    
#%% Next let's set up all parameters

# We want 20 diffusion steps, begin with 2 branches, have 3 branches at step 12 (=0.6*20)
# 10 branches at step 16 (=0.8*20) and 24 branches at step 18 (=0.9*20)
# Furthermore we want seed 993621550 for keyframeA and seed 54878562 for keyframeB ()

num_inference_steps = 30 # Number of diffusion interations
list_nmb_branches = [2, 6, 30, 100] # Specify the branching structure 
list_injection_strength = [0.0, 0.3, 0.73, 0.93] # Specify the branching structure
width = 512 
height = 512
guidance_scale = 5
#fixed_seeds = [993621550, 326814432]
#fixed_seeds = [993621550, 888839807]
fixed_seeds = [993621550, 753528763]
    
lb = LatentBlending(pipe, device, height, width, num_inference_steps, guidance_scale)
prompt1 = "photo of a beautiful forest covered in white flowers, ambient light, very detailed, magic"
prompt2 = "photo of a mystical sculpture in the middle of the desert, warm sunlight, sand, eery feeling"
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)

imgs_transition = lb.run_transition(list_nmb_branches, list_injection_strength, fixed_seeds=fixed_seeds)

#%

# let's get more frames
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


# MOVIE TODO: ueberschreiben! bad prints.