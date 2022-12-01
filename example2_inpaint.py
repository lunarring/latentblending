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
from movie_util import MovieSaver
from typing import Callable, List, Optional, Union
from latent_blending import LatentBlending, add_frames_linear_interp
from stable_diffusion_holder import StableDiffusionHolder
torch.set_grad_enabled(False)

#%% First let us spawn a stable diffusion holder
device = "cuda"
deepth_strength = 0.65
fp_ckpt= "../stable_diffusion_models/ckpt/512-inpainting-ema.ckpt"
fp_config = '../stablediffusion/configs//stable-diffusion/v2-inpainting-inference.yaml'

sdh = StableDiffusionHolder(fp_ckpt, fp_config, device)


#%% Let's make a source image and mask.
quality = 'low'
seed0 = 190791709

lb = LatentBlending(sdh)
lb.autosetup_branching(quality=quality, deepth_strength=deepth_strength)
prompt1 = "photo of a futuristic alien temple in a desert, mystic, glowing, organic, intricate, sci-fi movie, mesmerizing, scary"
lb.set_prompt1(prompt1)
lb.init_inpainting(init_empty=True)
lb.set_seed(seed0)
list_latents = lb.run_diffusion(lb.text_embedding1)
image_source = lb.sdh.latent2image(list_latents[-1])

mask_image = 255*np.ones([512,512], dtype=np.uint8)
mask_image[340:420, 170:280] = 0
mask_image = Image.fromarray(mask_image)


#%% Next let's set up all parameters
lb.inject_latents(list_latents, inject_img1=True)

fixed_seeds = [seed0, 6579436]
    
prompt1 = "photo of a futuristic alien temple in a desert, mystic, glowing, organic, intricate, sci-fi movie, mesmerizing, scary"
prompt2 = "aerial photo of a futuristic alien temple in a blue coastal area, the sun is shining with a bright light"
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
lb.init_inpainting(image_source, mask_image)
imgs_transition = lb.run_transition(recycle_img1=True, fixed_seeds=fixed_seeds)

#% let's get more cheap frames via linear interpolation
duration_transition = 3
fps = 60
imgs_transition_ext = add_frames_linear_interp(imgs_transition, duration_transition, fps)

# movie saving
fp_movie = "/home/lugo/git/latentblending/test.mp4"
if os.path.isfile(fp_movie):
    os.remove(fp_movie)
ms = MovieSaver(fp_movie, fps=fps, shape_hw=[lb.height, lb.width])
for img in tqdm(imgs_transition_ext):
    ms.write_frame(img)
ms.finalize()

