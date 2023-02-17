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

#%% First let us spawn a stable diffusion holder
fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_768-ema-pruned.ckpt"
sdh = StableDiffusionHolder(fp_ckpt)

    
#%% Next let's set up all parameters
depth_strength = 0.65 # Specifies how deep (in terms of diffusion iterations the first branching happens)
t_compute_max_allowed = 15 # Determines the quality of the transition in terms of compute time you grant it
fixed_seeds = [69731932, 504430820]
    
prompt1 = "photo of a beautiful cherry forest covered in white flowers, ambient light, very detailed, magic"
prompt2 = "photo of an golden statue with a funny hat, surrounded by ferns and vines, grainy analog photograph, mystical ambience, incredible detail"

duration_transition = 12 # In seconds
fps = 30

# Spawn latent blending
lb = LatentBlending(sdh)
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
#FIXME AssertionError: Either specify t_compute_max_allowed or nmb_max_branches
# Run latent blending
imgs_transition = lb.run_transition(
    depth_strength = depth_strength,
    t_compute_max_allowed = t_compute_max_allowed,
    fixed_seeds = fixed_seeds
    )
# Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
imgs_transition_ext = add_frames_linear_interp(imgs_transition, duration_transition, fps)

# Save as MP4
fp_movie = "movie_example1.mp4"
if os.path.isfile(fp_movie):
    os.remove(fp_movie)
ms = MovieSaver(fp_movie, fps=fps, shape_hw=[sdh.height, sdh.width])
for img in tqdm(imgs_transition_ext):
    ms.write_frame(img)
ms.finalize()


