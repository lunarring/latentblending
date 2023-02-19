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

fp_ckpt = "../stable_diffusion_models/riffusion/riffusion-model-v1.ckpt"
fp_config = "configs/v1-inference.yaml"

sdh = StableDiffusionHolder(fp_ckpt, fp_config)
    
#%% Next let's set up all parameters
depth_strength = 0.25 # Specifies how deep (in terms of diffusion iterations the first branching happens)
t_compute_max_allowed = 10 # Determines the quality of the transition in terms of compute time you grant it
fixed_seeds = [69731932, 504430820]
    
prompt1 = "ambient pad trippy"
prompt2 = "ambient pad psychedelic"


# Spawn latent blending
lb = LatentBlending(sdh)
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)


lb.branch1_influence = 0.0
lb.branch1_max_depth_influence = 0.65
lb.branch1_influence_decay = 0.8

lb.parental_influence = 0.0
lb.parental_max_depth_influence = 1.0
lb.parental_influence_decay = 1.0    

# x = lb.compute_latents1(True)
# Image.fromarray(x).save("//home/lugo/git/riffusion/testA.png")


# xxx
# Run latent blending
lb.run_transition(
    depth_strength = depth_strength,
    t_compute_max_allowed = t_compute_max_allowed,
    fixed_seeds = fixed_seeds
    )
dp_save = "/home/lugo/git/riffusion/latentblending"
for i in range(len(lb.tree_final_imgs)):
    fp_save = os.path.join(dp_save, f"sound_{str(i).zfill(3)}.png")
    Image.fromarray(lb.tree_final_imgs[i]).save(fp_save)
    
    
#%% take this file
"""
#!/bin/bash

FOLDER="/home/lugo/git/riffusion/latentblending"

for file in "$FOLDER"/*.png; do
  filename=$(basename -- "$file")   # get the filename without the folder path
  filename_no_ext="${filename%.*}"  # remove the file extension (i.e. png)

  # call riffusion.cli to convert the png to wav
  python -m riffusion.cli image-to-audio --image "$file" --audio "${FOLDER}/${filename_no_ext}.wav"
done
"""
