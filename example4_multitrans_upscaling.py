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

import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings
warnings.filterwarnings('ignore')
import warnings
from latent_blending import LatentBlending
from stable_diffusion_holder import StableDiffusionHolder
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download

# %% Define vars for low-resoltion pass
list_prompts = []
list_prompts.append("surrealistic statue made of glitter and dirt, standing in a lake, atmospheric light, strange glow")
list_prompts.append("statue of a mix between a tree and human, made of marble, incredibly detailed")
list_prompts.append("weird statue of a frog monkey, many colors, standing next to the ruins of an ancient city")
list_prompts.append("statue of a spider that looked like a human")
list_prompts.append("statue of a bird that looked like a scorpion")
list_prompts.append("statue of an ancient cybernetic messenger annoucing good news, golden, futuristic")

# You can optionally specify the seeds
list_seeds = [954375479, 332539350, 956051013, 408831845, 250009012, 675588737]

width = 512
height = 384
duration_single_trans = 6
num_inference_steps_lores = 40
nmb_max_branches_lores = 10
depth_strength_lores = 0.5

fp_ckpt_lores = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", filename="v2-1_512-ema-pruned.ckpt")

# %% Define vars for high-resoltion pass
fp_ckpt_hires = hf_hub_download(repo_id="stabilityai/stable-diffusion-x4-upscaler", filename="x4-upscaler-ema.ckpt")
depth_strength_hires = 0.65
num_inference_steps_hires = 100
nmb_branches_final_hires = 6

# %% Run low-res pass
sdh = StableDiffusionHolder(fp_ckpt_lores)
t_compute_max_allowed = 12  # Per segment
lb = LatentBlending(sdh)

list_movie_dirs = []
for i in range(len(list_prompts) - 1):
    # For a multi transition we can save some computation time and recycle the latents
    if i == 0:
        lb.set_prompt1(list_prompts[i])
        lb.set_prompt2(list_prompts[i + 1])
        recycle_img1 = False
    else:
        lb.swap_forward()
        lb.set_prompt2(list_prompts[i + 1])
        recycle_img1 = True

    dp_movie_part = f"tmp_part_{str(i).zfill(3)}"
    fp_movie_part = os.path.join(dp_movie_part, "movie_lowres.mp4")
    os.makedirs(dp_movie_part, exist_ok=True)
    fixed_seeds = list_seeds[i:i + 2]

    # Run latent blending
    lb.run_transition(
        depth_strength=depth_strength_lores,
        nmb_max_branches=nmb_max_branches_lores,
        fixed_seeds=fixed_seeds)

    # Save movie and images (needed for upscaling!)
    lb.write_movie_transition(fp_movie_part, duration_single_trans)
    lb.write_imgs_transition(dp_movie_part)
    list_movie_dirs.append(dp_movie_part)

# %% Run high-res pass on each segment
sdh = StableDiffusionHolder(fp_ckpt_hires)
lb = LatentBlending(sdh)
for dp_part in list_movie_dirs:
    lb.run_upscaling(dp_part, depth_strength_hires, num_inference_steps_hires, nmb_branches_final_hires)

# %% concatenate into one long movie
list_fp_movies = []
for dp_part in list_movie_dirs:
    fp_movie = os.path.join(dp_part, "movie_highres.mp4")
    assert os.path.isfile(fp_movie)
    list_fp_movies.append(fp_movie)

fp_final = "example4.mp4"
concatenate_movies(fp_final, list_fp_movies)
