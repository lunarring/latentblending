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

import torch
import warnings
from latent_blending import LatentBlending
from diffusers_holder import DiffusersHolder
from diffusers import DiffusionPipeline
from movie_util import concatenate_movies
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe.to('cuda')
dh = DiffusersHolder(pipe)

# %% Let's setup the multi transition
fps = 30
duration_single_trans = 20
depth_strength = 0.25  # Specifies how deep (in terms of diffusion iterations the first branching happens)
size_output = (1280, 768)
num_inference_steps = 30

# Specify a list of prompts below
list_prompts = []
list_prompts.append("Photo of a house, high detail")
list_prompts.append("Photo of an elephant in african savannah")
list_prompts.append("photo of a house, high detail")


# You can optionally specify the seeds
list_seeds = [95437579, 33259350, 956051013]
t_compute_max_allowed = 20  # per segment
fp_movie = 'movie_example2.mp4'
lb = LatentBlending(dh)
lb.set_dimensions(size_output)
lb.dh.set_num_inference_steps(num_inference_steps)


list_movie_parts = []
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

    fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
    fixed_seeds = list_seeds[i:i + 2]
    # Run latent blending
    lb.run_transition(
        recycle_img1=recycle_img1,
        depth_strength=depth_strength,
        t_compute_max_allowed=t_compute_max_allowed,
        fixed_seeds=fixed_seeds)

    # Save movie
    lb.write_movie_transition(fp_movie_part, duration_single_trans)
    list_movie_parts.append(fp_movie_part)

# Finally, concatente the result
concatenate_movies(fp_movie, list_movie_parts)
