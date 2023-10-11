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
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings
warnings.filterwarnings('ignore')
import warnings
from latent_blending import LatentBlending
from diffusers_holder import DiffusersHolder
from diffusers import DiffusionPipeline
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe.to('cuda:1')
dh = DiffusersHolder(pipe)

# %% Let's setup the multi transition
fps = 30
duration_single_trans = 20
depth_strength = 0.25  # Specifies how deep (in terms of diffusion iterations the first branching happens)

# Specify a list of prompts below
list_prompts = []
list_prompts.append("A panoramic photo of a sentient mirror maze amidst a neon-lit forest, where bioluminescent mushrooms glow eerily, reflecting off the mirrors, and cybernetic crows, with silver wings and ruby eyes, perch ominously, David Lynch, Gaspar Noé, Photograph.")
list_prompts.append("An unsettling tableau of spectral butterflies with clockwork wings, swirling around an antique typewriter perched precariously atop a floating, gnarled tree trunk, a stormy twilight sky, David Lynch's dreamscape, meticulously crafted.")
# list_prompts.append("A haunting tableau of an antique dollhouse swallowed by a giant venus flytrap under the neon glow of an alien moon, its uncanny light reflecting from shattered porcelain faces and marbles, in a quiet, abandoned amusement park.")


# You can optionally specify the seeds
list_seeds = [95437579, 33259350, 956051013, 408831845, 250009012, 675588737]
t_compute_max_allowed = 20  # per segment
fp_movie = 'movie_example2.mp4'
lb = LatentBlending(dh)
lb.dh.set_dimensions(1024, 704)
lb.dh.set_num_inference_steps(40)


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
        recycle_img1 = recycle_img1,
        depth_strength=depth_strength,
        t_compute_max_allowed=t_compute_max_allowed,
        fixed_seeds=fixed_seeds)

    # Save movie
    lb.write_movie_transition(fp_movie_part, duration_single_trans)
    list_movie_parts.append(fp_movie_part)

# Finally, concatente the result
concatenate_movies(fp_movie, list_movie_parts)
