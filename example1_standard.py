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
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe.to('cuda')
dh = DiffusersHolder(pipe)
# %% Next let's set up all parameters
depth_strength = 0.55  # Specifies how deep (in terms of diffusion iterations the first branching happens)
t_compute_max_allowed = 60  # Determines the quality of the transition in terms of compute time you grant it
num_inference_steps = 30
size_output = (1024, 1024)

prompt1 = "underwater landscape, fish, und the sea, incredible detail, high resolution"
prompt2 = "rendering of an alien planet, strange plants, strange creatures, surreal"
negative_prompt = "blurry, ugly, pale"  # Optional

fp_movie = 'movie_example1.mp4'
duration_transition = 12  # In seconds

# Spawn latent blending
lb = LatentBlending(dh)
lb.set_prompt1(prompt1)
lb.set_prompt2(prompt2)
lb.set_dimensions(size_output)
lb.set_negative_prompt(negative_prompt)

# Run latent blending
lb.run_transition(
    depth_strength=depth_strength,
    num_inference_steps=num_inference_steps,
    t_compute_max_allowed=t_compute_max_allowed)

# Save movie
lb.write_movie_transition(fp_movie, duration_transition)
