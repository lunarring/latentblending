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

from diffusers import AutoPipelineForText2Image
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

dh = DiffusersHolder(pipe)

lb = LatentBlending(dh)
lb.set_prompt1("photo of underwater landscape, fish, und the sea, incredible detail, high resolution")
lb.set_prompt2("rendering of an alien planet, strange plants, strange creatures, surreal")
lb.set_negative_prompt("blurry, ugly, pale")

# Run latent blending
lb.run_transition()

# Save movie
lb.write_movie_transition('movie_example1.mp4', duration_transition=12)
