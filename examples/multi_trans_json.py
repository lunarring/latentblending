import torch
import warnings
from diffusers import AutoPipelineForText2Image
from latentblending.blending_engine import BlendingEngine
from lunar_tools import concatenate_movies
import numpy as np
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

import json
# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_model_name_or_path = "stabilityai/sdxl-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16")
pipe.to('cuda')
be = BlendingEngine(pipe, do_compile=False)

fp_movie = f'test.mp4'
fp_json = "movie_240221_1520.json"
duration_single_trans = 10

# Load the JSON data from the file
with open(fp_json, 'r') as file:
    data = json.load(file)

# Set up width, height, num_inference steps
width = data[0]["width"]
height = data[0]["height"]
num_inference_steps = data[0]["num_inference_steps"]

be.set_dimensions((width, height))
be.set_num_inference_steps(num_inference_steps)

# Initialize lists for prompts, negative prompts, and seeds
list_prompts = []
list_negative_prompts = []
list_seeds = []

# Extract prompts, negative prompts, and seeds from the data
for item in data[1:]:  # Skip the first item as it contains settings
    list_prompts.append(item["prompt"])
    list_negative_prompts.append(item["negative_prompt"])
    list_seeds.append(item["seed"])


list_movie_parts = []
for i in range(len(list_prompts) - 1):
    # For a multi transition we can save some computation time and recycle the latents
    if i == 0:
        be.set_prompt1(list_prompts[i])
        be.set_negative_prompt(list_negative_prompts[i])
        be.set_prompt2(list_prompts[i + 1])
        recycle_img1 = False
    else:
        be.swap_forward()
        be.set_negative_prompt(list_negative_prompts[i+1])
        be.set_prompt2(list_prompts[i + 1])
        recycle_img1 = True

    fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
    fixed_seeds = list_seeds[i:i + 2]
    # Run latent blending
    be.run_transition(
        recycle_img1=recycle_img1,
        fixed_seeds=fixed_seeds)

    # Save movie
    be.write_movie_transition(fp_movie_part, duration_single_trans)
    list_movie_parts.append(fp_movie_part)

# Finally, concatente the result
concatenate_movies(fp_movie, list_movie_parts)
print(f"DONE! MOVIE SAVED IN {fp_movie}")