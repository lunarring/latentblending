import torch
import warnings
from diffusers import AutoPipelineForText2Image
from latentblending.movie_util import concatenate_movies
from latentblending.blending_engine import BlendingEngine
import numpy as np
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
# pretrained_model_name_or_path = "stabilityai/sdxl-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16")
pipe.to('cuda')
be = BlendingEngine(pipe, do_compile=True)
be.set_negative_prompt("blurry, pale, low-res, lofi")
# %% Let's setup the multi transition
fps = 30
duration_single_trans = 10
be.set_dimensions((1024, 1024))
nmb_prompts = 20





# Specify a list of prompts below
#%%

list_prompts = []
list_prompts.append("high resolution ultra 8K image with lake and forest")
# list_prompts.append("strange and alien desolate lanscapes 8K")
# list_prompts.append("ultra high res psychedelic skyscraper city landscape 8K unreal engine")
list_prompts = list_prompts*10

be.set_prompt1(list_prompts[0])
be.compute_latents1(True)

#img = pipe(list_prompts[0]).images[0]


#%%
# import os
# import random

# # Directory containing the text files
# dir_prompts = "/raid/data/diffusion/flamengalo/prompts_surreal"

# # List to store the contents of selected text files
# list_prompts = []

# # List all files in the directory
# file_list = os.listdir(dir_prompts)

# # Shuffle the file list to get random files
# random.shuffle(file_list)

# # Loop through the first nmb_prompts files and read their contents
# for filename in file_list[:nmb_prompts]:
#     file_path = os.path.join(dir_prompts, filename)
#     try:
#         with open(file_path, 'r') as file:
#             content = file.read()
#             list_prompts.append(content)
#     except Exception as e:
#         print(f"except {e}")
#%%
fp_movie = f'surreal_nmb{len(list_prompts)}.mp4'
# Specify the seeds
list_seeds = np.random.randint(0, 1111111111111, len(list_prompts))



list_movie_parts = []
for i in range(len(list_prompts) - 1):
    # For a multi transition we can save some computation time and recycle the latents
    if i == 0:
        be.set_prompt1(list_prompts[i])
        be.set_prompt2(list_prompts[i + 1])
        recycle_img1 = False
    else:
        be.swap_forward()
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
