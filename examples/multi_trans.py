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
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to('cuda')
be = BlendingEngine(pipe)

# %% Let's setup the multi transition
fps = 30
duration_single_trans = 10

# Specify a list of prompts below
list_prompts = []
list_prompts.append("Photo of a house, high detail")
list_prompts.append("Photo of an elephant in african savannah")
list_prompts.append("photo of a house, high detail")


# Specify the seeds
list_seeds = np.random.randint(0, 10^9, len(list_prompts))
fp_movie = 'movie_example2.mp4'


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
