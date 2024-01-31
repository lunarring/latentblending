import torch
import warnings
from diffusers import AutoPipelineForText2Image
from latentblending.blending_engine import BlendingEngine

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

be = BlendingEngine(pipe)
be.set_prompt1("photo of underwater landscape, fish, und the sea, incredible detail, high resolution")
be.set_prompt2("rendering of an alien planet, strange plants, strange creatures, surreal")
be.set_negative_prompt("blurry, ugly, pale")

# Run latent blending
be.run_transition()

# Save movie
be.write_movie_transition('movie_example1.mp4', duration_transition=12)
