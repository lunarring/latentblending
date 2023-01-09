Latent blending enables lightning-fast video transitions with incredible smoothness between prompts. Powered by [stable diffusion 2.1](https://stability.ai/blog/stablediffusion2-1-release7-dec-2022), this method involves specific mixing of intermediate latent representations to create a seamless transition – with users having the option to fully customize the transition and run high-resolution upscaling.

# Quickstart
```python
fp_ckpt = 'path_to_SD2.ckpt'
fp_config = 'path_to_config.yaml'

sdh = StableDiffusionHolder(fp_ckpt, fp_config, 'cuda')
lb = LatentBlending(sdh)

lb.load_branching_profile(quality='medium', depth_strength=0.4)
lb.set_prompt1('photo of my first prompt1')
lb.set_prompt2('photo of my second prompt')

imgs_transition = lb.run_transition()
```
## Gradio UI
To run the UI on your local machine, run `gradio_ui.py`
You can find the [most relevant parameters here](parameters.md)

## Example 1: Simple transition
![](example1.jpg)
To run a simple transition between two prompts, run `example1_standard.py`

## Example 2: Inpainting transition
![](example2.jpg)
To run a transition between two prompts where you want some part of the image to remain static, run `example2_inpaint.py`

## Example 3: Multi transition
To run multiple transition between K prompts, resulting in a stitched video, run `example3_multitrans.py`

## Example 4: High-resolution with upscaling
![](example4.jpg)
You can run a high-res transition using the x4 upscaling model in a two-stage procedure, see `example4_upscaling.py`

# Customization

## Most relevant parameters

### Change the height/width
```python 
lb.set_height(512)
lb.set_width(1024)
```
### Change guidance scale
```python 
lb.set_guidance_scale(5.0)
```
### depth_strength / list_injection_strength
The strength of the diffusion iterations determines when the blending process will begin. A value close to zero results in more creative and intricate outcomes, while a value closer to one indicates a simpler alpha blending. However, low values may also bring about the introduction of additional objects and motion.


## Set up the branching structure

There are three ways to change the branching structure.
### Presets
```python 
quality = 'medium' #choose from lowest, low, medium, high, ultra
depth_strength = 0.5 # see above (Most relevant parameters)

lb.load_branching_profile(quality, depth_strength)
```

### Autosetup tree setup
```python 
num_inference_steps = 30 # the number of diffusion steps
list_nmb_branches = [2, 4, 8, 20]
list_injection_strength = [0.0, 0.3, 0.5, 0.9]

lb.autosetup_branching(num_inference_steps, list_nmb_branches, list_injection_strength)
```

### Fully manual
```python 
depth_strength = 0.5 # see above (Most relevant parameters)
num_inference_steps = 30 # the number of diffusion steps
nmb_branches_final = 20 # how many diffusion images will be generated for the transition

lb.setup_branching(depth_strength, num_inference_steps, nmb_branches_final)
```



# Installation
#### Packages
```commandline
pip install -r requirements.txt
```
#### Download Models from Huggingface
[Download the Stable Diffusion v2-1_768 Model](https://huggingface.co/stabilityai/stable-diffusion-2-1)

[Download the Stable Diffusion Inpainting Model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

[Download the Stable Diffusion x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)

#### (Optional but recommended) Install [Xformers](https://github.com/facebookresearch/xformers)
With xformers, stable diffusion will run faster with smaller memory inprint. Necessary for higher resolutions / upscaling model.

```commandline
conda install xformers -c xformers/label/dev
```

Alternatively, you can build it from source:
```commandline
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
```

# How does it work
![](animation.gif)

what makes a transition a good transition?
* absence of movement
* every frame looks like a credible photo
