# What is latent blending?

Latent blending allows you to generate smooth video transitions between two prompts. It is based on [stable diffusion 2.1](https://stability.ai/blog/stablediffusion2-1-release7-dec-2022) and remixes the latent reprensetation using spherical linear interpolations. This results in imperceptible transitions, where one image slowly turns into another one. 

# Example 1: simple transition
(mp4), code

# Example 2: inpainting transition
(mp4), code

# Example 3: concatenated transition
(mp4), code

# Relevant parameters


# Installation
#### Packages
```commandline
pip install -r requirements.txt
```
#### Download Models from Huggingface
[Download the Stable Diffusion v2-1_768 Model](https://huggingface.co/stabilityai/stable-diffusion-2-1)

[Download the Stable Diffusion 2.0 Inpainting Model (optional)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

#### (Optional) Install [Xformers](https://github.com/facebookresearch/xformers)
With xformers, stable diffusion 2 will run a bit faster. The recommended way of installation is via the supplied binaries (Linux).

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
