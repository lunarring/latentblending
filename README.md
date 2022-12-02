# What is latent blending?

Latent blending allows you to generate smooth video transitions between two prompts. It is based on (stable diffusion 2.0)[https://stability.ai/blog/stable-diffusion-v2-release] and remixes the latent reprensetation using spherical linear interpolations. This results in imperceptible transitions, where one image slowly turns into another one. 

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
#### Models
[Download the Stable Diffusion 2.0 Standard Model](https://huggingface.co/stabilityai/stable-diffusion-2)

[Download the Stable Diffusion 2.0 Inpainting Model (optional)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

#### xformers efficient attention [(copied from stability)](https://github.com/Stability-AI/stablediffusion)
For more efficiency and speed on GPUs, 
we highly recommended installing the [xformers](https://github.com/facebookresearch/xformers)
library.

Tested on A100 with CUDA 11.4.
Installation needs a somewhat recent version of nvcc and gcc/g++, obtain those, e.g., via 
```commandline
export CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64=9.5.0
```

Then, run the following (compiling takes up to 30 min).

```commandline
cd ..
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ../stable-diffusion
```
Upon successful installation, the code will automatically default to [memory efficient attention](https://github.com/facebookresearch/xformers)
for the self- and cross-attention layers in the U-Net and autoencoder.

# How does it work
![](animation.gif)

what makes a transition a good transition?
* absence of movement
* every frame looks like a credible photo
