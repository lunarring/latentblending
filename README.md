# Quickstart

Latent blending enables video transitions with incredible smoothness between prompts, computed within seconds. Powered by [stable diffusion XL](https://stability.ai/stable-diffusion), this method involves specific mixing of intermediate latent representations to create a seamless transition – with users having the option to fully customize the transition directly in high-resolution.


```python
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to('cuda')
dh = DiffusersHolder(pipe)
lb = LatentBlending(dh)

lb.set_prompt1('photo of my first prompt1')
lb.set_prompt2('photo of my second prompt')
depth_strength = 0.6 # How deep the first branching happens
t_compute_max_allowed = 10 # How much compute time we give to the transition
imgs_transition = lb.run_transition(
    depth_strength=depth_strength,
    num_inference_steps=num_inference_steps,
    t_compute_max_allowed=t_compute_max_allowed)
```
## Gradio UI
Coming soon again :)

## Example 1: Simple transition
![](example1.jpg)
To run a simple transition between two prompts, run `example1_standard.py`

## Example 2: Multi transition
To run multiple transition between K prompts, resulting in a stitched video, run `example2_multitrans.py`.
[View a longer example video here.](https://vimeo.com/789052336/80dcb545b2)


# Customization

## Most relevant parameters
You can find the [most relevant parameters here.](parameters.md)

### Change the height/width
```python 
size_output = (1024, 768)
lb.set_dimensions(size_output)
```
### Change guidance scale
```python 
lb.set_guidance_scale(5.0)
```

### run_transition parameters
* num_inference_steps: number of diffusions steps.Number of diffusion steps. Higher values will take more compute time.
* depth_strength: The strength of the diffusion iterations determines when the blending process will begin. A value close to zero results in more creative and intricate outcomes, while a value closer to one indicates a simpler alpha blending. However, low values may also bring about the introduction of additional objects and motion.
* t_compute_max_allowed:  maximum time allowed for computation. Higher values give better results but take longer. Either provide t_compute_max_allowed or nmb_max_branches. 
* nmb_max_branches: The maximum number of branches to be computed. Higher values give better results. Use this if you want to have controllable results independent of your hardware. Either provide t_compute_max_allowed or nmb_max_branches. 

### Crossfeeding to the last image.
Cross-feeding latents is a key feature of latent blending. Here, you can set how much the first image branch influences the very last one. In the animation below, these are the blue arrows.

```
crossfeed_power = 0.5 # 50% of the latents in the last branch are copied from branch1
crossfeed_range = 0.7 # The crossfeed is active until 70% of num_iteration, then switched off
crossfeed_decay = 0.2 # The power of the crossfeed decreases over diffusion iterations, here it would be 0.5*0.2=0.1 in the end of the range.
lb.set_branch1_crossfeed(crossfeed_power, crossfeed_range, crossfeed_decay)
```

### Crossfeeding to all transition images
Here, you can set how much the parent branches influence the mixed one. In the animation below, these are the yellow arrows.

```
crossfeed_power = 0.5 # 50% of the latents in the last branch are copied from the parents
crossfeed_range = 0.7 # The crossfeed is active until 70% of num_iteration, then switched off
crossfeed_decay = 0.2 # The power of the crossfeed decreases over diffusion iterations, here it would be 0.5*0.2=0.1 in the end of the range.
lb.set_parental_crossfeed(crossfeed_power, crossfeed_range, crossfeed_decay)
```


# Installation
#### Packages
```commandline
pip install -r requirements.txt
```

# How does latent blending work?
## Method
![](animation.gif)

In the figure above, a diffusion tree is illustrated. The diffusion steps are represented on the y-axis, with temporal blending on the x-axis. The diffusion trajectory for the first prompt is the most left column, which is always computed first. Next, the the trajectory for the second prompt is computed, which may be influenced by the first branch (blue arrows, for a description see above at `Crossfeeding to the last image.`). Finally, all transition images in between are computed. For the transition, there can be an influence of the parents, which helps preserving structures (yellow arrows, for a description see above at `Crossfeeding to all transition images`). Importantly, the place of injection on the x-axis is not hardfixes a priori, but set dynamically using [Perceptual Similarity](https://richzhang.github.io/PerceptualSimilarity), always adding a branch where it is needed most.

The concrete parameters for the transition above would be:
```
lb.set_branch1_crossfeed(crossfeed_power=0.8, crossfeed_range=0.6, crossfeed_decay=0.4)
lb.set_parental_crossfeed(crossfeed_power=0.8, crossfeed_range=0.8, crossfeed_decay=0.2)
imgs_transition = lb.run_transition(num_inference_steps=10, depth_strength=0.2, nmb_max_branches=7)
```

## Perceptual aspects
With latent blending, we can create transitions that appear to defy the laws of nature, yet appear completely natural and believable. The key is to surpress processing in our [dorsal visual stream](https://en.wikipedia.org/wiki/Two-streams_hypothesis#Dorsal_stream), which is achieved by avoiding motion in the transition. Without motion, our visual system has difficulties detecting the transition, leaving viewers with the illusion of a single, continuous image, see [change blindness](https://en.wikipedia.org/wiki/Change_blindness). However, when motion is introduced, the visual system can detect the transition and the viewer becomes aware of the transition, leading to a jarring effect. Therefore, best results will be achieved when optimizing the transition parameters, particularly the crossfeeding parameters and the depth of the first injection.

# Changelog
* SD XL support 
* Diffusers backend, greatly simplifing installation and use (bring your own pipe)
* New blending engine with cross-feeding capabilities, enabling structure preserving transitions
* LPIPS image similarity for finding the next best injection branch, resulting in smoother transitions
* Time-based computation: instead of specifying how many frames your transition has, you can tell your compute budget and get a transition within that budget.
* New multi-movie engine
* Simpler and more powerful gradio ui. You can iterate faster and stitch together a multi movie.
* Inpaint support dropped (as it only makes sense for a single transition)

# Coming soon...
- [ ] Gradio interface
- [ ] Huggingface Space
- [ ] Controlnet
- [ ] IP-Adapter
- [ ] Latent Consistency



Stay tuned on twitter: ```@j_stelzer```

Contact: ```stelzer@lunar-ring.ai``` (Johannes Stelzer)


