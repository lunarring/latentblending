# Gradio Parameters

## depth_strength
- Determines when the blending process will begin in terms of diffusion steps.
- A value close to zero results in more creative and intricate outcomes, but may also introduce additional objects and motion.
- A value closer to one indicates a simpler alpha blending.

## branch1_influence
- Determines the extent to which the initial branch affects the final branch. This generally improves the transitions!
- A value of 0.0 disables this crossfeeding.
- A value of 1.0 fully copies the latents from the first branch to the second.
- Before the tree branches out into multiple parts, crossfeeding occurs at a specific point, known as list_injection_idx[1]. The precise location is determined by a value called branch1_fract_crossfeed, which ranges from 0.0 to 1.0.

## guidance_scale
- Higher guidance scale encourages the creation of images that are closely aligned with the text.
- Lower values are recommended for the best results in latent blending.

## guidance_scale_mid_damper
- Decreases the guidance scale in the middle of a transition.
- A value of 1 maintains a constant guidance scale.
- A value of 0 decreases the guidance scale to 1 at the midpoint of the transition.

## num_inference_steps
- Determines the quality of the results.
- Higher values improve the outcome, but also require more computation time.

## nmb_trans_images
- Final number of images computed in the last branch of the tree.
- Higher values give better results but require more computation time.
