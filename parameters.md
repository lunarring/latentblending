# Gradio Parameters

## depth_strength
- Determines when the blending process will begin in terms of diffusion steps.
- A value close to zero results in more creative and intricate outcomes, but may also introduce additional objects and motion.
- A value closer to one indicates a simpler alpha blending.

## branch1_crossfeed_power
- Controls the level of cross-feeding between the first and last image branch. This allows to preserve structures from the first image.
- A value of 0.0 disables crossfeeding.
- A value of 1.0 fully copies the latents from the first branch to the last.

## branch1_crossfeed_range
- Sets the duration of active crossfeed during development. High values enforce strong structural similarity.
- The value x ranges from [0,1], and the crossfeeding is deactivated after x*num_inference_steps steps

## branch1_crossfeed_decay
- Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
- The value x ranges from [0,1], and the branch1_crossfeed_power is decreased until the end of the branch1_crossfeed_range to a value of x*branch1_crossfeed_power

## parental_crossfeed_power
Similar to branch1_crossfeed_power, however applied for the images withinin the transition.

## parental_crossfeed_range
Similar to branch1_crossfeed_range, however applied for the images withinin the transition.

## parental_crossfeed_power_decay
Similar to branch1_crossfeed_decay, however applied for the images withinin the transition.

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
