# Gradio Parameters

## depth_strength
- Determines when the blending process will begin in terms of diffusion steps.
- A value close to zero results in more creative and intricate outcomes, but may also introduce additional objects and motion.
- A value closer to one indicates a simpler alpha blending.

## guidance_scale
- Higher guidance scale encourages the creation of images that are closely aligned with the text.
- Lower values are recommended for the best results in latent blending.

## guidance_scale_mid_damper
- Decreases the guidance scale in the middle of a transition.
- A value of 1 maintains a constant guidance scale.
- A value of 0 decreases the guidance scale to 1 at the midpoint of the transition.

## mid_compression_scaler
- Stretches the spacing towards the center.
- A value of 1 results in linear spacing.
- A value of 2 increases sampling density in the middle.

## num_inference_steps
- Determines the quality of the results.
- Higher values improve the outcome, but also require more computation time.

## nmb_trans_images
- Final number of images computed in the last branch of the tree.
- Higher values give better results but require more computation time.
