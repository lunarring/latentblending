# Gradio parameters

## depth_strength
determines when the blending process will begin in terms of diffusion steps. A value close to zero results in more creative and intricate outcomes, while a value closer to one indicates a simpler alpha blending. However, low values may also bring about the introduction of additional objects and motion.

## guidance_scale
higher guidance scale encourages the creation of images that are closely aligned with the text. However, the best results for latent blending are achieved with lower values.

## guidance_scale_mid_damper
decreases the guidance scale in the middle of a transition. A value of 1 would maintain a constant guidance scale, while a value of 0 would decrease the guidance scale to 1 at the midpoint of the transition

## mid_compression_scaler
stretches the spacing towards the center, with a linear spacing at mid_compression_scaler=1 and a higher sampling density in the middle at mid_compression_scaler=2

## num_inference_steps
determines the quality of the results. While an increase in this value may improve the outcome, it will also require more computation time.

## nmb_trans_images
final number of images computed in the last branch of the tree. Higher values will give better results but require more computation time.


