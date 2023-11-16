import unittest
from latent_blending import LatentBlending
from diffusers_holder import DiffusersHolder
from diffusers import DiffusionPipeline
import torch

default_pipe = "stabilityai/stable-diffusion-xl-base-1.0"


class TestDiffusersHolder(unittest.TestCase):

    def test_load_diffusers_holder(self):
        pipe = DiffusionPipeline.from_pretrained(default_pipe, torch_dtype=torch.float16).to('cuda')
        dh = DiffusersHolder(pipe)
        self.assertIsNotNone(dh, "Failed to load DiffusersHolder")


class TestSingleImageGeneration(unittest.TestCase):

    def test_single_image_generation(self):
        pipe = DiffusionPipeline.from_pretrained(default_pipe, torch_dtype=torch.float16).to('cuda')
        dh = DiffusersHolder(pipe)
        dh.set_dimensions((1024, 704))
        dh.set_num_inference_steps(40)
        prompt = "Your prompt here"
        text_embeddings = dh.get_text_embedding(prompt)
        generator = torch.Generator(device=dh.device).manual_seed(int(420))
        latents_start = dh.get_noise()
        list_latents_1 = dh.run_diffusion(text_embeddings, latents_start)
        img_orig = dh.latent2image(list_latents_1[-1])
        self.assertIsNotNone(img_orig, "Failed to generate an image")


class TestImageTransition(unittest.TestCase):

    def test_image_transition(self):
        pipe = DiffusionPipeline.from_pretrained(default_pipe, torch_dtype=torch.float16).to('cuda')
        dh = DiffusersHolder(pipe)
        lb = LatentBlending(dh)

        lb.set_prompt1('photo of my first prompt1')
        lb.set_prompt2('photo of my second prompt')
        depth_strength = 0.6
        t_compute_max_allowed = 10
        num_inference_steps = 30
        imgs_transition = lb.run_transition(
            depth_strength=depth_strength,
            num_inference_steps=num_inference_steps,
            t_compute_max_allowed=t_compute_max_allowed)

        self.assertTrue(len(imgs_transition) > 0, "No transition images generated")

if __name__ == '__main__':
    unittest.main()
