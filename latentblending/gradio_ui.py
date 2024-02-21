import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm
from PIL import Image
import gradio as gr
import shutil
import uuid
from diffusers import AutoPipelineForText2Image
from latentblending.blending_engine import BlendingEngine
import datetime

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
import json



class BlendingFrontend():
    def __init__(
            self,
            be,
            share=False):
        r"""
        Gradio Helper Class to collect UI data and start latent blending.
        Args:
            be:
                Blendingengine
            share: bool
                Set true to get a shareable gradio link (e.g. for running a remote server)
        """
        self.be = be
        self.share = share

        # UI Defaults
        self.seed1 = 420
        self.seed2 = 420
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""

        # Vars
        self.prompt = None
        self.negative_prompt = None
        self.list_seeds = []
        self.idx_movie = 0
        self.data = []

    def take_image0(self):
        return self.take_image(0)
    
    def take_image1(self):
        return self.take_image(1)
    
    def take_image2(self):
        return self.take_image(2)
    
    def take_image3(self):
        return self.take_image(3)
    

    def take_image(self, id_img):
        if self.prompt is None:
            print("Cannot take because no prompt was set!")
            return [None, None, None, None, ""]
        if self.idx_movie == 0:
            current_time = datetime.datetime.now()
            self.fp_out = "movie_" + current_time.strftime("%y%m%d_%H%M") + ".json"
            self.data.append({"settings": "sdxl", "width": bf.be.dh.width_img, "height": self.be.dh.height_img, "num_inference_steps": self.be.dh.num_inference_steps})

        seed = self.list_seeds[id_img]
        
        self.data.append({"iteration": self.idx_movie, "seed": seed, "prompt": self.prompt, "negative_prompt": self.negative_prompt})

        # Write the data list to a JSON file
        with open(self.fp_out, 'w') as f:
            json.dump(self.data, f, indent=4)

        self.idx_movie += 1
        self.prompt = None
        return [None, None, None, None, ""]


    def compute_imgs(self, prompt, negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.be.set_prompt1(prompt)
        self.be.set_prompt2(prompt)
        self.be.set_negative_prompt(negative_prompt)
        self.list_seeds = []
        self.list_images = []
        for i in range(4):
            seed = np.random.randint(0, 1000000000)
            self.be.seed1 = seed
            self.list_seeds.append(seed)
            img = self.be.compute_latents1(return_image=True)
            self.list_images.append(img)
        return self.list_images
        
        


if __name__ == "__main__":

    width = 786
    height = 1024
    num_inference_steps = 4
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    # pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    be = BlendingEngine(pipe)
    be.set_dimensions((width, height))
    be.set_num_inference_steps(num_inference_steps)

    bf = BlendingFrontend(be)

    with gr.Blocks() as demo:

        with gr.Row():
            prompt = gr.Textbox(label="prompt")
            negative_prompt = gr.Textbox(label="negative prompt")

        with gr.Row():
            b_compute = gr.Button('compute new images', variant='primary')

        with gr.Row():
            with gr.Column():
                img0 = gr.Image(label="seed1")
                b_take0 = gr.Button('take', variant='primary')
            with gr.Column():
                img1 = gr.Image(label="seed2")
                b_take1 = gr.Button('take', variant='primary')
            with gr.Column():
                img2 = gr.Image(label="seed3")
                b_take2 = gr.Button('take', variant='primary')
            with gr.Column():
                img3 = gr.Image(label="seed4")
                b_take3 = gr.Button('take', variant='primary')

        b_compute.click(bf.compute_imgs, inputs=[prompt, negative_prompt], outputs=[img0, img1, img2, img3])
        b_take0.click(bf.take_image0, outputs=[img0, img1, img2, img3, prompt])
        b_take1.click(bf.take_image1, outputs=[img0, img1, img2, img3, prompt])
        b_take2.click(bf.take_image2, outputs=[img0, img1, img2, img3, prompt])
        b_take3.click(bf.take_image3, outputs=[img0, img1, img2, img3, prompt])

    demo.launch(share=bf.share, inbrowser=True, inline=False)
