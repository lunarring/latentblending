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
import tempfile
import json
from lunar_tools import concatenate_movies


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
        self.nmb_preview_images = 4

        # Vars
        self.prompt = None
        self.negative_prompt = None
        self.list_seeds = []
        self.idx_movie = 0
        self.list_seeds = []
        self.list_images_preview = []
        self.data = []
        self.idx_img_selected = None
        self.jpg_quality = 80 
        self.fp_movie = ''
        self.duration_single_trans = 10
        

    def preview_img_selected(self, data: gr.SelectData, button):
        self.idx_img_selected = data.index
        print(f"gallery image {self.idx_img_selected} selected, seed {self.list_seeds[self.idx_img_selected]}")
        return gr.Button(interactive=True) 


    def compute_imgs(self, prompt, negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.be.set_prompt1(prompt)
        self.be.set_prompt2(prompt)
        self.be.set_negative_prompt(negative_prompt)
        self.list_seeds = []
        self.list_images_preview = []
        self.idx_img_selected = None
        for i in range(self.nmb_preview_images):
            seed = np.random.randint(0, np.iinfo(np.int32).max)
            self.be.seed1 = seed
            self.list_seeds.append(seed)
            img = self.be.compute_latents1(return_image=True)
            fn_img_tmp = f"image_{uuid.uuid4()}.jpg"
            temp_img_path = os.path.join(tempfile.gettempdir(), fn_img_tmp)
            img.save(temp_img_path)
            img.save(temp_img_path, quality=self.jpg_quality, optimize=True)
            self.list_images_preview.append(temp_img_path)
        return self.list_images_preview
    

    def get_list_images_movie(self):
        return [entry["preview_image"] for entry in self.data[1:]]


    def init_new_movie(self):
        current_time = datetime.datetime.now()
        self.fp_movie = "movie_" + current_time.strftime("%y%m%d_%H%M") + ".mp4"
        self.fp_json = "movie_" + current_time.strftime("%y%m%d_%H%M") + ".json"
        self.data.append({"settings": "sdxl", "width": bf.be.dh.width_img, "height": self.be.dh.height_img, "num_inference_steps": self.be.dh.num_inference_steps})


    def add_image_to_video(self):
        if self.prompt is None:
            print("Cannot take because no prompt was set!")
            return self.get_list_images_movie()
        if self.idx_movie == 0:
            self.init_new_movie()

        self.data.append({"iteration": self.idx_movie, 
        "seed": self.list_seeds[self.idx_img_selected], 
        "prompt": self.prompt, 
        "negative_prompt": self.negative_prompt,
        "preview_image": self.list_images_preview[self.idx_img_selected]
        })

        # Write the data list to a JSON file
        with open(self.fp_json, 'w') as f:
            json.dump(self.data, f, indent=4)

        self.idx_movie += 1
        self.prompt = None

        return self.get_list_images_movie()

    def generate_movie(self):
        print("starting movie gen")
        list_prompts = []
        list_negative_prompts = []
        list_seeds = []

        # Extract prompts, negative prompts, and seeds from the data
        for item in self.data[1:]:  # Skip the first item as it contains settings
            list_prompts.append(item["prompt"])
            list_negative_prompts.append(item["negative_prompt"])
            list_seeds.append(item["seed"])

        list_movie_parts = []
        for i in range(len(list_prompts) - 1):
            # For a multi transition we can save some computation time and recycle the latents
            if i == 0:
                self.be.set_prompt1(list_prompts[i])
                self.be.set_negative_prompt(list_negative_prompts[i])
                self.be.set_prompt2(list_prompts[i + 1])
                recycle_img1 = False
            else:
                self.be.swap_forward()
                self.be.set_negative_prompt(list_negative_prompts[i+1])
                self.be.set_prompt2(list_prompts[i + 1])
                recycle_img1 = True

            fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
            fixed_seeds = list_seeds[i:i + 2]
            # Run latent blending
            self.be.run_transition(
                recycle_img1=recycle_img1,
                fixed_seeds=fixed_seeds)

            # Save movie
            self.be.write_movie_transition(fp_movie_part, self.duration_single_trans)
            list_movie_parts.append(fp_movie_part)

        # Finally, concatenate the result
        concatenate_movies(self.fp_movie, list_movie_parts)
        print(f"DONE! MOVIE SAVED IN {self.fp_movie}")
        return self.fp_movie



if __name__ == "__main__":
    width = 512
    height = 512
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
            b_compute = gr.Button('generate preview images', variant='primary')

        # with gr.Row():


        with gr.Row():
            gallery_preview = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    , columns=[bf.nmb_preview_images], rows=[1], object_fit="contain", height="auto", allow_preview=False, interactive=False)


        with gr.Row():
            b_select = gr.Button('add selected image to video', variant='primary', interactive=False)        


        with gr.Row():
            gr.Markdown("Your movie contains so far the below frames")
        with gr.Row():
            gallery_movie = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    , columns=[20], rows=[1], object_fit="contain", height="auto", allow_preview=False, interactive=False)        
            
        with gr.Row():
            b_generate_movie = gr.Button('generate movie', variant='primary')

        with gr.Row():
            movie = gr.Video()
           

        # bindings
        b_compute.click(bf.compute_imgs, inputs=[prompt, negative_prompt], outputs=gallery_preview)
        b_select.click(bf.add_image_to_video, None, gallery_movie)
        b_generate_movie.click(bf.generate_movie, None, movie)
        gallery_preview.select(bf.preview_img_selected, None, b_select)



    demo.launch(share=bf.share, inbrowser=True, inline=False)
