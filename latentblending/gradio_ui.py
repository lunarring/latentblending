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
import argparse

"""
TODO
- time per segment
- init phase (model, res, nmb iter)
- recycle existing movies
- hf spaces integration
"""

class MultiUserRouter():
    def __init__(
            self,
            do_compile=False
        ):
        self.user_blendingvariableholder = {}
        self.do_compile = do_compile
        self.list_models = ["stabilityai/sdxl-turbo", "stabilityai/stable-diffusion-xl-base-1.0"]

        self.init_models()

    def init_models(self):
        self.dict_blendingengines = {}
        for m in self.list_models:
            pipe = AutoPipelineForText2Image.from_pretrained(m, torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")
            be = BlendingEngine(pipe, do_compile=self.do_compile)
            
            self.dict_blendingengines[m] = be

    def register_new_user(self, model, width, height):
        user_id = str(uuid.uuid4().hex.upper()[0:8])
        be = self.dict_blendingengines[model]
        be.set_dimensions((width, height))
        self.user_blendingvariableholder[user_id] = BlendingVariableHolder(be)
        return user_id

    def user_overflow_protection(self):
        pass

    def preview_img_selected(self, user_id, data: gr.SelectData, button):
        return self.user_blendingvariableholder[user_id].preview_img_selected(data, button)

    def movie_img_selected(self, user_id, data: gr.SelectData, button):
        return self.user_blendingvariableholder[user_id].movie_img_selected(data, button)

    def compute_imgs(self, user_id, prompt, negative_prompt):
        return self.user_blendingvariableholder[user_id].compute_imgs(prompt, negative_prompt)
    
    def get_list_images_movie(self, user_id):
        return self.user_blendingvariableholder[user_id].get_list_images_movie()
    
    def init_new_movie(self, user_id):
        return self.user_blendingvariableholder[user_id].init_new_movie()
    
    def write_json(self, user_id):
        return self.user_blendingvariableholder[user_id].write_json()
    
    def add_image_to_video(self, user_id):
        return self.user_blendingvariableholder[user_id].add_image_to_video()
    
    def img_movie_delete(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_delete()
    
    def img_movie_later(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_later()
    
    def img_movie_earlier(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_earlier()
    
    def generate_movie(self, user_id, t_per_segment):
        return self.user_blendingvariableholder[user_id].generate_movie(t_per_segment)

#%% BlendingVariableHolder Class
class BlendingVariableHolder():
    def __init__(
            self,
            be):
        r"""
        Gradio Helper Class to collect UI data and start latent blending.
        Args:
            be:
                Blendingengine
            share: bool
                Set true to get a shareable gradio link (e.g. for running a remote server)
        """
        self.be = be

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
        self.idx_img_preview_selected = None
        self.idx_img_movie_selected = None
        self.jpg_quality = 80 
        self.fp_movie = ''

    def preview_img_selected(self, data: gr.SelectData, button):
        self.idx_img_preview_selected = data.index
        print(f"preview image {self.idx_img_preview_selected} selected, seed {self.list_seeds[self.idx_img_preview_selected]}")

    def movie_img_selected(self, data: gr.SelectData, button):
        self.idx_img_movie_selected = data.index
        print(f"movie image {self.idx_img_movie_selected} selected")

    def compute_imgs(self, prompt, negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.be.set_prompt1(prompt)
        self.be.set_prompt2(prompt)
        self.be.set_negative_prompt(negative_prompt)
        self.list_seeds = []
        self.list_images_preview = []
        self.idx_img_preview_selected = None
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
        return [entry["preview_image"] for entry in self.data]


    def init_new_movie(self):
        current_time = datetime.datetime.now()
        self.fp_movie = "movie_" + current_time.strftime("%y%m%d_%H%M") + ".mp4"
        self.fp_json = "movie_" + current_time.strftime("%y%m%d_%H%M") + ".json"
        

    def write_json(self):
        # Write the data list to a JSON file
        data_copy = self.data.copy()
        data_copy.insert(0, {"settings": "sdxl", "width": self.be.dh.width_img, "height": self.be.dh.height_img, "num_inference_steps": self.be.dh.num_inference_steps})
        with open(self.fp_json, 'w') as f:
            json.dump(data_copy, f, indent=4)

    def add_image_to_video(self):
        if self.prompt is None:
            print("Cannot take because no prompt was set!")
            return self.get_list_images_movie()
        if self.idx_movie == 0:
            self.init_new_movie()

        self.data.append({"iteration": self.idx_movie, 
        "seed": self.list_seeds[self.idx_img_preview_selected], 
        "prompt": self.prompt, 
        "negative_prompt": self.negative_prompt,
        "preview_image": self.list_images_preview[self.idx_img_preview_selected]
        })

        self.write_json()
        self.idx_movie += 1
        return self.get_list_images_movie()

    def img_movie_delete(self):
        if self.idx_img_movie_selected is not None and 0 <= self.idx_img_movie_selected < len(self.data)+1:
            del self.data[self.idx_img_movie_selected]
            self.idx_img_movie_selected = None
        else:
            print(f"Invalid movie image index for deletion: {self.idx_img_movie_selected}")
        return self.get_list_images_movie()

    def img_movie_later(self):
        if self.idx_img_movie_selected is not None and self.idx_img_movie_selected < len(self.data):
            # Swap the selected image with the next one
            self.data[self.idx_img_movie_selected], self.data[self.idx_img_movie_selected + 1] = \
                self.data[self.idx_img_movie_selected+1], self.data[self.idx_img_movie_selected]
            self.idx_img_movie_selected = None
        else:
            print("Cannot move the image later in the sequence.")
        return self.get_list_images_movie()

    def img_movie_earlier(self):
        if self.idx_img_movie_selected is not None and self.idx_img_movie_selected > 0:
            # Swap the selected image with the previous one
            self.data[self.idx_img_movie_selected-1], self.data[self.idx_img_movie_selected] = \
                self.data[self.idx_img_movie_selected], self.data[self.idx_img_movie_selected-1]
            self.idx_img_movie_selected = None
        else:
            print("Cannot move the image earlier in the sequence.")
        return self.get_list_images_movie()
    

    def generate_movie(self, t_per_segment=10):
        print("starting movie gen")
        list_prompts = []
        list_negative_prompts = []
        list_seeds = []

        # Extract prompts, negative prompts, and seeds from the data
        for item in self.data: 
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
            self.be.write_movie_transition(fp_movie_part, t_per_segment)
            list_movie_parts.append(fp_movie_part)

        # Finally, concatenate the result
        concatenate_movies(self.fp_movie, list_movie_parts)
        print(f"DONE! MOVIE SAVED IN {self.fp_movie}")
        return self.fp_movie

#%% Runtime engine

if __name__ == "__main__":

    # Change Parameters below
    parser = argparse.ArgumentParser(description="Latent Blending GUI")
    parser.add_argument("--do_compile", type=bool, default=False)
    parser.add_argument("--nmb_preview_images", type=int, default=4)
    parser.add_argument("--server_name", type=str, default=None)
    try:
        args = parser.parse_args()
        nmb_preview_images = args.nmb_preview_images
        do_compile = args.do_compile
        server_name = args.server_name

    except SystemExit:
        # If the script is run in an interactive environment (like Jupyter), parse_args might fail.
        nmb_preview_images = 4
        do_compile = False # compile SD pipes with sdfast
        server_name = None

    mur = MultiUserRouter(do_compile=do_compile)
    with gr.Blocks() as demo:
        with gr.Accordion("Setup", open=True) as accordion_setup:
            # New user registration, model selection, ...
            with gr.Row():
                model = gr.Dropdown(mur.list_models, value=mur.list_models[0], label="model")
                width = gr.Slider(256, 2048, 512, step=128, label='width', interactive=True)
                height = gr.Slider(256, 2048, 512, step=128, label='height', interactive=True)
                user_id = gr.Textbox(label="user id (filled automatically)", interactive=False)
                b_start_session = gr.Button('start session', variant='primary')

        with gr.Accordion("Latent Blending (expand with arrow on right side after you clicked 'start session')", open=False) as accordion_latentblending:
            with gr.Row():
                prompt = gr.Textbox(label="prompt")
                negative_prompt = gr.Textbox(label="negative prompt")
                b_compute = gr.Button('generate preview images', variant='primary')
                b_select = gr.Button('add selected image to video', variant='primary')        

            with gr.Row():
                gallery_preview = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        , columns=[nmb_preview_images], rows=[1], object_fit="contain", height="auto", allow_preview=False, interactive=False)


            with gr.Row():
                gr.Markdown("Your movie contains the following images (see below)")
            with gr.Row():
                gallery_movie = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        , columns=[20], rows=[1], object_fit="contain", height="auto", allow_preview=False, interactive=False)        
                
            
            with gr.Row():
                b_delete = gr.Button('delete selected image')
                b_move_earlier = gr.Button('move image to earlier time')
                b_move_later = gr.Button('move image to later time')

            with gr.Row():
                b_generate_movie = gr.Button('generate movie', variant='primary')
                t_per_segment = gr.Slider(1, 30, 10, step=0.1, label='time per segment', interactive=True)

            with gr.Row():
                movie = gr.Video()

            # bindings
            b_start_session.click(mur.register_new_user, inputs=[model, width, height], outputs=user_id)
            b_compute.click(mur.compute_imgs, inputs=[user_id, prompt, negative_prompt], outputs=gallery_preview)
            b_select.click(mur.add_image_to_video, user_id, gallery_movie)
            gallery_preview.select(mur.preview_img_selected, user_id, None)
            gallery_movie.select(mur.movie_img_selected, user_id, None)
            b_delete.click(mur.img_movie_delete, user_id, gallery_movie)
            b_move_earlier.click(mur.img_movie_earlier, user_id, gallery_movie)
            b_move_later.click(mur.img_movie_later, user_id, gallery_movie)
            b_generate_movie.click(mur.generate_movie, [user_id, t_per_segment], movie)


    if server_name is None:
        demo.launch(share=False, inbrowser=True, inline=False)
    else:
        demo.launch(share=False, inbrowser=True, inline=False, server_name=server_name)
