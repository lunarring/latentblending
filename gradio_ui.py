# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
import torch
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import warnings
import torch
from tqdm.auto import tqdm
from PIL import Image
import torch
from movie_util import MovieSaver, concatenate_movies
from typing import Callable, List, Optional, Union
from latent_blending import get_time, yml_save, LatentBlending, add_frames_linear_interp, compare_dicts
from stable_diffusion_holder import StableDiffusionHolder
torch.set_grad_enabled(False)
import gradio as gr
import copy
from dotenv import find_dotenv, load_dotenv
import shutil



#%%

class BlendingFrontend():
    def __init__(self, sdh=None):
        self.num_inference_steps = 30
        if sdh is None:
            self.use_debug = True
            self.height = 768
            self.width = 768
        else:
            self.use_debug = False
            self.lb = LatentBlending(sdh)
            self.lb.sdh.num_inference_steps = self.num_inference_steps
            self.height = self.lb.sdh.height
            self.width = self.lb.sdh.width
        
        self.init_save_dir()
        self.save_empty_image()
        self.share = True
        self.transition_can_be_computed = False
        self.depth_strength = 0.25
        self.seed1 = 420
        self.seed2 = 420
        self.guidance_scale = 4.0
        self.guidance_scale_mid_damper = 0.5
        self.mid_compression_scaler = 1.2
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""
        self.state_current = {}
        self.branch1_crossfeed_power = self.lb.branch1_crossfeed_power
        self.branch1_crossfeed_range = self.lb.branch1_crossfeed_range
        self.branch1_crossfeed_decay = self.lb.branch1_crossfeed_decay
        self.parental_crossfeed_power = self.lb.parental_crossfeed_power
        self.parental_crossfeed_range = self.lb.parental_crossfeed_range
        self.parental_crossfeed_power_decay = self.lb.parental_crossfeed_power_decay
        self.fps = 30
        self.duration_video = 10
        self.t_compute_max_allowed = 10
        self.list_fp_imgs_current = []
        self.current_timestamp = None
        self.recycle_img1 = False
        self.recycle_img2 = False
        self.fp_img1 = None
        self.fp_img2 = None
        self.multi_idx_current = -1
        self.list_imgs_shown_last = 5*[self.fp_img_empty]
        self.list_all_segments = []
        self.dp_session = ""
        
        
    def init_save_dir(self):
        load_dotenv(find_dotenv(), verbose=False) 
        self.dp_out = os.getenv("DIR_OUT")
        if self.dp_out is None:
            self.dp_out = ""
        self.dp_imgs = os.path.join(self.dp_out, "imgs")
        os.makedirs(self.dp_imgs, exist_ok=True)
        self.dp_movies = os.path.join(self.dp_out, "movies")
        os.makedirs(self.dp_movies, exist_ok=True)
        
        
        # make dummy image
    def save_empty_image(self):
        self.fp_img_empty = os.path.join(self.dp_imgs, 'empty.jpg')
        Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8)).save(self.fp_img_empty, quality=5)
        
        
    def randomize_seed1(self):
        # Dont randomize seed if we are in a multi concat mode. we don't want to change this one otherwise the movie breaks
        if len(self.list_all_segments) > 0:
            seed = self.seed1
        else:
            seed = np.random.randint(0, 10000000)
        self.seed1 = int(seed)
        print(f"randomize_seed1: new seed = {self.seed1}")
        return seed
        
    def randomize_seed2(self):
        seed = np.random.randint(0, 10000000)
        self.seed2 = int(seed)
        print(f"randomize_seed2: new seed = {self.seed2}")
        return seed
        
    
    def setup_lb(self, list_ui_elem):
        # Collect latent blending variables
        self.state_current = self.get_state_dict()
        self.lb.set_width(list_ui_elem[list_ui_keys.index('width')])
        self.lb.set_height(list_ui_elem[list_ui_keys.index('height')])
        self.lb.set_prompt1(list_ui_elem[list_ui_keys.index('prompt1')])
        self.lb.set_prompt2(list_ui_elem[list_ui_keys.index('prompt2')])
        self.lb.set_negative_prompt(list_ui_elem[list_ui_keys.index('negative_prompt')])
        self.lb.guidance_scale = list_ui_elem[list_ui_keys.index('guidance_scale')]
        self.lb.guidance_scale_mid_damper = list_ui_elem[list_ui_keys.index('guidance_scale_mid_damper')]
        self.t_compute_max_allowed = list_ui_elem[list_ui_keys.index('duration_compute')]
        self.lb.num_inference_steps = list_ui_elem[list_ui_keys.index('num_inference_steps')]
        self.lb.sdh.num_inference_steps = list_ui_elem[list_ui_keys.index('num_inference_steps')]
        self.duration_video = list_ui_elem[list_ui_keys.index('duration_video')]
        self.lb.seed1 = list_ui_elem[list_ui_keys.index('seed1')] #seed
        self.lb.seed2 = list_ui_elem[list_ui_keys.index('seed2')]
        
        self.lb.branch1_crossfeed_power = list_ui_elem[list_ui_keys.index('branch1_crossfeed_power')]
        self.lb.branch1_crossfeed_range = list_ui_elem[list_ui_keys.index('branch1_crossfeed_range')]
        self.lb.branch1_crossfeed_decay = list_ui_elem[list_ui_keys.index('branch1_crossfeed_decay')]
        self.lb.parental_crossfeed_power = list_ui_elem[list_ui_keys.index('parental_crossfeed_power')]
        self.lb.parental_crossfeed_range = list_ui_elem[list_ui_keys.index('parental_crossfeed_range')]
        self.lb.parental_crossfeed_power_decay = list_ui_elem[list_ui_keys.index('parental_crossfeed_power_decay')]
        self.num_inference_steps = list_ui_elem[list_ui_keys.index('num_inference_steps')]
        self.depth_strength = list_ui_elem[list_ui_keys.index('depth_strength')]
        
    
    def compute_img1(self, *args):
        list_ui_elem = args
        self.setup_lb(list_ui_elem)
        self.fp_img1 = os.path.join(self.dp_imgs, f"img1_{get_time('second')}.jpg")
        img1 = Image.fromarray(self.lb.compute_latents1(return_image=True))
        img1.save(self.fp_img1)
        self.recycle_img1 = True
        self.recycle_img2 = False
        return [self.fp_img1, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty]
    
    def compute_img2(self, *args):
        if self.fp_img1 is None: # don't do anything
            return [self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty]
        list_ui_elem = args
        self.setup_lb(list_ui_elem)
        self.fp_img2 = os.path.join(self.dp_imgs, f"img2_{get_time('second')}.jpg")
        img2 = Image.fromarray(self.lb.compute_latents2(return_image=True))
        img2.save(self.fp_img2)
        self.recycle_img2 = True
        self.transition_can_be_computed = True
        return [self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img2]
        
    def compute_transition(self, *args):
        
        if not self.transition_can_be_computed:
            list_return = [self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty]
            return list_return
        
        list_ui_elem = args
        self.setup_lb(list_ui_elem)
        print("STARTING TRANSITION...")
        if self.use_debug:
            list_imgs = [(255*np.random.rand(self.height,self.width,3)).astype(np.uint8) for l in range(5)]
            list_imgs = [Image.fromarray(l) for l in list_imgs]
            print("DONE! SENDING BACK RESULTS")
            return list_imgs
        
        fixed_seeds = [self.seed1, self.seed2]
        
        # Run Latent Blending
        imgs_transition = self.lb.run_transition(
            recycle_img1=self.recycle_img1, 
            recycle_img2=self.recycle_img2, 
            num_inference_steps=self.num_inference_steps, 
            depth_strength=self.depth_strength, 
            t_compute_max_allowed=self.t_compute_max_allowed,
            fixed_seeds=fixed_seeds
            )
        print(f"Latent Blending pass finished. Resulted in {len(imgs_transition)} images")
        
        # Subselect three preview images
        idx_img_prev = np.round(np.linspace(0, len(imgs_transition)-1, 5)[1:-1]).astype(np.int32)
        list_imgs_preview = []
        for j in idx_img_prev:
            list_imgs_preview.append(Image.fromarray(imgs_transition[j]))
            
        # Save the preview imgs as jpgs on disk so we are not sending umcompressed data around
        self.current_timestamp = get_time('second')
        self.list_fp_imgs_current = []
        for i in range(len(list_imgs_preview)):
            fp_img = os.path.join(self.dp_imgs, f"img_preview_{i}_{self.current_timestamp}.jpg")
            list_imgs_preview[i].save(fp_img)
            self.list_fp_imgs_current.append(fp_img)
        
        # Insert cheap frames for the movie
        imgs_transition_ext = add_frames_linear_interp(imgs_transition, self.duration_video, self.fps)

        # Save as movie
        self.fp_movie = os.path.join(self.dp_movies, f"movie_{self.current_timestamp}.mp4") 
        if os.path.isfile(self.fp_movie):
            os.remove(self.fp_movie)
        ms = MovieSaver(self.fp_movie, fps=self.fps)
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()
        print("DONE SAVING MOVIE! SENDING BACK...")
        
        # Assemble Output, updating the preview images and le movie
        list_return = self.list_fp_imgs_current + [self.fp_movie]
        return list_return

    
    def stack_forward(self, prompt2, seed2):
        # Save preview images, prompts and seeds into dictionary for stacking
        if len(self.list_all_segments) == 0:
            timestamp_session = get_time('second')
            self.dp_session = os.path.join(self.dp_out, f"session_{timestamp_session}")
            os.makedirs(self.dp_session)
            
        self.transition_can_be_computed = False

        idx_segment = len(self.list_all_segments) 
        dp_segment = os.path.join(self.dp_session, f"segment_{str(idx_segment).zfill(3)}")
            
        self.list_all_segments.append(dp_segment)
        self.lb.write_imgs_transition(dp_segment)
        shutil.copyfile(self.fp_movie, os.path.join(dp_segment, "movie.mp4"))
        
        self.lb.swap_forward()
        fp_multi = self.multi_concat()
        list_out = [fp_multi]
        list_out.extend([self.fp_img2])
        list_out.extend([self.fp_img_empty]*4)
        list_out.append(gr.update(interactive=False, value=prompt2))
        list_out.append(gr.update(interactive=False, value=seed2))
        list_out.append("")
        list_out.append(np.random.randint(0, 10000000))
        print(f"stack_forward: fp_multi {fp_multi}")
        return list_out

       
    def multi_concat(self):
        list_fp_movies = []
        for dp_segment in self.list_all_segments:
            list_fp_movies.append(os.path.join(dp_segment, "movie.mp4"))
    
        # Concatenate movies and save
        fp_final = os.path.join(self.dp_session, "movie.mp4")
        concatenate_movies(fp_final, list_fp_movies)
        return fp_final

    def get_state_dict(self):
        state_dict = {}
        grab_vars = ['prompt1', 'prompt2', 'seed1', 'seed2', 'height', 'width',
                     'num_inference_steps', 'depth_strength', 'guidance_scale',
                     'guidance_scale_mid_damper', 'mid_compression_scaler']
        
        for v in grab_vars:
            state_dict[v] = getattr(self, v)
        return state_dict   


        
if __name__ == "__main__":    
    
    # fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_768-ema-pruned.ckpt" 
    fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 
    bf = BlendingFrontend(StableDiffusionHolder(fp_ckpt)) 
    # self = BlendingFrontend(None) 
    
    with gr.Blocks() as demo:
        with gr.Row():
            prompt1 = gr.Textbox(label="prompt 1")
            prompt2 = gr.Textbox(label="prompt 2")
        
        with gr.Row():
            duration_compute = gr.Slider(5, 200, bf.t_compute_max_allowed, step=1, label='compute budget', interactive=True) 
            duration_video = gr.Slider(1, 100, bf.duration_video, step=0.1, label='video duration', interactive=True) 
            height = gr.Slider(256, 2048, bf.height, step=128, label='height', interactive=True)
            width = gr.Slider(256, 2048, bf.width, step=128, label='width', interactive=True) 
            
        with gr.Accordion("Advanced Settings (click to expand)", open=False):

            with gr.Accordion("Diffusion settings", open=True):
                with gr.Row():
                    num_inference_steps = gr.Slider(5, 100, bf.num_inference_steps, step=1, label='num_inference_steps', interactive=True)
                    guidance_scale = gr.Slider(1, 25, bf.guidance_scale, step=0.1, label='guidance_scale', interactive=True) 
                    negative_prompt = gr.Textbox(label="negative prompt")          
            
            with gr.Accordion("Seed control: adjust seeds for first and last images", open=True):
                with gr.Row():
                    b_newseed1 = gr.Button("randomize seed 1", variant='secondary')
                    seed1 = gr.Number(bf.seed1, label="seed 1", interactive=True)
                    seed2 = gr.Number(bf.seed2, label="seed 2", interactive=True)
                    b_newseed2 = gr.Button("randomize seed 2", variant='secondary')
                    
            with gr.Accordion("Last image crossfeeding.", open=True):
                with gr.Row():
                    branch1_crossfeed_power = gr.Slider(0.0, 1.0, bf.branch1_crossfeed_power, step=0.01, label='branch1 crossfeed power', interactive=True) 
                    branch1_crossfeed_range = gr.Slider(0.0, 1.0, bf.branch1_crossfeed_range, step=0.01, label='branch1 crossfeed range', interactive=True) 
                    branch1_crossfeed_decay = gr.Slider(0.0, 1.0, bf.branch1_crossfeed_decay, step=0.01, label='branch1 crossfeed decay', interactive=True) 

            with gr.Accordion("Transition settings", open=True):
                with gr.Row():
                    parental_crossfeed_power = gr.Slider(0.0, 1.0, bf.parental_crossfeed_power, step=0.01, label='parental crossfeed power', interactive=True) 
                    parental_crossfeed_range = gr.Slider(0.0, 1.0, bf.parental_crossfeed_range, step=0.01, label='parental crossfeed range', interactive=True) 
                    parental_crossfeed_power_decay = gr.Slider(0.0, 1.0, bf.parental_crossfeed_power_decay, step=0.01, label='parental crossfeed decay', interactive=True) 
                with gr.Row():
                    depth_strength = gr.Slider(0.01, 0.99, bf.depth_strength, step=0.01, label='depth_strength', interactive=True) 
                    guidance_scale_mid_damper = gr.Slider(0.01, 2.0, bf.guidance_scale_mid_damper, step=0.01, label='guidance_scale_mid_damper', interactive=True) 
        
                
        with gr.Row():
            b_compute1 = gr.Button('compute first image', variant='primary')
            b_compute_transition = gr.Button('compute transition', variant='primary')
            b_compute2 = gr.Button('compute last image', variant='primary')
        
        with gr.Row():
            img1 = gr.Image(label="1/5")
            img2 = gr.Image(label="2/5", show_progress=False)
            img3 = gr.Image(label="3/5", show_progress=False)
            img4 = gr.Image(label="4/5", show_progress=False)
            img5 = gr.Image(label="5/5")
        
        with gr.Row():
            vid_single = gr.Video(label="single trans")
            vid_multi = gr.Video(label="multi trans")
            
        with gr.Row():
            # b_restart = gr.Button("RESTART EVERYTHING")
            b_stackforward = gr.Button('append last movie segment (left) to multi movie (right)', variant='primary')
            
        with gr.Row():
            gr.Markdown(
                """
                # Parameters
                ## Main
                - compute budget: set your waiting time for the transition. high values = better quality
                - video duration: seconds per segment
                - height/width: in pixels
                
                ## Diffusion settings
                - num_inference_steps: number of diffusion steps
                - guidance_scale: latent blending seems to prefer lower values here
                - negative prompt: enter negative prompt here, applied for all images
                
                ## Last iamge crossfeeding
                - branch1_crossfeed_power: Controls the level of cross-feeding between the first and last image branch. For preserving structures.
                - branch1_crossfeed_range: Sets the duration of active crossfeed during development. High values enforce strong structural similarity.
                - branch1_crossfeed_decay: Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
                
                ## Transition settings
                - parental_crossfeed_power: Similar to branch1_crossfeed_power, however applied for the images withinin the transition.
                - parental_crossfeed_range: Similar to branch1_crossfeed_range, however applied for the images withinin the transition.
                - parental_crossfeed_power_decay: Similar to branch1_crossfeed_decay, however applied for the images withinin the transition.
                - depth_strength: Determines when the blending process will begin in terms of diffusion steps. Low values more inventive but can cause motion.
                - guidance_scale_mid_damper: Decreases the guidance scale in the middle of a transition.
                """
                )
                        
        
        # Collect all UI elemts in list to easily pass as inputs in gradio
        dict_ui_elem = {}
        dict_ui_elem["prompt1"] = prompt1
        dict_ui_elem["negative_prompt"] = negative_prompt
        dict_ui_elem["prompt2"] = prompt2
         
        dict_ui_elem["duration_compute"] = duration_compute
        dict_ui_elem["duration_video"] = duration_video
        dict_ui_elem["height"] = height
        dict_ui_elem["width"] = width
         
        dict_ui_elem["depth_strength"] = depth_strength
        dict_ui_elem["branch1_crossfeed_power"] = branch1_crossfeed_power
        dict_ui_elem["branch1_crossfeed_range"] = branch1_crossfeed_range
        dict_ui_elem["branch1_crossfeed_decay"] = branch1_crossfeed_decay
        
        dict_ui_elem["num_inference_steps"] = num_inference_steps
        dict_ui_elem["guidance_scale"] = guidance_scale
        dict_ui_elem["guidance_scale_mid_damper"] = guidance_scale_mid_damper
        dict_ui_elem["seed1"] = seed1
        dict_ui_elem["seed2"] = seed2
        
        dict_ui_elem["parental_crossfeed_range"] = parental_crossfeed_range
        dict_ui_elem["parental_crossfeed_power"] = parental_crossfeed_power
        dict_ui_elem["parental_crossfeed_power_decay"] = parental_crossfeed_power_decay
        
        # Convert to list, as gradio doesn't seem to accept dicts
        list_ui_elem = []
        list_ui_keys = []
        for k in dict_ui_elem.keys():
            list_ui_elem.append(dict_ui_elem[k])
            list_ui_keys.append(k)
        bf.list_ui_keys = list_ui_keys
        
        b_newseed1.click(bf.randomize_seed1, outputs=seed1)
        b_newseed2.click(bf.randomize_seed2, outputs=seed2)
        b_compute1.click(bf.compute_img1, inputs=list_ui_elem, outputs=[img1, img2, img3, img4, img5])
        b_compute2.click(bf.compute_img2, inputs=list_ui_elem, outputs=[img2, img3, img4, img5])
        b_compute_transition.click(bf.compute_transition, 
                                    inputs=list_ui_elem,
                                    outputs=[img2, img3, img4, vid_single])
        
        b_stackforward.click(bf.stack_forward, 
                      inputs=[prompt2, seed2], 
                      outputs=[vid_multi, img1, img2, img3, img4, img5, prompt1, seed1, prompt2])

               
    demo.launch(share=bf.share, inbrowser=True, inline=False)
