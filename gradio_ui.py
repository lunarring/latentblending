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



#%%

class BlendingFrontend():
    def __init__(self, sdh=None):
        if sdh is None:
            self.use_debug = True
        else:
            self.use_debug = False
            self.lb = LatentBlending(sdh)
            
        self.share = True
        self.num_inference_steps = 20
        self.depth_strength = 0.25
        self.seed1 = 42
        self.seed2 = 420
        self.guidance_scale = 4.0
        self.guidance_scale_mid_damper = 0.5
        self.mid_compression_scaler = 1.2
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""
        self.list_settings = []
        self.state_current = {}
        self.showing_current = True
        self.branch1_influence = 0.02
        self.nmb_branches_final = 9
        self.nmb_imgs_show = 5 # don't change
        self.fps = 30
        self.duration = 10
        self.dict_multi_trans = {}
        self.dict_multi_trans_include = {}
        self.multi_trans_currently_shown = []
        self.list_fp_imgs_current = []
        self.current_timestamp = None
        self.nmb_trans_stack = 8
        
        if not self.use_debug:
            self.lb.sdh.num_inference_steps = self.num_inference_steps
            self.height = self.lb.sdh.height
            self.width = self.lb.sdh.width
        else:
            self.height = 768
            self.width = 768
        
        # make dummy image
        self.fp_img_empty = 'empty.jpg'
        Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8)).save(self.fp_img_empty, quality=5)
        
    def change_depth_strength(self, value):
        self.depth_strength = value
        print(f"changed depth_strength to {value}")
    
    def change_num_inference_steps(self, value):
        self.num_inference_steps = value
        print(f"changed num_inference_steps to {value}")
        
    def change_guidance_scale(self, value):
        self.guidance_scale = value
        self.lb.set_guidance_scale(value)
        print(f"changed guidance_scale to {value}")
        
    def change_guidance_scale_mid_damper(self, value):
        self.guidance_scale_mid_damper = value
        print(f"changed guidance_scale_mid_damper to {value}")
        
    def change_mid_compression_scaler(self, value):
        self.mid_compression_scaler = value
        print(f"changed mid_compression_scaler to {value}")
        
    def change_branch1_influence(self, value):
        self.branch1_influence = value
        print(f"changed branch1_influence to {value}")
    
    def change_height(self, value):
        self.height = value
        print(f"changed height to {value}")
        
    def change_width(self, value):
        self.width = value
        print(f"changed width to {value}")
        
    def change_nmb_branches_final(self, value):
        self.nmb_branches_final  = value
        print(f"changed nmb_branches_final to {value}")
        
    def change_duration(self, value):
        self.duration  = value
        print(f"changed duration to {value}")
        
    def change_fps(self, value):
        self.fps  = value
        print(f"changed fps to {value}")
        
    def change_negative_prompt(self, value):
        self.negative_prompt = value
        
    def change_seed1(self, value):
        self.seed1 = int(value)
        
    def change_seed2(self, value):
        self.seed2 = int(value)
        
    def randomize_seed1(self):
        seed = np.random.randint(0, 10000000)
        self.change_seed1(seed)
        print(f"randomize_seed1: new seed = {self.seed1}")
        return seed
        
    def randomize_seed2(self):
        seed = np.random.randint(0, 10000000)
        self.change_seed2(seed)
        print(f"randomize_seed2: new seed = {self.seed2}")
        return seed
        
    
    def compute_transition(self, prompt1, prompt2):
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        print("STARTING DIFFUSION!")
        self.state_current = self.get_state_dict()
        if self.use_debug:
            list_imgs = [(255*np.random.rand(self.height,self.width,3)).astype(np.uint8) for l in range(5)]
            list_imgs = [Image.fromarray(l) for l in list_imgs]
            print("DONE! SENDING BACK RESULTS")
            return list_imgs
        
        # Collect latent blending variables
        self.lb.set_width(self.width)
        self.lb.set_height(self.height)
        self.lb.autosetup_branching(
                depth_strength = self.depth_strength,
                num_inference_steps = self.num_inference_steps,
                nmb_branches_final = self.nmb_branches_final,
                nmb_mindist = 3)
        self.lb.set_prompt1(self.prompt1)
        self.lb.set_prompt2(self.prompt2)
        self.lb.set_negative_prompt(self.negative_prompt)
        
        self.lb.guidance_scale = self.guidance_scale
        self.lb.guidance_scale_mid_damper = self.guidance_scale_mid_damper
        self.lb.mid_compression_scaler = self.mid_compression_scaler
        self.lb.branch1_influence = self.branch1_influence
        fixed_seeds = [self.seed1, self.seed2]
        
        # Run Latent Blending
        imgs_transition = self.lb.run_transition(fixed_seeds=fixed_seeds)
        print(f"Latent Blending pass finished. Resulted in {len(imgs_transition)} images")
        
        # Subselect the preview images (hard fixed to self.nmb_imgs_show=5)
        assert np.mod((self.nmb_branches_final-self.nmb_imgs_show)/4, 1)==0, 'self.nmb_branches_final illegal value!'
        idx_list = np.linspace(0, self.nmb_branches_final-1, self.nmb_imgs_show).astype(np.int32)
        list_imgs_preview = []
        for j in idx_list:
            list_imgs_preview.append(Image.fromarray(imgs_transition[j]))
            
        # Save the preview imgs as jpgs on disk so we are not sending umcompressed data around
        self.current_timestamp = get_time('second')
        self.list_fp_imgs_current = []
        for i in range(len(list_imgs_preview)):
            fp_img = f"img_preview_{i}_{self.current_timestamp}.jpg"
            list_imgs_preview[i].save(fp_img)
            self.list_fp_imgs_current.append(fp_img)
        
        # Insert cheap frames for the movie
        imgs_transition_ext = add_frames_linear_interp(imgs_transition, self.duration, self.fps)

        # Save as movie
        fp_movie = self.get_fp_movie(self.current_timestamp)
        if os.path.isfile(fp_movie):
            os.remove(fp_movie)
        ms = MovieSaver(fp_movie, fps=self.fps)
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()
        print("DONE SAVING MOVIE! SENDING BACK...")
        
        # Assemble Output, updating the preview images and le movie
        list_return = self.list_fp_imgs_current + [fp_movie]
        return list_return

    def get_fp_movie(self, timestamp, is_stacked=False):
        if not is_stacked:
            return f"movie_{timestamp}.mp4"
        else:
            return f"movie_stacked_{timestamp}.mp4"
            
    
    def stack_forward(self, prompt2, seed2):
        # Save preview images, prompts and seeds into dictionary for stacking
        self.dict_multi_trans[self.current_timestamp] = generate_list_output(self.prompt1, self.prompt2, self.seed1, self.seed2, self.list_fp_imgs_current)
        self.dict_multi_trans_include[self.current_timestamp] = True
        
        self.lb.swap_forward()
        list_out = [self.list_fp_imgs_current[-1]]
        list_out.extend([self.fp_img_empty]*4)
        list_out.append(prompt2)
        list_out.append(seed2)
        list_out.append("")
        list_out.append(np.random.randint(0, 10000000))
        
        list_out_multi_tab = self.update_trans_stacks()
        
        list_out.extend(list_out_multi_tab)
        # self.nmb_trans_stack = len(self.dict_multi_trans_include)
        return list_out

    def stack_movie(self):
        # collect all that are in...
        list_fp_movies = []
        for timestamp in self.multi_trans_currently_shown:
            if timestamp is not None:
                list_fp_movies.append(self.get_fp_movie(timestamp))
        
        fp_stacked = self.get_fp_movie(get_time('second'), True)
        concatenate_movies(fp_stacked, list_fp_movies)
        return fp_stacked
        

    def get_state_dict(self):
        state_dict = {}
        grab_vars = ['prompt1', 'prompt2', 'seed1', 'seed2', 'height', 'width',
                     'num_inference_steps', 'depth_strength', 'guidance_scale',
                     'guidance_scale_mid_damper', 'mid_compression_scaler']
        
        for v in grab_vars:
            state_dict[v] = getattr(self, v)
        return state_dict
    
    
    def update_trans_stacks(self):
        print("Updating transition stack...")
        
        self.multi_trans_currently_shown = []
        list_output = []
        # Figure out which transitions should be shown
        for timestamp in self.dict_multi_trans_include.keys():
            if len(self.multi_trans_currently_shown) >= self.nmb_trans_stack:
                continue
            
            if self.dict_multi_trans_include[timestamp]:
                last_timestamp_vals = self.dict_multi_trans[timestamp]
                list_output.extend(self.dict_multi_trans[timestamp])
                self.multi_trans_currently_shown.append(timestamp)
                print(f"including timestamp: {timestamp}")
        
        # Fill with empty images if below nmb_trans_stack
        nmb_empty_missing = self.nmb_trans_stack - len(self.multi_trans_currently_shown)
        for i in range(nmb_empty_missing):
            list_output.extend([gr.update(visible=False)]*len(last_timestamp_vals))
            self.multi_trans_currently_shown.append(None)
        
        return list_output
        

    def remove_trans(self, idx_row):
        idx_row = int(idx_row)
        # do removal...
        if idx_row < len(self.multi_trans_currently_shown):
            timestamp = self.multi_trans_currently_shown[idx_row]
            if timestamp in self.dict_multi_trans_include.keys():
                self.dict_multi_trans_include[timestamp] = False
                print(f"remove_trans called: {timestamp}")
        else:
            print(f"remove_trans called: idx_row too large {idx_row}")
            
        return self.update_trans_stacks()

def get_img_rand():
    return (255*np.random.rand(self.height,self.width,3)).astype(np.uint8)

def generate_list_output(
        prompt1,
        prompt2,
        seed1,
        seed2,
        list_fp_imgs,
        ):
    list_output = []
    list_output.append(prompt1)
    list_output.append(prompt2)
    list_output.append(seed1)
    list_output.append(seed2)
    for fp_img in list_fp_imgs:
        list_output.append(fp_img)
    
    return list_output


        
if __name__ == "__main__":    
    
    # fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_768-ema-pruned.ckpt" 
    fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 
    sdh = StableDiffusionHolder(fp_ckpt)
    
    self = BlendingFrontend(sdh) # Yes this is possible in python and yes it is an awesome trick
    
    with gr.Blocks() as demo:
        with gr.Row():
            prompt1 = gr.Textbox(label="prompt 1")
            prompt2 = gr.Textbox(label="prompt 2")
            negative_prompt = gr.Textbox(label="negative prompt")          
            
        with gr.Row():
            nmb_branches_final = gr.Slider(5, 125, self.nmb_branches_final, step=4, label='nmb trans images', interactive=True) 
            height = gr.Slider(256, 2048, self.height, step=128, label='height', interactive=True)
            width = gr.Slider(256, 2048, self.width, step=128, label='width', interactive=True) 
            
        with gr.Row():
            num_inference_steps = gr.Slider(5, 100, self.num_inference_steps, step=1, label='num_inference_steps', interactive=True)
            branch1_influence = gr.Slider(0.0, 1.0, self.branch1_influence, step=0.01, label='branch1_influence', interactive=True) 
            guidance_scale = gr.Slider(1, 25, self.guidance_scale, step=0.1, label='guidance_scale', interactive=True) 
    
        with gr.Row():
            depth_strength = gr.Slider(0.01, 0.99, self.depth_strength, step=0.01, label='depth_strength', interactive=True) 
            duration = gr.Slider(0.1, 30, self.duration, step=0.1, label='video duration', interactive=True) 
            guidance_scale_mid_damper = gr.Slider(0.01, 2.0, self.guidance_scale_mid_damper, step=0.01, label='guidance_scale_mid_damper', interactive=True) 
            
        with gr.Row():
            seed1 = gr.Number(42, label="seed 1", interactive=True)
            b_newseed1 = gr.Button("randomize seed 1", variant='secondary')
            seed2 = gr.Number(420, label="seed 2", interactive=True)
            b_newseed2 = gr.Button("randomize seed 2", variant='secondary')
        with gr.Row():
            b_compute_transition = gr.Button('compute transition', variant='primary')
        
        with gr.Row():
            img1 = gr.Image(label="1/5")
            img2 = gr.Image(label="2/5")
            img3 = gr.Image(label="3/5")
            img4 = gr.Image(label="4/5")
            img5 = gr.Image(label="5/5")
        
        with gr.Row():
            vid_transition = gr.Video()
        
        # Bind the on-change methods
        depth_strength.change(fn=self.change_depth_strength, inputs=depth_strength)
        num_inference_steps.change(fn=self.change_num_inference_steps, inputs=num_inference_steps)
        nmb_branches_final.change(fn=self.change_nmb_branches_final, inputs=nmb_branches_final)
        
        guidance_scale.change(fn=self.change_guidance_scale, inputs=guidance_scale)
        guidance_scale_mid_damper.change(fn=self.change_guidance_scale_mid_damper, inputs=guidance_scale_mid_damper)
        
        height.change(fn=self.change_height, inputs=height)
        width.change(fn=self.change_width, inputs=width)
        negative_prompt.change(fn=self.change_negative_prompt, inputs=negative_prompt)
        seed1.change(fn=self.change_seed1, inputs=seed1)
        seed2.change(fn=self.change_seed2, inputs=seed2)
        duration.change(fn=self.change_duration, inputs=duration)
        branch1_influence.change(fn=self.change_branch1_influence, inputs=branch1_influence)
    
        b_newseed1.click(self.randomize_seed1, outputs=seed1)
        b_newseed2.click(self.randomize_seed2, outputs=seed2)
        # b_stackforward.click(self.stack_forward, 
        #                      inputs=[prompt2, seed2], 
        #                      outputs=[img1, img2, img3, img4, img5, prompt1, seed1, prompt2])
        b_compute_transition.click(self.compute_transition, 
                                   inputs=[prompt1, prompt2],
                                   outputs=[img1, img2, img3, img4, img5, vid_transition])
        


    demo.launch(share=self.share, inbrowser=True, inline=False)
