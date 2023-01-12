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
from movie_util import MovieSaver
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

        self.num_inference_steps = 30
        self.depth_strength = 0.25
        self.seed1 = 42
        self.seed2 = 420
        self.guidance_scale = 4.0
        self.guidance_scale_mid_damper = 0.5
        self.mid_compression_scaler = 1.2
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""
        self.dp_base = "/output/"
        self.list_settings = []
        self.state_prev = {}
        self.state_current = {}
        self.showing_current = True
        self.branch1_influence = 0.3
        self.imgs_show_last = []
        self.imgs_show_current = []
        self.nmb_branches_final = 13
        self.nmb_imgs_show = 5
        self.fps = 30
        self.duration = 10
        self.max_size_imgs = 200 # gradio otherwise mega slow 
        
        if not self.use_debug:
            self.lb.sdh.num_inference_steps = self.num_inference_steps
            self.height = self.lb.sdh.height
            self.width = self.lb.sdh.width
        else:
            self.height = 768
            self.width = 768
        
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
        
    def change_prompt1(self, value):
        self.prompt1 = value
        # print(f"changed prompt1 to {value}")
        
    def change_prompt2(self, value):
        self.prompt2 = value
        # print(f"changed prompt2 to {value}")
        
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
        
    
    def downscale_imgs(self, list_imgs):
        return [l.resize((self.max_size_imgs, self.max_size_imgs)) for l in list_imgs]
    
    def run(self, x):
        print("STARTING DIFFUSION!")
        self.state_prev = self.state_current.copy()
        self.state_current = self.get_state_dict()
        # Copy last iteration
        self.imgs_show_last = copy.deepcopy(self.imgs_show_current)
        
        if self.use_debug:
            list_imgs = [(255*np.random.rand(self.height,self.width,3)).astype(np.uint8) for l in range(5)]
            list_imgs = [Image.fromarray(l) for l in list_imgs]
            list_imgs = self.downscale_imgs(list_imgs)
            self.imgs_show_current = copy.deepcopy(list_imgs)
            print("DONE! SENDING BACK RESULTS")
            return list_imgs
        
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
        imgs_transition = self.lb.run_transition(fixed_seeds=fixed_seeds)
        imgs_transition = [Image.fromarray(l) for l in imgs_transition]
        print(f"DONE DIFFUSION! Resulted in {len(imgs_transition)} images")
        
        assert np.mod((self.nmb_branches_final-self.nmb_imgs_show)/4, 1)==0, 'self.nmb_branches_final illegal value!'
        idx_list = np.linspace(0, self.nmb_branches_final-1, self.nmb_imgs_show).astype(np.int32)
        list_imgs = []
        for j in idx_list:
            list_imgs.append(imgs_transition[j])
        
        list_imgs = self.downscale_imgs(list_imgs)
        
        self.imgs_show_current = copy.deepcopy(list_imgs)
        return list_imgs
    

        
    def save(self):
        if self.lb.tree_final_imgs[0] is None:
            return
        print("save is called!")
        dp_img = os.path.join(self.dp_base, get_time("second"))
        imgs_transition = self.lb.tree_final_imgs
        self.lb.write_imgs_transition(dp_img, imgs_transition)
        
        fps = self.fps
        # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
        imgs_transition_ext = add_frames_linear_interp(imgs_transition, self.duration, fps)

        # Save as MP4
        fp_movie = os.path.join(dp_img, "movie_lowres.mp4")
        if os.path.isfile(fp_movie):
            os.remove(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps)
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()
        return fp_movie


        
    def get_state_dict(self):
        state_dict = {}
        grab_vars = ['prompt1', 'prompt2', 'seed1', 'seed2', 'height', 'width',
                     'num_inference_steps', 'depth_strength', 'guidance_scale',
                     'guidance_scale_mid_damper', 'mid_compression_scaler']
        
        for v in grab_vars:
            state_dict[v] = getattr(self, v)
        return state_dict
        
        
    def compare_last(self):
        if len(self.state_prev) == 0 or len(self.state_current) == 0:
            return ""
        
        if self.showing_current:
            # inject the last images that were shown and return str of changes
            str_fill = "showing last version: "
            list_return = self.imgs_show_last
            idx = 0
            verb = 'was'
            self.showing_current = False

        elif not self.showing_current:
            # inject the current images and show no string
            str_fill = "showing current version: "
            verb = 'is'
            idx = 1
            list_return = self.imgs_show_current
            self.showing_current = True
        
        dict_diff = compare_dicts(self.state_prev, self.state_current)
        for key in dict_diff:
            str_fill += f"{key} {verb} {dict_diff[key][idx]}, "
        str_fill = str_fill[:-2]
        list_return.extend([str_fill])
        return list_return 
        
if __name__ == "__main__":    
    
    fp_ckpt = "../stable_diffusion_models/ckpt/v2-1_512-ema-pruned.ckpt" 
    sdh = StableDiffusionHolder(fp_ckpt)
    
    self = BlendingFrontend(sdh)
    
    
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
            guidance_scale_mid_damper = gr.Slider(0.01, 2.0, self.guidance_scale_mid_damper, step=0.01, label='guidance_scale_mid_damper', interactive=True) 
            mid_compression_scaler = gr.Slider(1.0, 2.0, self.mid_compression_scaler, step=0.01, label='mid_compression_scaler', interactive=True) 
                
        with gr.Row():
            b_newseed1 = gr.Button("rand seed 1")
            seed1 = gr.Number(42, label="seed 1", interactive=True)
            b_newseed2 = gr.Button("rand seed 2")
            seed2 = gr.Number(420, label="seed 2", interactive=True)
            b_compare = gr.Button("compare")
            
        with gr.Row():
            b_run = gr.Button('run preview!')
            
        with gr.Row():
            img1 = gr.Image(label="1/5")
            img2 = gr.Image(label="2/5")
            img3 = gr.Image(label="3/5")
            img4 = gr.Image(label="4/5")
            img5 = gr.Image(label="5/5")
            
        with gr.Row():
            compare_text = gr.Textbox(label="")
            
        with gr.Row():
            fps = gr.Slider(1, 120, self.fps, step=1, label='fps', interactive=True)
            duration = gr.Slider(0.1, 30, self.duration, step=0.1, label='duration', interactive=True) 
            b_save = gr.Button('save video')
        
        with gr.Row():
            vid = gr.Video()
    
        # Bind the on-change methods
        depth_strength.change(fn=self.change_depth_strength, inputs=depth_strength)
        num_inference_steps.change(fn=self.change_num_inference_steps, inputs=num_inference_steps)
        nmb_branches_final.change(fn=self.change_nmb_branches_final, inputs=nmb_branches_final)
        
        guidance_scale.change(fn=self.change_guidance_scale, inputs=guidance_scale)
        guidance_scale_mid_damper.change(fn=self.change_guidance_scale_mid_damper, inputs=guidance_scale_mid_damper)
        mid_compression_scaler.change(fn=self.change_mid_compression_scaler, inputs=mid_compression_scaler)
        
        height.change(fn=self.change_height, inputs=height)
        width.change(fn=self.change_width, inputs=width)
        prompt1.change(fn=self.change_prompt1, inputs=prompt1)
        prompt2.change(fn=self.change_prompt2, inputs=prompt2)
        negative_prompt.change(fn=self.change_negative_prompt, inputs=negative_prompt)
        seed1.change(fn=self.change_seed1, inputs=seed1)
        seed2.change(fn=self.change_seed2, inputs=seed2)
        fps.change(fn=self.change_fps, inputs=fps)
        duration.change(fn=self.change_duration, inputs=duration)
        branch1_influence.change(fn=self.change_branch1_influence, inputs=branch1_influence)
    
        b_newseed1.click(self.randomize_seed1, outputs=seed1)
        b_newseed2.click(self.randomize_seed2, outputs=seed2)
        b_compare.click(self.compare_last, outputs=[img1, img2, img3, img4, img5, compare_text])
        b_run.click(self.run, outputs=[img1, img2, img3, img4, img5])
        b_save.click(self.save, outputs=vid)
    
    demo.launch(share=self.share)
