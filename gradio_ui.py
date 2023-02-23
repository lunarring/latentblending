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

import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import warnings
from tqdm.auto import tqdm
from PIL import Image
from movie_util import MovieSaver, concatenate_movies
from latent_blending import LatentBlending
from stable_diffusion_holder import StableDiffusionHolder
import gradio as gr
from dotenv import find_dotenv, load_dotenv
import shutil
import random
from utils import get_time, add_frames_linear_interp
from huggingface_hub import hf_hub_download


class BlendingFrontend():
    def __init__(
            self,
            sdh,
            share=False):
        r"""
        Gradio Helper Class to collect UI data and start latent blending.
        Args:
            sdh:
                StableDiffusionHolder
            share: bool
                Set true to get a shareable gradio link (e.g. for running a remote server)
        """
        self.share = share

        # UI Defaults
        self.num_inference_steps = 30
        self.depth_strength = 0.25
        self.seed1 = 420
        self.seed2 = 420
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""
        self.fps = 30
        self.duration_video = 8
        self.t_compute_max_allowed = 10

        self.lb = LatentBlending(sdh)
        self.lb.sdh.num_inference_steps = self.num_inference_steps
        self.init_parameters_from_lb()
        self.init_save_dir()

        # Vars
        self.list_fp_imgs_current = []
        self.recycle_img1 = False
        self.recycle_img2 = False
        self.list_all_segments = []
        self.dp_session = ""
        self.user_id = None

    def init_parameters_from_lb(self):
        r"""
        Automatically init parameters from latentblending instance
        """
        self.height = self.lb.sdh.height
        self.width = self.lb.sdh.width
        self.guidance_scale = self.lb.guidance_scale
        self.guidance_scale_mid_damper = self.lb.guidance_scale_mid_damper
        self.mid_compression_scaler = self.lb.mid_compression_scaler
        self.branch1_crossfeed_power = self.lb.branch1_crossfeed_power
        self.branch1_crossfeed_range = self.lb.branch1_crossfeed_range
        self.branch1_crossfeed_decay = self.lb.branch1_crossfeed_decay
        self.parental_crossfeed_power = self.lb.parental_crossfeed_power
        self.parental_crossfeed_range = self.lb.parental_crossfeed_range
        self.parental_crossfeed_power_decay = self.lb.parental_crossfeed_power_decay

    def init_save_dir(self):
        r"""
        Initializes the directory where stuff is being saved.
        You can specify this directory in a ".env" file in your latentblending root, setting
        DIR_OUT='/path/to/saving'
        """
        load_dotenv(find_dotenv(), verbose=False)
        self.dp_out = os.getenv("DIR_OUT")
        if self.dp_out is None:
            self.dp_out = ""
        self.dp_imgs = os.path.join(self.dp_out, "imgs")
        os.makedirs(self.dp_imgs, exist_ok=True)
        self.dp_movies = os.path.join(self.dp_out, "movies")
        os.makedirs(self.dp_movies, exist_ok=True)
        self.save_empty_image()

    def save_empty_image(self):
        r"""
        Saves an empty/black dummy image.
        """
        self.fp_img_empty = os.path.join(self.dp_imgs, 'empty.jpg')
        Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8)).save(self.fp_img_empty, quality=5)

    def randomize_seed1(self):
        r"""
        Randomizes the first seed
        """
        seed = np.random.randint(0, 10000000)
        self.seed1 = int(seed)
        print(f"randomize_seed1: new seed = {self.seed1}")
        return seed

    def randomize_seed2(self):
        r"""
        Randomizes the second seed
        """
        seed = np.random.randint(0, 10000000)
        self.seed2 = int(seed)
        print(f"randomize_seed2: new seed = {self.seed2}")
        return seed

    def setup_lb(self, list_ui_vals):
        r"""
        Sets all parameters from the UI. Since gradio does not support to pass dictionaries,
        we have to instead pass keys (list_ui_keys, global) and values (list_ui_vals)
        """
        # Collect latent blending variables
        self.lb.set_width(list_ui_vals[list_ui_keys.index('width')])
        self.lb.set_height(list_ui_vals[list_ui_keys.index('height')])
        self.lb.set_prompt1(list_ui_vals[list_ui_keys.index('prompt1')])
        self.lb.set_prompt2(list_ui_vals[list_ui_keys.index('prompt2')])
        self.lb.set_negative_prompt(list_ui_vals[list_ui_keys.index('negative_prompt')])
        self.lb.guidance_scale = list_ui_vals[list_ui_keys.index('guidance_scale')]
        self.lb.guidance_scale_mid_damper = list_ui_vals[list_ui_keys.index('guidance_scale_mid_damper')]
        self.t_compute_max_allowed = list_ui_vals[list_ui_keys.index('duration_compute')]
        self.lb.num_inference_steps = list_ui_vals[list_ui_keys.index('num_inference_steps')]
        self.lb.sdh.num_inference_steps = list_ui_vals[list_ui_keys.index('num_inference_steps')]
        self.duration_video = list_ui_vals[list_ui_keys.index('duration_video')]
        self.lb.seed1 = list_ui_vals[list_ui_keys.index('seed1')]
        self.lb.seed2 = list_ui_vals[list_ui_keys.index('seed2')]
        self.lb.branch1_crossfeed_power = list_ui_vals[list_ui_keys.index('branch1_crossfeed_power')]
        self.lb.branch1_crossfeed_range = list_ui_vals[list_ui_keys.index('branch1_crossfeed_range')]
        self.lb.branch1_crossfeed_decay = list_ui_vals[list_ui_keys.index('branch1_crossfeed_decay')]
        self.lb.parental_crossfeed_power = list_ui_vals[list_ui_keys.index('parental_crossfeed_power')]
        self.lb.parental_crossfeed_range = list_ui_vals[list_ui_keys.index('parental_crossfeed_range')]
        self.lb.parental_crossfeed_power_decay = list_ui_vals[list_ui_keys.index('parental_crossfeed_power_decay')]
        self.num_inference_steps = list_ui_vals[list_ui_keys.index('num_inference_steps')]
        self.depth_strength = list_ui_vals[list_ui_keys.index('depth_strength')]

        if len(list_ui_vals[list_ui_keys.index('user_id')]) > 1:
            self.user_id = list_ui_vals[list_ui_keys.index('user_id')]
        else:
            # generate new user id
            self.user_id = ''.join((random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8)))
            print(f"made new user_id: {self.user_id} at {get_time('second')}")

    def save_latents(self, fp_latents, list_latents):
        r"""
        Saves a latent trajectory on disk, in npy format.
        """
        list_latents_cpu = [l.cpu().numpy() for l in list_latents]
        np.save(fp_latents, list_latents_cpu)

    def load_latents(self, fp_latents):
        r"""
        Loads a latent trajectory from disk, converts to torch tensor.
        """
        list_latents_cpu = np.load(fp_latents)
        list_latents = [torch.from_numpy(l).to(self.lb.device) for l in list_latents_cpu]
        return list_latents

    def compute_img1(self, *args):
        r"""
        Computes the first transition image and returns it for display.
        Sets all other transition images and last image to empty (as they are obsolete with this operation)
        """
        list_ui_vals = args
        self.setup_lb(list_ui_vals)
        fp_img1 = os.path.join(self.dp_imgs, f"img1_{self.user_id}")
        img1 = Image.fromarray(self.lb.compute_latents1(return_image=True))
        img1.save(fp_img1 + ".jpg")
        self.save_latents(fp_img1 + ".npy", self.lb.tree_latents[0])
        self.recycle_img1 = True
        self.recycle_img2 = False
        return [fp_img1 + ".jpg", self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.user_id]

    def compute_img2(self, *args):
        r"""
        Computes the last transition image and returns it for display.
        Sets all other transition images to empty (as they are obsolete with this operation)
        """
        if not os.path.isfile(os.path.join(self.dp_imgs, f"img1_{self.user_id}.jpg")):  # don't do anything
            return [self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, self.user_id]
        list_ui_vals = args
        self.setup_lb(list_ui_vals)

        self.lb.tree_latents[0] = self.load_latents(os.path.join(self.dp_imgs, f"img1_{self.user_id}.npy"))
        fp_img2 = os.path.join(self.dp_imgs, f"img2_{self.user_id}")
        img2 = Image.fromarray(self.lb.compute_latents2(return_image=True))
        img2.save(fp_img2 + '.jpg')
        self.save_latents(fp_img2 + ".npy", self.lb.tree_latents[-1])
        self.recycle_img2 = True
        # fixme save seeds. change filenames?
        return [self.fp_img_empty, self.fp_img_empty, self.fp_img_empty, fp_img2 + ".jpg", self.user_id]

    def compute_transition(self, *args):
        r"""
        Computes transition images and movie.
        """
        list_ui_vals = args
        self.setup_lb(list_ui_vals)
        print("STARTING TRANSITION...")
        fixed_seeds = [self.seed1, self.seed2]
        # Inject loaded latents (other user interference)
        self.lb.tree_latents[0] = self.load_latents(os.path.join(self.dp_imgs, f"img1_{self.user_id}.npy"))
        self.lb.tree_latents[-1] = self.load_latents(os.path.join(self.dp_imgs, f"img2_{self.user_id}.npy"))
        imgs_transition = self.lb.run_transition(
            recycle_img1=self.recycle_img1,
            recycle_img2=self.recycle_img2,
            num_inference_steps=self.num_inference_steps,
            depth_strength=self.depth_strength,
            t_compute_max_allowed=self.t_compute_max_allowed,
            fixed_seeds=fixed_seeds)
        print(f"Latent Blending pass finished ({get_time('second')}). Resulted in {len(imgs_transition)} images")

        # Subselect three preview images
        idx_img_prev = np.round(np.linspace(0, len(imgs_transition) - 1, 5)[1:-1]).astype(np.int32)

        list_imgs_preview = []
        for j in idx_img_prev:
            list_imgs_preview.append(Image.fromarray(imgs_transition[j]))

        # Save the preview imgs as jpgs on disk so we are not sending umcompressed data around
        current_timestamp = get_time('second')
        self.list_fp_imgs_current = []
        for i in range(len(list_imgs_preview)):
            fp_img = os.path.join(self.dp_imgs, f"img_preview_{i}_{current_timestamp}.jpg")
            list_imgs_preview[i].save(fp_img)
            self.list_fp_imgs_current.append(fp_img)
        # Insert cheap frames for the movie
        imgs_transition_ext = add_frames_linear_interp(imgs_transition, self.duration_video, self.fps)

        # Save as movie
        self.fp_movie = self.get_fp_video_last()
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
        r"""
        Allows to generate multi-segment movies. Sets last image -> first image with all
        relevant parameters.
        """
        # Save preview images, prompts and seeds into dictionary for stacking
        if len(self.list_all_segments) == 0:
            timestamp_session = get_time('second')
            self.dp_session = os.path.join(self.dp_out, f"session_{timestamp_session}")
            os.makedirs(self.dp_session)

        idx_segment = len(self.list_all_segments)
        dp_segment = os.path.join(self.dp_session, f"segment_{str(idx_segment).zfill(3)}")

        self.list_all_segments.append(dp_segment)
        self.lb.write_imgs_transition(dp_segment)

        fp_movie_last = self.get_fp_video_last()
        fp_movie_next = self.get_fp_video_next()

        shutil.copyfile(fp_movie_last, fp_movie_next)

        self.lb.tree_latents[0] = self.load_latents(os.path.join(self.dp_imgs, f"img1_{self.user_id}.npy"))
        self.lb.tree_latents[-1] = self.load_latents(os.path.join(self.dp_imgs, f"img2_{self.user_id}.npy"))
        self.lb.swap_forward()

        shutil.copyfile(os.path.join(self.dp_imgs, f"img2_{self.user_id}.npy"), os.path.join(self.dp_imgs, f"img1_{self.user_id}.npy"))
        fp_multi = self.multi_concat()
        list_out = [fp_multi]

        list_out.extend([os.path.join(self.dp_imgs, f"img2_{self.user_id}.jpg")])
        list_out.extend([self.fp_img_empty] * 4)
        list_out.append(gr.update(interactive=False, value=prompt2))
        list_out.append(gr.update(interactive=False, value=seed2))
        list_out.append("")
        list_out.append(np.random.randint(0, 10000000))
        print(f"stack_forward: fp_multi {fp_multi}")
        return list_out

    def multi_concat(self):
        r"""
        Concatentates all stacked segments into one long movie.
        """
        list_fp_movies = self.get_fp_video_all()
        # Concatenate movies and save
        fp_final = os.path.join(self.dp_session, f"concat_{self.user_id}.mp4")
        concatenate_movies(fp_final, list_fp_movies)
        return fp_final

    def get_fp_video_all(self):
        r"""
        Collects all stacked movie segments.
        """
        list_all = os.listdir(self.dp_movies)
        str_beg = f"movie_{self.user_id}_"
        list_user = [l for l in list_all if str_beg in l]
        list_user.sort()
        list_user = [os.path.join(self.dp_movies, l) for l in list_user]
        return list_user

    def get_fp_video_next(self):
        r"""
        Gets the filepath of the next movie segment.
        """
        list_videos = self.get_fp_video_all()
        if len(list_videos) == 0:
            idx_next = 0
        else:
            idx_next = len(list_videos)
        fp_video_next = os.path.join(self.dp_movies, f"movie_{self.user_id}_{str(idx_next).zfill(3)}.mp4")
        return fp_video_next

    def get_fp_video_last(self):
        r"""
        Gets the current video that was saved.
        """
        fp_video_last = os.path.join(self.dp_movies, f"last_{self.user_id}.mp4")
        return fp_video_last


if __name__ == "__main__":
    fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", filename="v2-1_512-ema-pruned.ckpt")
    # fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1", filename="v2-1_768-ema-pruned.ckpt")
    bf = BlendingFrontend(StableDiffusionHolder(fp_ckpt))
    # self = BlendingFrontend(None)

    with gr.Blocks() as demo:
        gr.HTML("""<h1>Latent Blending</h1>
<p>Create butter-smooth transitions between prompts, powered by stable diffusion</p>
<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<br/>
<a href="https://huggingface.co/spaces/lunarring/latentblending?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>""")

        with gr.Row():
            prompt1 = gr.Textbox(label="prompt 1")
            prompt2 = gr.Textbox(label="prompt 2")

        with gr.Row():
            duration_compute = gr.Slider(10, 25, bf.t_compute_max_allowed, step=1, label='waiting time', interactive=True)
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
            b_compute1 = gr.Button('step1: compute first image', variant='primary')
            b_compute2 = gr.Button('step2: compute last image', variant='primary')
            b_compute_transition = gr.Button('step3: compute transition', variant='primary')

        with gr.Row():
            img1 = gr.Image(label="1/5")
            img2 = gr.Image(label="2/5", show_progress=False)
            img3 = gr.Image(label="3/5", show_progress=False)
            img4 = gr.Image(label="4/5", show_progress=False)
            img5 = gr.Image(label="5/5")

        with gr.Row():
            vid_single = gr.Video(label="current single trans")
            vid_multi = gr.Video(label="concatented multi trans")

        with gr.Row():
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

                ## Last image crossfeeding
                - branch1_crossfeed_power: Controls the level of cross-feeding between the first and last image branch. For preserving structures.
                - branch1_crossfeed_range: Sets the duration of active crossfeed during development. High values enforce strong structural similarity.
                - branch1_crossfeed_decay: Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.

                ## Transition settings
                - parental_crossfeed_power: Similar to branch1_crossfeed_power, however applied for the images withinin the transition.
                - parental_crossfeed_range: Similar to branch1_crossfeed_range, however applied for the images withinin the transition.
                - parental_crossfeed_power_decay: Similar to branch1_crossfeed_decay, however applied for the images withinin the transition.
                - depth_strength: Determines when the blending process will begin in terms of diffusion steps. Low values more inventive but can cause motion.
                - guidance_scale_mid_damper: Decreases the guidance scale in the middle of a transition.
                """)

        with gr.Row():
            user_id = gr.Textbox(label="user id", interactive=False)

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
        dict_ui_elem["user_id"] = user_id

        # Convert to list, as gradio doesn't seem to accept dicts
        list_ui_vals = []
        list_ui_keys = []
        for k in dict_ui_elem.keys():
            list_ui_vals.append(dict_ui_elem[k])
            list_ui_keys.append(k)
        bf.list_ui_keys = list_ui_keys

        b_newseed1.click(bf.randomize_seed1, outputs=seed1)
        b_newseed2.click(bf.randomize_seed2, outputs=seed2)
        b_compute1.click(bf.compute_img1, inputs=list_ui_vals, outputs=[img1, img2, img3, img4, img5, user_id])
        b_compute2.click(bf.compute_img2, inputs=list_ui_vals, outputs=[img2, img3, img4, img5, user_id])
        b_compute_transition.click(bf.compute_transition,
                                   inputs=list_ui_vals,
                                   outputs=[img2, img3, img4, vid_single])

        b_stackforward.click(bf.stack_forward,
                             inputs=[prompt2, seed2],
                             outputs=[vid_multi, img1, img2, img3, img4, img5, prompt1, seed1, prompt2])

    demo.launch(share=bf.share, inbrowser=True, inline=False)
