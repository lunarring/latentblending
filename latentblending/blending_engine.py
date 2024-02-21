import os
import torch
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
from PIL import Image
from typing import List, Optional
import lpips
import platform
from latentblending.diffusers_holder import DiffusersHolder
from latentblending.utils import interpolate_spherical, interpolate_linear, add_frames_linear_interp
from lunar_tools import MovieSaver, fill_up_frames_linear_interpolation
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)


class BlendingEngine():
    def __init__(
            self,
            pipe: None,
            do_compile: bool = False,
            guidance_scale_mid_damper: float = 0.5,
            mid_compression_scaler: float = 1.2):
        r"""
        Initializes the latent blending class.
        Args:
            pipe: diffusers pipeline (SDXL)
            do_compile: compile pipeline for faster inference using stable fast
            guidance_scale_mid_damper: float = 0.5
                Reduces the guidance scale towards the middle of the transition.
                A value of 0.5 would decrease the guidance_scale towards the middle linearly by 0.5.
            mid_compression_scaler: float = 2.0
                Increases the sampling density in the middle (where most changes happen). Higher value
                imply more values in the middle. However the inflection point can occur outside the middle,
                thus high values can give rough transitions. Values around 2 should be fine.
        """
        assert guidance_scale_mid_damper > 0 \
            and guidance_scale_mid_damper <= 1.0, \
            f"guidance_scale_mid_damper neees to be in interval (0,1], you provided {guidance_scale_mid_damper}"

    
        self.dh = DiffusersHolder(pipe)
        self.device = self.dh.device
        self.set_dimensions()

        self.guidance_scale_mid_damper = guidance_scale_mid_damper
        self.mid_compression_scaler = mid_compression_scaler
        self.seed1 = 0
        self.seed2 = 0

        # Initialize vars
        self.prompt1 = ""
        self.prompt2 = ""

        self.tree_latents = [None, None]
        self.tree_fracts = None
        self.idx_injection = []
        self.tree_status = None
        self.tree_final_imgs = []

        self.text_embedding1 = None
        self.text_embedding2 = None
        self.image1_lowres = None
        self.image2_lowres = None
        self.negative_prompt = None

        self.set_guidance_scale()
        self.multi_transition_img_first = None
        self.multi_transition_img_last = None
        self.dt_unet_step = 0
        if platform.system() == "Darwin":
            self.lpips = lpips.LPIPS(net='alex')
        else:
            self.lpips = lpips.LPIPS(net='alex').cuda(self.device)

        self.set_prompt1("")
        self.set_prompt2("")
        
        self.set_branch1_crossfeed()
        self.set_parental_crossfeed()
        
        self.set_num_inference_steps()
        self.benchmark_speed()
        self.set_branching()
        
        if do_compile:
            print("starting compilation")
            from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
            self.dh.pipe.enable_xformers_memory_efficient_attention()
            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.dh.pipe = compile(self.dh.pipe, config)
        
        
        
    def benchmark_speed(self):
        """
        Measures the time per diffusion step and for the vae decoding
        """
        print("starting speed benchmark...")
        text_embeddings = self.dh.get_text_embedding("test")
        latents_start = self.dh.get_noise(np.random.randint(111111))
        # warmup
        list_latents = self.dh.run_diffusion_sd_xl(text_embeddings=text_embeddings, latents_start=latents_start, return_image=False, idx_start=self.num_inference_steps-1)
        # bench unet
        t0 = time.time()
        list_latents = self.dh.run_diffusion_sd_xl(text_embeddings=text_embeddings, latents_start=latents_start, return_image=False, idx_start=self.num_inference_steps-1)
        self.dt_unet_step = time.time() - t0
        
        # bench vae
        t0 = time.time()
        img = self.dh.latent2image(list_latents[-1])
        self.dt_vae = time.time() - t0
        print(f"time per unet iteration: {self.dt_unet_step} time for vae: {self.dt_vae}")

    def set_dimensions(self, size_output=None):
        r"""
        sets the size of the output video.
        Args:
            size_output: tuple
                width x height
                Note: the size will get automatically adjusted to be divisable by 32.
        """
        if size_output is None:
            if self.dh.is_sdxl_turbo:
                size_output = (512, 512)
            else:
                size_output = (1024, 1024)
        self.dh.set_dimensions(size_output)

    def set_guidance_scale(self, guidance_scale=None):
        r"""
        sets the guidance scale.
        """
        if guidance_scale is None:
            if self.dh.is_sdxl_turbo:
                guidance_scale = 0.0
            else:
                guidance_scale = 4.0
        
        self.guidance_scale_base = guidance_scale
        self.guidance_scale = guidance_scale
        self.dh.guidance_scale = guidance_scale

    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported
        """
        self.negative_prompt = negative_prompt
        self.dh.set_negative_prompt(negative_prompt)

    def set_guidance_mid_dampening(self, fract_mixing):
        r"""
        Tunes the guidance scale down as a linear function of fract_mixing,
        towards 0.5 the minimum will be reached.
        """
        mid_factor = 1 - np.abs(fract_mixing - 0.5) / 0.5
        max_guidance_reduction = self.guidance_scale_base * (1 - self.guidance_scale_mid_damper) - 1
        guidance_scale_effective = self.guidance_scale_base - max_guidance_reduction * mid_factor
        self.guidance_scale = guidance_scale_effective
        self.dh.guidance_scale = guidance_scale_effective

    def set_branch1_crossfeed(self, crossfeed_power=0, crossfeed_range=0, crossfeed_decay=0):
        r"""
        Sets the crossfeed parameters for the first branch to the last branch.
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding between the first and last image branch.
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        self.branch1_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.branch1_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.branch1_crossfeed_decay = np.clip(crossfeed_decay, 0, 1)

    def set_parental_crossfeed(self, crossfeed_power=None, crossfeed_range=None, crossfeed_decay=None):
        r"""
        Sets the crossfeed parameters for all transition images (within the first and last branch).
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding from the parental branches
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        
        if self.dh.is_sdxl_turbo:
            if crossfeed_power is None:
                crossfeed_power = 1.0
            if crossfeed_range is None:
                crossfeed_range = 1.0
            if crossfeed_decay is None:
                crossfeed_decay = 1.0
        else:
            crossfeed_power = 0.3
            crossfeed_range = 0.6
            crossfeed_decay = 0.9
            
        self.parental_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.parental_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.parental_crossfeed_decay = np.clip(crossfeed_decay, 0, 1)

    def set_prompt1(self, prompt: str):
        r"""
        Sets the first prompt (for the first keyframe) including text embeddings.
        Args:
            prompt: str
                ABC trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt1 = prompt
        self.text_embedding1 = self.get_text_embeddings(self.prompt1)

    def set_prompt2(self, prompt: str):
        r"""
        Sets the second prompt (for the second keyframe) including text embeddings.
        Args:
            prompt: str
                XYZ trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt2 = prompt
        self.text_embedding2 = self.get_text_embeddings(self.prompt2)

    def set_image1(self, image: Image):
        r"""
        Sets the first image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image1_lowres = image

    def set_image2(self, image: Image):
        r"""
        Sets the second image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image2_lowres = image
        
    def set_num_inference_steps(self, num_inference_steps=None):
        if self.dh.is_sdxl_turbo:
            if num_inference_steps is None:
                num_inference_steps = 4
        else:
            if num_inference_steps is None:
                num_inference_steps = 30
            
        self.num_inference_steps = num_inference_steps
        self.dh.set_num_inference_steps(num_inference_steps)
        
    def set_branching(self, depth_strength=None, t_compute_max_allowed=None, nmb_max_branches=None):
        """
        Sets the branching structure of the blending tree. Default arguments depend on pipe!
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed:
                Either provide t_compute_max_allowed or nmb_max_branches.
                The maximum time allowed for computation. Higher values give better results but take longer.
            nmb_max_branches: int
                Either provide t_compute_max_allowed or nmb_max_branches. The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
        """
        if self.dh.is_sdxl_turbo:
            assert t_compute_max_allowed is None, "time-based branching not supported for SDXL Turbo"
            if depth_strength is not None:
                idx_inject = int(round(self.num_inference_steps*depth_strength))
            else:
                idx_inject = 2
            if nmb_max_branches is None:
                nmb_max_branches = 10
                
            self.list_idx_injection = [idx_inject]
            self.list_nmb_stems = [nmb_max_branches]
            
        else:
            if depth_strength is None:
                depth_strength = 0.5
            if t_compute_max_allowed is None and nmb_max_branches is None:
                t_compute_max_allowed = 20
            elif t_compute_max_allowed is not None and nmb_max_branches is not None:
                raise ValueErorr("Either specify t_compute_max_allowed or nmb_max_branches")
            
            self.list_idx_injection, self.list_nmb_stems = self.get_time_based_branching(depth_strength, t_compute_max_allowed, nmb_max_branches)    

    def run_transition(
            self,
            recycle_img1: Optional[bool] = False,
            recycle_img2: Optional[bool] = False,
            fixed_seeds: Optional[List[int]] = None):
        r"""
        Function for computing transitions.
        Returns a list of transition images using spherical latent blending.
        Args:
            recycle_img1: Optional[bool]:
                Don't recompute the latents for the first keyframe (purely prompt1). Saves compute.
            recycle_img2: Optional[bool]:
                Don't recompute the latents for the second keyframe (purely prompt2). Saves compute.
            num_inference_steps:
                Number of diffusion steps. Higher values will take more compute time.

            fixed_seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
        """

        # Sanity checks first
        assert self.text_embedding1 is not None, 'Set the first text embedding with .set_prompt1(...) before'
        assert self.text_embedding2 is not None, 'Set the second text embedding with .set_prompt2(...) before'
        

        # Random seeds
        if fixed_seeds is not None:
            if fixed_seeds == 'randomize':
                fixed_seeds = list(np.random.randint(0, 1000000, 2).astype(np.int32))
            else:
                assert len(fixed_seeds) == 2, "Supply a list with len = 2"

            self.seed1 = fixed_seeds[0]
            self.seed2 = fixed_seeds[1]

        
        # Compute / Recycle first image
        if not recycle_img1 or len(self.tree_latents[0]) != self.num_inference_steps:
            list_latents1 = self.compute_latents1()
        else:
            list_latents1 = self.tree_latents[0]

        # Compute / Recycle first image
        if not recycle_img2 or len(self.tree_latents[-1]) != self.num_inference_steps:
            list_latents2 = self.compute_latents2()
        else:
            list_latents2 = self.tree_latents[-1]

        # Reset the tree, injecting the edge latents1/2 we just generated/recycled
        self.tree_latents = [list_latents1, list_latents2]
        self.tree_fracts = [0.0, 1.0]
        self.tree_final_imgs = [self.dh.latent2image((self.tree_latents[0][-1])), self.dh.latent2image((self.tree_latents[-1][-1]))]
        self.tree_idx_injection = [0, 0]
        self.tree_similarities = [self.get_tree_similarities]


        # Run iteratively, starting with the longest trajectory.
        # Always inserting new branches where they are needed most according to image similarity
        for s_idx in tqdm(range(len(self.list_idx_injection))):
            nmb_stems = self.list_nmb_stems[s_idx]
            idx_injection = self.list_idx_injection[s_idx]

            for i in range(nmb_stems):
                fract_mixing, b_parent1, b_parent2 = self.get_mixing_parameters(idx_injection)
                self.set_guidance_mid_dampening(fract_mixing)
                list_latents = self.compute_latents_mix(fract_mixing, b_parent1, b_parent2, idx_injection)
                self.insert_into_tree(fract_mixing, idx_injection, list_latents)
                # print(f"fract_mixing: {fract_mixing} idx_injection {idx_injection} bp1 {b_parent1} bp2 {b_parent2}")

        return self.tree_final_imgs
    

        

    def compute_latents1(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the first image
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        print("starting compute_latents1")
        list_conditionings = self.get_mixed_conditioning(0)
        t0 = time.time()
        latents_start = self.get_noise(self.seed1)
        list_latents1 = self.run_diffusion(
            list_conditionings,
            latents_start=latents_start,
            idx_start=0)
        t1 = time.time()
        self.dt_unet_step = (t1 - t0) / self.num_inference_steps
        self.tree_latents[0] = list_latents1
        if return_image:
            return self.dh.latent2image(list_latents1[-1])
        else:
            return list_latents1

    def compute_latents2(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the last image, which may be affected by the first image's trajectory.
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        print("starting compute_latents2")
        list_conditionings = self.get_mixed_conditioning(1)
        latents_start = self.get_noise(self.seed2)
        # Influence from branch1
        if self.branch1_crossfeed_power > 0.0:
            # Set up the mixing_coeffs
            idx_mixing_stop = int(round(self.num_inference_steps * self.branch1_crossfeed_range))
            mixing_coeffs = list(np.linspace(self.branch1_crossfeed_power, self.branch1_crossfeed_power * self.branch1_crossfeed_decay, idx_mixing_stop))
            mixing_coeffs.extend((self.num_inference_steps - idx_mixing_stop) * [0])
            list_latents_mixing = self.tree_latents[0]
            list_latents2 = self.run_diffusion(
                list_conditionings,
                latents_start=latents_start,
                idx_start=0,
                list_latents_mixing=list_latents_mixing,
                mixing_coeffs=mixing_coeffs)
        else:
            list_latents2 = self.run_diffusion(list_conditionings, latents_start)
        self.tree_latents[-1] = list_latents2

        if return_image:
            return self.dh.latent2image(list_latents2[-1])
        else:
            return list_latents2

    def compute_latents_mix(self, fract_mixing, b_parent1, b_parent2, idx_injection):
        r"""
        Runs a diffusion trajectory, using the latents from the respective parents
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            b_parent1: int
                index of parent1 to be used
            b_parent2: int
                index of parent2 to be used
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        list_conditionings = self.get_mixed_conditioning(fract_mixing)
        fract_mixing_parental = (fract_mixing - self.tree_fracts[b_parent1]) / (self.tree_fracts[b_parent2] - self.tree_fracts[b_parent1])
        # idx_reversed = self.num_inference_steps - idx_injection

        list_latents_parental_mix = []
        for i in range(self.num_inference_steps):
            latents_p1 = self.tree_latents[b_parent1][i]
            latents_p2 = self.tree_latents[b_parent2][i]
            if latents_p1 is None or latents_p2 is None:
                latents_parental = None
            else:
                latents_parental = interpolate_spherical(latents_p1, latents_p2, fract_mixing_parental)
            list_latents_parental_mix.append(latents_parental)

        idx_mixing_stop = int(round(self.num_inference_steps * self.parental_crossfeed_range))
        mixing_coeffs = idx_injection * [self.parental_crossfeed_power]
        nmb_mixing = idx_mixing_stop - idx_injection
        if nmb_mixing > 0:
            mixing_coeffs.extend(list(np.linspace(self.parental_crossfeed_power, self.parental_crossfeed_power * self.parental_crossfeed_decay, nmb_mixing)))
        mixing_coeffs.extend((self.num_inference_steps - len(mixing_coeffs)) * [0])
        latents_start = list_latents_parental_mix[idx_injection - 1]
        list_latents = self.run_diffusion(
            list_conditionings,
            latents_start=latents_start,
            idx_start=idx_injection,
            list_latents_mixing=list_latents_parental_mix,
            mixing_coeffs=mixing_coeffs)
        return list_latents

    def get_time_based_branching(self, depth_strength, t_compute_max_allowed=None, nmb_max_branches=None):
        r"""
        Sets up the branching scheme dependent on the time that is granted for compute.
        The scheme uses an estimation derived from the first image's computation speed.
        Either provide t_compute_max_allowed or nmb_max_branches
        Args:
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed: float
                The maximum time allowed for computation. Higher values give better results
                but take longer. Use this if you want to fix your waiting time for the results.
            nmb_max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
        """
        idx_injection_base = int(np.floor(self.num_inference_steps * depth_strength))
        
        steps = int(np.ceil(self.num_inference_steps/10))
        list_idx_injection = np.arange(idx_injection_base, self.num_inference_steps, steps)
        list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
        t_compute = 0

        if nmb_max_branches is None:
            assert t_compute_max_allowed is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "t_compute_max_allowed"
        elif t_compute_max_allowed is None:
            assert nmb_max_branches is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "nmb_max_branches"
            nmb_max_branches -= 2  # Discounting the outer frames
        else:
            raise ValueError("Either specify t_compute_max_allowed or nmb_max_branches")
        stop_criterion_reached = False
        is_first_iteration = True
        while not stop_criterion_reached:
            list_compute_steps = self.num_inference_steps - list_idx_injection
            list_compute_steps *= list_nmb_stems
            t_compute = np.sum(list_compute_steps) * self.dt_unet_step + self.dt_vae * np.sum(list_nmb_stems)
            t_compute += 2 * (self.num_inference_steps * self.dt_unet_step + self.dt_vae) # outer branches
            increase_done = False
            for s_idx in range(len(list_nmb_stems) - 1):
                if list_nmb_stems[s_idx + 1] / list_nmb_stems[s_idx] >= 1:
                    list_nmb_stems[s_idx] += 1
                    increase_done = True
                    break
            if not increase_done:
                list_nmb_stems[-1] += 1

            if stop_criterion == "t_compute_max_allowed" and t_compute > t_compute_max_allowed:
                stop_criterion_reached = True
            elif stop_criterion == "nmb_max_branches" and np.sum(list_nmb_stems) >= nmb_max_branches:
                stop_criterion_reached = True
                if is_first_iteration:
                    # Need to undersample.
                    list_idx_injection = np.linspace(list_idx_injection[0], list_idx_injection[-1], nmb_max_branches).astype(np.int32)
                    list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
            else:
                is_first_iteration = False

            # print(f"t_compute {t_compute} list_nmb_stems {list_nmb_stems}")
        return list_idx_injection, list_nmb_stems

    def get_mixing_parameters(self, idx_injection):
        r"""
        Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place
        where the metric is maximal.
        Args:
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        # get_lpips_similarity
        similarities = self.tree_similarities
        # similarities = self.get_tree_similarities()
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1 + 1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]
        fract_mixing = (fract_closest1 + fract_closest2) / 2

        # Ensure that the parents are indeed older
        b_parent1 = b_closest1
        while True:
            if self.tree_idx_injection[b_parent1] < idx_injection:
                break
            else:
                b_parent1 -= 1
        b_parent2 = b_closest2
        while True:
            if self.tree_idx_injection[b_parent2] < idx_injection:
                break
            else:
                b_parent2 += 1
        return fract_mixing, b_parent1, b_parent2

    def insert_into_tree(self, fract_mixing, idx_injection, list_latents):
        r"""
        Inserts all necessary parameters into the trajectory tree.
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
            list_latents: list
                list of the latents to be inserted
        """
        img_insert = self.dh.latent2image(list_latents[-1])
        
        b_parent1, b_parent2 = self.get_closest_idx(fract_mixing)
        left_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent1])
        right_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent2])
        idx_insert = b_parent1 + 1
        self.tree_latents.insert(idx_insert, list_latents)
        self.tree_final_imgs.insert(idx_insert, img_insert)
        self.tree_fracts.insert(idx_insert, fract_mixing)
        self.tree_idx_injection.insert(idx_insert, idx_injection)
        
        # update similarities
        self.tree_similarities[b_parent1] = left_sim
        self.tree_similarities.insert(idx_insert, right_sim)
        

    def get_noise(self, seed):
        r"""
        Helper function to get noise given seed.
        Args:
            seed: int
        """
        return self.dh.get_noise(seed)

    @torch.no_grad()
    def run_diffusion(
            self,
            list_conditionings,
            latents_start: torch.FloatTensor = None,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):
        r"""
        Wrapper function for diffusion runners.
        Depending on the mode, the correct one will be executed.

        Args:
            list_conditionings: list
                List of all conditionings for the diffusion model.
            latents_start: torch.FloatTensor
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            list_latents_mixing: torch.FloatTensor
                List of latents (latent trajectories) that are used for mixing
            mixing_coeffs: float or list
                Coefficients, how strong each element of list_latents_mixing will be mixed in.
            return_image: Optional[bool]
                Optionally return image directly
        """

        # Ensure correct num_inference_steps in Holder
        self.dh.set_num_inference_steps(self.num_inference_steps)
        assert type(list_conditionings) is list, "list_conditionings need to be a list"

        text_embeddings = list_conditionings[0]
        return self.dh.run_diffusion_sd_xl(
            text_embeddings=text_embeddings,
            latents_start=latents_start,
            idx_start=idx_start,
            list_latents_mixing=list_latents_mixing,
            mixing_coeffs=mixing_coeffs,
            return_image=return_image)




    @torch.no_grad()
    def get_mixed_conditioning(self, fract_mixing):
        text_embeddings_mix = []
        for i in range(len(self.text_embedding1)):
            if self.text_embedding1[i] is None:
                mix = None
            else:
                mix = interpolate_linear(self.text_embedding1[i], self.text_embedding2[i], fract_mixing)
            text_embeddings_mix.append(mix)
        list_conditionings = [text_embeddings_mix]

        return list_conditionings

    @torch.no_grad()
    def get_text_embeddings(
            self,
            prompt: str):
        r"""
        Computes the text embeddings provided a string with a prompts.
        Adapted from stable diffusion repo
        Args:
            prompt: str
                ABC trending on artstation painted by Old Greg.
        """
        return self.dh.get_text_embedding(prompt)

    def write_imgs_transition(self, dp_img):
        r"""
        Writes the transition images into the folder dp_img.
        Requires run_transition to be completed.
        Args:
            dp_img: str
                Directory, into which the transition images, yaml file and latents are written.
        """
        imgs_transition = self.tree_final_imgs
        os.makedirs(dp_img, exist_ok=True)
        for i, img in enumerate(imgs_transition):
            img_leaf = Image.fromarray(img)
            img_leaf.save(os.path.join(dp_img, f"lowres_img_{str(i).zfill(4)}.jpg"))
        fp_yml = os.path.join(dp_img, "lowres.yaml")

    def write_movie_transition(self, fp_movie, duration_transition, fps=30):
        r"""
        Writes the transition movie to fp_movie, using the given duration and fps..
        The missing frames are linearly interpolated.
        Args:
            fp_movie: str
                file pointer to the final movie.
            duration_transition: float
                duration of the movie in seonds
            fps: int
                fps of the movie
        """

        # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
        imgs_transition_ext = fill_up_frames_linear_interpolation(self.tree_final_imgs, duration_transition, fps)

        # Save as MP4
        if os.path.isfile(fp_movie):
            os.remove(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps, shape_hw=[self.dh.height_img, self.dh.width_img])
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()


    def get_state_dict(self):
        state_dict = {}
        grab_vars = ['prompt1', 'prompt2', 'seed1', 'seed2', 'height', 'width',
                     'num_inference_steps', 'depth_strength', 'guidance_scale',
                     'guidance_scale_mid_damper', 'mid_compression_scaler', 'negative_prompt',
                     'branch1_crossfeed_power', 'branch1_crossfeed_range', 'branch1_crossfeed_decay'
                     'parental_crossfeed_power', 'parental_crossfeed_range', 'parental_crossfeed_decay']
        for v in grab_vars:
            if hasattr(self, v):
                if v == 'seed1' or v == 'seed2':
                    state_dict[v] = int(getattr(self, v))
                elif v == 'guidance_scale':
                    state_dict[v] = float(getattr(self, v))

                else:
                    try:
                        state_dict[v] = getattr(self, v)
                    except Exception:
                        pass
        return state_dict


    def swap_forward(self):
        r"""
        Moves over keyframe two -> keyframe one. Useful for making a sequence of transitions
        as in run_multi_transition()
        """
        # Move over all latents
        self.tree_latents[0] = self.tree_latents[-1]
        # Move over prompts and text embeddings
        self.prompt1 = self.prompt2
        self.text_embedding1 = self.text_embedding2
        # Final cleanup for extra sanity
        self.tree_final_imgs = []

    def get_lpips_similarity(self, imgA, imgB):
        r"""
        Computes the image similarity between two images imgA and imgB.
        Used to determine the optimal point of insertion to create smooth transitions.
        High values indicate low similarity.
        """
        tensorA = torch.from_numpy(np.asarray(imgA)).float().cuda(self.device)
        tensorA = 2 * tensorA / 255.0 - 1
        tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)
        tensorB = torch.from_numpy(np.asarray(imgB)).float().cuda(self.device)
        tensorB = 2 * tensorB / 255.0 - 1
        tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
        lploss = self.lpips(tensorA, tensorB)
        lploss = float(lploss[0][0][0][0])
        return lploss

    def get_tree_similarities(self):
        similarities = []
        for i in range(len(self.tree_final_imgs) - 1):
            similarities.append(self.get_lpips_similarity(self.tree_final_imgs[i], self.tree_final_imgs[i + 1]))
        return similarities

    # Auxiliary functions
    def get_closest_idx(
            self,
            fract_mixing: float):
        r"""
        Helper function to retrieve the parents for any given mixing.
        Example: fract_mixing = 0.4 and self.tree_fracts = [0, 0.3, 0.6, 1.0]
        Will return the two closest values here, i.e. [1, 2]
        """

        pdist = fract_mixing - np.asarray(self.tree_fracts)
        pdist_pos = pdist.copy()
        pdist_pos[pdist_pos < 0] = np.inf
        b_parent1 = np.argmin(pdist_pos)
        pdist_neg = -pdist.copy()
        pdist_neg[pdist_neg <= 0] = np.inf
        b_parent2 = np.argmin(pdist_neg)

        if b_parent1 > b_parent2:
            tmp = b_parent2
            b_parent2 = b_parent1
            b_parent1 = tmp

        return b_parent1, b_parent2

#%%
if __name__ == "__main__":
    
    # %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
    from diffusers_holder import DiffusersHolder
    from diffusers import DiffusionPipeline
    from diffusers import AutoencoderTiny
    # pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_model_name_or_path = "stabilityai/sdxl-turbo"
    pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    
    
    # pipe.to("mps")
    pipe.to("cuda")
    
    # pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    # pipe.vae = pipe.vae.cuda()

    dh = DiffusersHolder(pipe)
    
    xxx
    # %% Next let's set up all parameters
    prompt1 = "photo of underwater landscape, fish, und the sea, incredible detail, high resolution"
    prompt2 = "rendering of an alien planet, strange plants, strange creatures, surreal"
    negative_prompt = "blurry, ugly, pale"  # Optional

    duration_transition = 12  # In seconds

    # Spawn latent blending
    be = BlendingEngine(dh)
    be.set_prompt1(prompt1)
    be.set_prompt2(prompt2)
    be.set_negative_prompt(negative_prompt)

    # Run latent blending
    t0 = time.time()
    be.run_transition(fixed_seeds=[420, 421])
    dt = time.time() - t0
    print(f"dt = {dt}")

    # Save movie
    fp_movie = f'test.mp4'
    be.write_movie_transition(fp_movie, duration_transition)
    



