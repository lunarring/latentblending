# Copyright 2022 Lunar Ring. All rights reserved.
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

import subprocess
import os
import numpy as np
from tqdm import tqdm
import cv2
from typing import Callable, List, Optional, Union
import ffmpeg # pip install ffmpeg-python. if error with broken pipe: conda update ffmpeg

#%%
            
class MovieSaver():
    def __init__(
            self, 
            fp_out: str,  
            fps: int = 24, 
            crf: int = 24,
            codec: str = 'libx264',
            preset: str ='fast',
            pix_fmt: str = 'yuv420p', 
            silent_ffmpeg: bool = True
        ):
        r"""
        Initializes movie saver class - a human friendly ffmpeg wrapper.
        After you init the class, you can dump numpy arrays x into moviesaver.write_frame(x). 
        Don't forget toi finalize movie file with moviesaver.finalize().
        Args:
            fp_out: str
                Output file name. If it already exists, it will be deleted.
            fps: int
                Frames per second.
            crf: int
                ffmpeg doc: the range of the CRF scale is 0–51, where 0 is lossless
                (for 8 bit only, for 10 bit use -qp 0), 23 is the default, and 51 is worst quality possible. 
                A lower value generally leads to higher quality, and a subjectively sane range is 17–28. 
                Consider 17 or 18 to be visually lossless or nearly so; 
                it should look the same or nearly the same as the input but it isn't technically lossless. 
                The range is exponential, so increasing the CRF value +6 results in 
                roughly half the bitrate / file size, while -6 leads to roughly twice the bitrate.  
            codec: int
                Number of diffusion steps. Larger values will take more compute time.
            preset: str
                Choose between ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
                ffmpeg doc: A preset is a collection of options that will provide a certain encoding speed 
                to compression ratio. A slower preset will provide better compression 
                (compression is quality per filesize). 
                This means that, for example, if you target a certain file size or constant bit rate, 
                you will achieve better quality with a slower preset. Similarly, for constant quality encoding,
                you will simply save bitrate by choosing a slower preset. 
            pix_fmt: str
                Pixel format. Run 'ffmpeg -pix_fmts' in your shell to see all options.
            silent_ffmpeg: bool
                Surpress the output from ffmpeg.
        """
        
        self.fp_out = fp_out
        self.fps = fps
        self.crf = crf
        self.pix_fmt = pix_fmt
        self.codec = codec
        self.preset = preset
        self.silent_ffmpeg = silent_ffmpeg
        
        if os.path.isfile(fp_out):
            os.remove(fp_out)
        
        self.init_done = False
        self.nmb_frames = 0
        self.shape_hw = [-1, -1]
        
        print(f"MovieSaver initialized. fps={fps} crf={crf} pix_fmt={pix_fmt} codec={codec} preset={preset}")
        
        
    def initialize(self):
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.shape_hw[1], self.shape_hw[0]), framerate=self.fps)
            .output(self.fp_out, crf=self.crf, pix_fmt=self.pix_fmt, c=self.codec, preset=self.preset)
            .overwrite_output()
            .compile()
        )
        if self.silent_ffmpeg:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.init_done = True
        print(f"First frame initialization done. Movie shape: {self.shape_hw}")
    
        
    def write_frame(self, out_frame: np.ndarray):
        r"""
        Function to dump a numpy array as frame of a movie.
        Args:
            out_frame: np.ndarray
                Numpy array, in np.uint8 format. Convert with np.astype(x, np.uint8).
                Dim 0: y
                Dim 1: x
                Dim 2: RGB
        """
        
        assert out_frame.dtype == np.uint8, "Convert to np.uint8 before"
        assert len(out_frame.shape) == 3, "out_frame needs to be three dimensional, Y X C"
        assert out_frame.shape[2] == 3, f"need three color channels, but you provided {out_frame.shape[2]}."
        
        if not self.init_done:
            self.shape_hw = out_frame.shape
            self.initialize()
            
        assert self.shape_hw == out_frame.shape, "You cannot change the image size after init. Initialized with {self.shape_hw}, out_frame {out_frame.shape}"

        # write frame        
        self.ffmpg_process.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

        self.nmb_frames += 1
    
    
    def finalize(self):
        r"""
        Call this function to finalize the movie. If you forget to call it your movie will be garbage.
        """
        self.ffmpg_process.stdin.close()
        self.ffmpg_process.wait()
        duration = int(self.nmb_frames / self.fps)
        print(f"Movie saved, {duration}s playtime, watch here: \n{self.fp_out}")



def concatenate_movies(fp_final: str, list_fp_movies: List[str]):
    r"""
    Concatenate multiple movie segments into one long movie, using ffmpeg.

    Parameters
    ----------
    fp_final : str
        Full path of the final movie file. Should end with .mp4
    list_fp_movies : list[str]
        List of full paths of movie segments. 
    """
    assert fp_final.endswith(".mp4"), "fp_final should end with .mp4"
    for fp in list_fp_movies:
        assert os.path.isfile(fp), f"Input movie does not exist: {fp}"
        assert os.path.getsize(fp) > 100, f"Input movie seems empty: {fp}"
    
    if os.path.isfile(fp_final):
        os.remove(fp_final)

    # make a list for ffmpeg
    list_concat = []
    for fp_part in list_fp_movies:
        list_concat.append(f"""file '{fp_part}'""")
    
    # save this list
    fp_list = fp_final[:-3] + "txt"
    with open(fp_list, "w") as fa:
        for item in list_concat:
            fa.write("%s\n" % item)
            
    dp_movie = os.path.split(fp_final)[0]
    cmd = f'ffmpeg -f concat -safe 0 -i {fp_list} -c copy {fp_final}'
    subprocess.call(cmd, shell=True, cwd=dp_movie)
    os.remove(fp_list)
    print(f"concatenate_movies: success! Watch here: \n{fp_final}")

            
class MovieReader():
    r"""
    Class to read in a movie.
    """
    def __init__(self, fp_movie):
        self.video_player_object = cv2.VideoCapture(fp_movie)
        self.nmb_frames = int(self.video_player_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_movie = int(self.video_player_object.get(cv2.CAP_PROP_FPS))
        self.shape = [100,100,3]
        self.shape_is_set = False
    
    def get_next_frame(self):
        success, image = self.video_player_object.read()
        if success:
            if not self.shape_is_set:
                self.shape_is_set = True
                self.shape = image.shape
            return image
        else:
            return np.zeros(self.shape)

#%%
if __name__ == "__main__": 
    ms = MovieSaver("/tmp/bubu.mp4", fps=fps)
    for img in list_imgs_interp:
        ms.write_frame(img)
    ms.finalize()
if False:
    fps=2
    list_fp_movies = []
    for k in range(4):
        fp_movie = f"/tmp/my_random_movie_{k}.mp4"
        list_fp_movies.append(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps)
        for fn in tqdm(range(30)):
            img = (np.random.rand(512, 1024, 3)*255).astype(np.uint8)
            ms.write_frame(img)
        ms.finalize()
    
    fp_final = "/tmp/my_concatenated_movie.mp4"
    concatenate_movies(fp_final, list_fp_movies)

