"""
Module: video_processing
Contains functionality for video metadata extraction and frame extraction.
"""

import subprocess
import os
import json
from typing import List, Optional

class FFmpegWrapper:
    """
    Wrapper class for extracting video frames and metadata using FFmpeg.
    """
    def __init__(self, video_path: str, output_dir: str):
        """
        Initialize FFmpegWrapper by setting up directories, extracting metadata, and extracting frames.
        
        Parameters:
            video_path (str): Path to the input video.
            output_dir (str): Directory where extracted frames will be stored.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.tmp_path = os.path.join(os.path.dirname(video_path), "tmp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)
        self.fps, self.duration, self.width, self.height = self._get_video_metadata()
        self._extract_all_small_frames()
        self._get_frames_ids()
    
    def _get_video_metadata(self):
        """
        Retrieve FPS, duration, and resolution of the video using ffprobe.

        Returns:
            Tuple containing the frame rate, video duration, width and height.
        """
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration,width,height",
            "-of", "json", self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        stream = metadata['streams'][0]
        frame_rate = eval(stream['r_frame_rate'])
        duration = float(stream['duration'])
        width = int(stream['width'])
        height = int(stream['height'])
        return frame_rate, duration, width, height

    def _extract_all_small_frames(self):
        """
        Extract small frames (scaled to height 480) from the video if not already extracted.
        """
        if len(os.listdir(self.tmp_path)) == 0:
            extract_frames_cmd = f"ffmpeg -i {self.video_path} -pix_fmt rgb8 -q:v 4 -vf 'scale=-1:480' {self.tmp_path}/%08d.jpeg"
            exit_code = os.system(extract_frames_cmd)
            if exit_code != 0:
                print("error extracting frames")
                exit(exit_code)
        else:
            print("Frames already extracted")

    def _get_frames_ids(self):
        """
        Generate a sorted list of extracted frame filenames.
        """
        self.frames = sorted(os.listdir(self.tmp_path))

    def extract_specific_frames(self, frame_indices, full_res=False):
        """
        Extract specific frames from the video based on given frame indices.
        
        Parameters:
            frame_indices (list): List of frame indices to extract.
        """
        # Build select filter for ffmpeg command
        select_filter = "+".join([f"eq(n\\,{f})" for f in frame_indices])
        if full_res:
            scale_part = ""
        else:
            scale_part = ", scale=-1:960"

        cmd = f'ffmpeg -i {self.video_path} -vf "select={select_filter}{scale_part}" -vsync vfr -pix_fmt rgb8 -q:v 4 {self.output_dir}/%08d.jpeg'
        exit_code = os.system(cmd)
        if exit_code != 0:
            print("error extracting frames")
            exit(exit_code)

    def ind_to_frame_name(self, ind):
        """
        Convert a frame index to its corresponding filename.
        
        Parameters:
            ind (int): Frame index.
        
        Returns:
            str: Full path to the frame image.
        """
        name = "{:08d}.jpeg".format(ind)
        return os.path.join(self.tmp_path, name)

    def get_list_of_n_frames(self, n: int, start_frame: Optional[str] = None, end_frame: Optional[str] = None) -> List[str]:
        """
        Returns a list of n frame paths, equally distributed within the valid frame range.
        
        Parameters:
            n (int): Number of frames to select.
            start_frame (str, optional): Starting frame filename.
            end_frame (str, optional): Ending frame filename.
        
        Returns:
            List[str]: List of selected frame file paths.
        """
        if not self.frames or n <= 0:
            return []
        
        start_idx = self.frames.index(start_frame) if start_frame in self.frames else 0
        end_idx = self.frames.index(end_frame) if end_frame in self.frames else len(self.frames) - 1
        
        valid_frames = self.frames[start_idx:end_idx+1]
        total_frames = len(valid_frames)
        
        if n >= total_frames:
            print(f"Selected range has only {total_frames} frames. Returning all.")
            return [os.path.join(self.tmp_path, frame) for frame in valid_frames]
        
        step = total_frames / n
        indices = sorted(set(round(i * step) for i in range(n)))  # Ensure unique indices
        selected_frames = [valid_frames[i] for i in indices if i < total_frames]
        return [os.path.join(self.tmp_path, frame) for frame in selected_frames]
    
    def get_frames_between_pairs(self, peak_pairs: List[tuple], n: int) -> List[str]:
        """
        Returns a list of n frame paths between each specified pair of frames.
        
        Parameters:
            peak_pairs (List[tuple]): List of tuples containing pair of frame filenames.
            n (int): Number of frames to select between each pair.
        
        Returns:
            List[str]: List of selected frame file paths.
        """
        selected_frames = []
        for p1, p2 in peak_pairs:
            if p1 in self.frames and p2 in self.frames:
                selected_frames.extend(self.get_list_of_n_frames(n, start_frame=p1, end_frame=p2))
        return selected_frames
