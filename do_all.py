#!/usr/bin/env python3
"""
Video to 3D Gaussian Splatting Pipeline

This script automates the end-to-end process of converting a video to a 3D Gaussian Splatting model.
The pipeline includes:
1. Video frame extraction
2. Structure from Motion (SfM) using COLMAP
3. Depth estimation using Depth Anything V2
4. Depth scale estimation
5. 3D Gaussian Splatting training

Author: Original repository author
"""

import os
import sys
import shutil
import time
from argparse import ArgumentParser

# Get the current script directory
CURR_PATH = os.path.dirname(os.path.abspath(__file__))

def get_video_length(filename):
    """
    Get the duration, frame count, and FPS of a video file.
    
    Args:
        filename (str): Path to the video file
        
    Returns:
        tuple: (duration in seconds, total frame count, frames per second)
    """
    try:
        import cv2
        video = cv2.VideoCapture(filename)
        
        if not video.isOpened():
            print(f"Error: Could not open video file {filename}")
            return None, None, None
            
        video_fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count / video_fps)
        
        video.release()
        return duration, frame_count, video_fps
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return None, None, None

def run_command(command, error_message):
    """
    Run a shell command and handle errors.
    
    Args:
        command (str): Command to execute
        error_message (str): Error message to display if command fails
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"Running: {command}")
    exit_code = os.system(command)
    
    if exit_code != 0:
        print(f"ERROR: {error_message}")
        print(f"Command failed with exit code {exit_code}")
        return False
    return True


def do_one(source_p, n_frames, clean=False, minimal=False, full=False, full_res=False):
    """
    Process a single video through the entire pipeline.
    
    Args:
        source_p (str): Path to the directory containing the video
        n_frames (int): Maximum number of frames to extract
        clean (bool): Whether to clean existing processed data
        minimal (bool): Use minimal frame selection
        full (bool): Use all frames and longer training
        full_res (bool): Extract final frames at full resolution
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    start_time = time.time()
    
    # Find video file in the source directory
    files_n = os.listdir(source_p)
    video_n = None
    for f in files_n:
        if f.lower().endswith(('.mp4', '.mov', '.avi')):
            video_n = f
            break
    
    if video_n is None and "input" not in files_n:
        print(f"Error: No video file found in {source_p}")
        return False
    
    # Define output directories
    images_p = os.path.join(source_p, 'images')
    sparse_p = os.path.join(source_p, 'sparse')
    model_p = os.path.join(source_p, 'model')
    depths_p = os.path.join(source_p, 'd_images')
    
    print(f"\n{'='*80}")
    print(f"Processing: {source_p}")
    if video_n:
        print(f"Video: {video_n}")
        video_path = os.path.join(source_p, video_n)
        duration, frame_count, fps = get_video_length(video_path)
        if duration:
            print(f"Video duration: {duration}s, {frame_count} frames, {fps} FPS")
    print(f"{'='*80}\n")
    
    # Step 1: Extract frames and perform SfM
    if not (os.path.isdir(images_p) and os.path.isdir(sparse_p)) or clean:
        print("\n--- Step 1: Frame extraction and Structure from Motion ---")
        sfm_command = f"python preprocess/main_video_process.py -s {source_p} -n {n_frames} --robust"
        
        if clean:
            sfm_command += " -c"
        if minimal:
            sfm_command += " -m"
        if full:
            sfm_command += " -f"
        if full_res:
            sfm_command += " --full_res"
            
        if not run_command(sfm_command, "Failed during frame extraction or SfM"):
            return False
    
    sfm_time = time.time()
    
    # # Step 2: Estimate depth maps
    # if not os.path.isdir(depths_p):
    #     print("\n--- Step 2: Depth estimation ---")
    #     depth_command = f"cd {CURR_PATH}/submodules/DepthAnythingV2_docker/ && python run.py --encoder vitl --pred-only --grayscale --img-path {images_p} --outdir {depths_p}"
        
    #     if not run_command(depth_command, "Failed during depth estimation"):
    #         return False
    
    # # Step 3: Estimate depth scale
    # print("\n--- Step 3: Depth scale estimation ---")
    # scale_cmd = f"python {CURR_PATH}/submodules/gaussian-splatting/utils/make_depth_scale.py --base_dir {source_p} --depths_dir {depths_p}"
    
    # if not run_command(scale_cmd, "Failed during depth scale estimation"):
    #     return False
    
    depth_time = time.time()
    
    # Step 4: Train 3D Gaussian Splatting
    print("\n--- Step 4: 3D Gaussian Splatting training ---")
    train_cmd = (
        "conda run -n gsplat --no-capture-output CUDA_VISIBLE_DEVICES=0 python /v2gs/submodules/gsplat/examples/simple_trainer.py mcmc"
        f" --data_dir {source_p}  --data_factor 1 --result_dir {model_p}"
        f" --save-ply --pose-opt --depth-loss --disable-viewer --visible-adam --app-opt"
    )


    if not run_command(train_cmd, "Failed during 3D Gaussian Splatting training"):
        return False
    
    end_time = time.time()
    
    # Print timing information
    print(f"\n{'='*80}")
    print(f"Processing completed successfully!")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"  - SfM time: {sfm_time - start_time:.2f}s")
    print(f"  - Depth estimation time: {depth_time - sfm_time:.2f}s")
    print(f"  - 3D Gaussian Splatting time: {end_time - depth_time:.2f}s")
    print(f"{'='*80}\n")
    
    print(f"Results saved to: {model_p}")    
    return True

def main(args):
    """
    Main function to process videos based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    source_p = args.source_path
    n_frames = args.max_number_of_frames
    clean = args.clean
    minimal = args.minimal
    full = args.full
    full_res = args.full_res
    
    if not os.path.exists(source_p):
        print(f"Error: Source path {source_p} does not exist")
        return
    
    if not args.all:
        # Process a single video
        do_one(source_p, n_frames, clean=clean, minimal=minimal, full=full, full_res=full_res)
    else:
        # Process all subdirectories
        print(f"Processing all subdirectories in {source_p}")
        dirs = os.listdir(source_p)
        successful = 0
        failed = 0
        
        for d in dirs:
            tmp = os.path.join(source_p, d)
            if not os.path.isdir(tmp):
                continue
                
            print(f"\nProcessing directory: {d}")
            result = do_one(tmp, n_frames, clean=clean, minimal=minimal, full=full, full_res=full_res)
            
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f"\nProcessing complete: {successful} successful, {failed} failed")

if __name__ == '__main__':
    parser = ArgumentParser(description="Video to 3D Gaussian Splatting Pipeline")
    parser.add_argument("--source_path", "-s", required=True, type=str,
                        help="Path to the directory containing the video. Use aboslute path! /v2gs/dataset_gs/your_folder")
    parser.add_argument("--max_number_of_frames", "-n", default=400, type=int,
                        help="Maximum number of frames to extract (default: 400)")
    parser.add_argument("--clean", "-c", action='store_true',
                        help="Clean existing processed data and start fresh")
    parser.add_argument("--minimal", "-m", action='store_true',
                        help="Use minimal frame selection after final reconstruction")
    parser.add_argument("--full", "-f", action='store_true',
                        help="Use all frames for reconstruction and longer training")
    parser.add_argument("--full_res", action='store_true',
                        help="Extract final frames at full resolution")
    parser.add_argument("--all", "-a", action='store_true',
                        help="Process all subdirectories in the source path")
    
    args = parser.parse_args()
    main(args)
