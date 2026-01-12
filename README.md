# Video to 3D Gaussian Splatting

This repository provides an end-to-end pipeline for generating 3D Gaussian Splatting models from videos. It automates the process of frame extraction, Structure from Motion (SfM), depth estimation, and 3D Gaussian Splatting training.

## Overview

The pipeline consists of the following steps:
1. Video frame extraction and selection
2. Structure from Motion (SfM) using COLMAP
3. Depth estimation using Depth Anything V2
4. Depth scale estimation
5. 3D Gaussian Splatting training

## Installation

### Prerequisites
- Docker
- NVIDIA GPU with CUDA support
- WSL2 (if using Windows) or Ubuntu 22.04

### Setup

1. Clone this repository:
   ```bash
   git clone --recursive https://github.com/yourusername/video_to_3dgs.git
   cd video_to_3dgs
   ```

2. Download Depth Anything V2 model:
   - Download the Depth-Anything-V2-Large model from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
   - Place it in: `submodules/DepthAnythingV2_docker/checkpoints/depth_anything_v2_vitl.pth`

3. Build and start the Docker container:
   ```bash
   ./start.sh [--dataset=/path/to/dataset_gs]
   ```
   This script will:
   - Automatically build the Docker container with all required dependencies
   - Detect your environment (WSL1, WSL2, or native Linux) and configure X11 forwarding
   - Use the default dataset path (`../datasets_gs`) or a custom path specified with `--dataset`

## Usage

### Dataset Preparation

1. Create a dataset directory:
   ```
   datasets_gs/your_scene_name/your_video.mp4
   ```
   The video file can have any name with a `.mp4` extension.

   By default, the pipeline looks for datasets in the `../datasets_gs` directory relative to the project root. You can specify a different location when starting the container:
   ```bash
   ./start.sh --dataset=/path/to/your/datasets
   ```

### Running the Pipeline

To process a single video:
```bash
python do_all.py -s /v2gs/datasets_gs/your_scene_name -n 300
```

Parameters:
- `-s, --source_path`: Path to the directory containing your video
- `-n, --max_number_of_frames`: Maximum number of frames to extract (default: 400)
- `-c, --clean`: Clean existing processed data and start fresh
- `-m, --minimal`: Use minimal frame selection after final reconstruction
- `-f, --full`: Use all frames for reconstruction and longer training
- `--full_res`: Extract final frames at full resolution
- `-a, --all`: Process all subdirectories in the source path

### Output

The pipeline generates the following directories:
- `images/`: Extracted video frames
- `sparse/`: COLMAP sparse reconstruction
- `d_images/`: Depth maps estimated by Depth Anything V2
- `model/`: Trained 3D Gaussian Splatting model

## Viewing Results

Use the webtool: [supersplat](https://superspl.at/editor)
Simply drag and drop the .ply file in the browser

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses the following open-source projects:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [COLMAP](https://github.com/colmap/colmap)