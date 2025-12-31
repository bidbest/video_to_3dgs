## Docker

## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06
ENV DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Update Conda and use new solver
# RUN conda update -n base conda && \
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda init bash && exec bash

RUN pip install pycolmap==3.11.1 open3d==0.19.0

# Install colmap
RUN apt update && apt-get install -y \
    git \
    ffmpeg \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libomp-dev

WORKDIR /tmp/
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /tmp/colmap

RUN git checkout 682ea9ac4020a143047758739259b3ff04dabe8d &&\
    mkdir build && cd build &&\
    cmake .. -GNinja \
    -DCMAKE_CUDA_ARCHITECTURES=all-major \
    -DOPENMP_ENABLED=ON && \
    ninja &&\
    ninja install

# Install DepthAnything dependencies
COPY ./submodules/DepthAnythingV2_docker/requirements.txt /tmp/requirements.txt
WORKDIR /tmp/

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install gsplat
COPY ./environment_gsplat.yml ./environment_gsplat.yml
RUN conda env create --file environment_gsplat.yml

RUN conda run -n gsplat python -m pip install ninja numpy jaxtyping rich fsspec
COPY ./requirements_gsplat.txt ./requirements_gsplat.txt
RUN conda run -n gsplat python -m pip install -r ./requirements_gsplat.txt

RUN conda run -n gsplat python -m pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124

COPY ./submodules/gsplat/examples/requirements.txt ./requirements.txt
RUN conda run -n gsplat python -m pip install -r ./requirements.txt

COPY ./thirdparty/Depth-Anything-3 /tmp/thirdparty/Depth-Anything-3
WORKDIR /tmp/thirdparty/Depth-Anything-3

RUN pip install xformers torch\>=2 torchvision
RUN pip install -e .
RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

WORKDIR /v2gs

# This error occurs because there’s a conflict between the threading layer used
# by Intel MKL (Math Kernel Library) and the libgomp library, 
# which is typically used by OpenMP for parallel processing. 
# This often happens when libraries like NumPy or SciPy are used in combination
# with a multithreaded application (e.g., your Docker container or Python environment).
# Solution, set threading layer explicitly! (GNU or INTEL)
ENV MKL_THREADING_LAYER=GNU