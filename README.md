# Introduction
This is a sand box repository to use the MMDetection library which is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

# Environment Details
- Windows 11 WIP build 22504.1010
- **WSL2** 
    - Kernel 5.10.60.1
    - Ubuntu 20.04.3 LTS (Installed from windows store)
- **CUDA**
    - Toolkit 11.3
    - NvidiaDriver version 472.34
    - Nvidia CUDA Driver 11.4.150
- **Hardware**
    - Laptop: Lenovo X1 Xtreme
    - CPU:    Intel Core i7-9750H
    - GPU:    NVIDIA Geforce GTX 1650 Max-Q Design
    - Memory: 16 gb 


# Set up Environment
Several elements need to be taken care of on order:
- Windows WSL enable & Nvidia Drivers installation
- WSL Kernel update
- Ubuntu download

## Windows
- Follow the instructions to install the [WSL NVIDIA DRIVER](https://developer.nvidia.com/cuda/wsl) [MS Youtube Video](https://www.youtube.com/watch?v=PdxXlZJiuxA&list=LL&index=1&t=467s&ab_channel=MicrosoftDeveloper)
## Linux
- Setup CUDA Toolkit by following the [WSL NVIDIA GUIDE](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#rel-470-76) [MS Youtube Video](https://www.youtube.com/watch?v=PdxXlZJiuxA&list=LL&index=1&t=467s&ab_channel=MicrosoftDeveloper)
    - On step 4.2.6 (Using the WSL-Ubuntu Package) use these commands instead [WSL Cuda 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local):

    `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-3-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda`
- Update .bashrc (located under home/<user>) and add following lines

    `#CUDA LOCATION FROM WSL
    export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-11.3/bin:$PATH"
    export CUDA_HOME=/usr/local/cuda-11.3`
- run `source .bashrc`
- Install conda
    `sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    `bash Miniconda3-latest-Linux-x86_64.sh`


## MMDetection conda installation  
- Follow steps on the [set up guide](https://mmdetection.readthedocs.io/en/v2.19.0/get_started.html#installation)
- on step #2 use `conda install pytorch=1.10.0 cudatoolkit=11.3 torchvision -c pytorch`

## Export to tensor RT
## Export to ONNX
- pip install onnx onnxruntime-gpu
- Current version of mmdetection does not support onnxruntime-gpu>= 1.9.0 (But since we are using CUDA Toolkit 11.3 and that is supported on > 1.9.0) so a small tweak is needed:
    - Modify file '/home/johan/miniconda3/envs/openmmlab/lib/python3.7/site-packages/mmdet/core/export/model_wrappers.py"
        - Comment lines 115 y 123
        add 'sess = ort.InferenceSession(onnx_file, session_options, providers=providers)' on line 124

## Covert to TensorRT
- Donwload deb file https://developer.nvidia.com/nvidia-tensorrt-8x-download
- Follow [Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
- PIP Wheel
    - python3 -m pip install --upgrade setuptools pip
    - python3 -m pip install nvidia-pyindex
    - python3 -m pip install --upgrade nvidia-tensorrt

# Frequent used Commands
- sudo apt update
- sudo apt install gedit -y
- stat <foldername> 
- sudo chown -R <username> <folder>
- sudo apt-get --purge -y remove 'cuda*'
- nvcc --version
- wsl --update
- wsl -l -o
- wsl -l -v
- uname -a
- source .bashrc
- conda update conda
- conda env list
- conda create
- conda activate
- conda deactivate