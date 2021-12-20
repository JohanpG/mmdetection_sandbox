#!/bin/sh
export LD_LIBRARY_PATH="/mnt/c/'Program Files'/'NVIDIA GPU Computing Toolkit'/CUDA/v11.3/lib/x64:$LD_LIBRARY_PATH"
export PATH="/mnt/c/'Program Files'/'NVIDIA GPU Computing Toolkit'/CUDA/v11.3/bin:$PATH"
export CUDA_HOME=/mnt/c/'Program Files'/'NVIDIA GPU Computing Toolkit'/CUDA/v11.3
export DOCKER_HOST='tcp://localhost:2375'
export ONNX_BACKEND=MMCVTensorRT
