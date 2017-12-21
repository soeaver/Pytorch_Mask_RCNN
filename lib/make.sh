#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling crop_and_resize kernels by nvcc..."
cd roi_align/src/cuda
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py

cd ../
echo "Compiling nms kernels by nvcc..."

cd nms/src/cuda
$CUDA_PATH/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py

