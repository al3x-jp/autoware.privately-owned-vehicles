# oneDNN_SceneSeg
oneDNN Implementation of SceneSeg Network

Network data extraction script based on/modified version of "prepareData.py" from here: "oneDNN/examples/network/prepareData.py"
Network construction based on/modified versions of "pointnet.cpp" from "oneDNN/examples/network/pointnet.cpp", and "cnn_inference_f32.cpp" from "oneDNN/examples/cnn_inference_f32.cpp"

## Extract Pre-trained Data from *.pth file
python3 extract_pretrained_data.py <path-to-*.pth-file>

This will create a directory called "./pretrained-data" where the pre-trained network data will be extracted and stored as *.bin files for use in the oneDNN application

## Build Instructions
export DPCPP_HOME=~/sycl_workspace
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH

mkdir build && cd build

cmake -DDPCPP_HOME=<path-to-sycl-workspace> -DLIBTORCH_INSTALL_ROOT=<path-to-libtorch-install> -DOPENCV_INSTALL_ROOT=<path-to-opencv-install> -DONEDNN_INSTALL=<path-to-onednn-install> -DUSE_CUDA_BACKEND=<{True | False}> ..

make

## Example Command Line
DNNL_VERBOSE=2 ONEAPI_DEVICE_SELECTOR=cuda:gpu SYCL_UR_TRACE=1 ./onednn_scene_seg ../pretrained-data <path-to-network-input-file.png>
DNNL_VERBOSE=2 ONEAPI_DEVICE_SELECTOR=openmcl:gpu SYCL_UR_TRACE=1 ./onednn_scene_seg ../pretrained-data <path-to-network-input-file.png>