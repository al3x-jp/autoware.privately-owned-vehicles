cmake_minimum_required(VERSION 3.22)

PROJECT(onednn_scene_seg)
#SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

MESSAGE(STATUS "Linux")

## These should be variables (TODO)
set(USE_CUDNN OFF)
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)

## Find libTorch installation
find_package(Torch REQUIRED HINTS ${LIBTORCH_INSTALL_ROOT}/share/cmake/Torch/ QUIET)

## Find OpenCV installation
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)

# Temporary disable sycl 2020 deprecations warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -Wno-unknown-cuda-version -Wno-sycl-target")

# Use LLVM Sycl Compiler
SET(CMAKE_CXX_COMPILER "${DPCPP_HOME}/llvm/build/bin/clang++")

if(CUDA_BACKEND)
	# Build for CUDA BACK-END (requires an NVIDIA CARD)
	MESSAGE(STATUS "Building for CUDA back-end.")
	##SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda-11.8")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL2020_DISABLE_DEPRECATION_WmARNINGS -Wno-unknown-cuda-version -Wno-sycl-target -std=c++17 -O3 -fsycl -Xsycl-target-backend --cuda-gpu-arch=sm_80 -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda-11.8 -D__SYCL_INTERNAL_API")
else()
	# Build Native Platform
    MESSAGE(STATUS "Building for OpenCL back-end.")
	##SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fsycl -fno-sycl-instrument-device-code")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -Wno-sycl-target -std=c++17 -O3 -fsycl -fno-sycl-instrument-device-code -D__SYCL_INTERNAL_API -Wno-unused-command-line-argument")
endif()

SET(TARGET onednn_scene_seg)

include_directories(    ${ONEDNN_INSTALL}/include
                        ${TORCH_INCLUDE_DIRS}
                        ${OpenCV_INCLUDE_DIRS}
                    )

# Add source code to build
ADD_EXECUTABLE(${TARGET} 
        sycl_main.cpp
)

target_link_libraries(${TARGET} ${ONEDNN_INSTALL}/lib/libdnnl.so ${OpenCV_LIBS} ${TORCH_LIBRARIES} opencv_imgcodecs)