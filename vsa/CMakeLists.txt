cmake_minimum_required(VERSION 3.10)
project(CUDA_TEST LANGUAGES CUDA CXX)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_FLAGS "-g -G")
set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++11" "-O0")



set(CONDA_PREFIX "/data/users/jiangwentao/Apps/miniconda3/envs/cuda")

set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}/lib/python3.9/site-packages/torch/share/cmake")
set(Python_EXECUTABLE "${CONDA_PREFIX}/bin/python")
set(Python_INCLUDE_DIRS "${CONDA_PREFIX}/include/python3.9")
set(Python_LIBRARIES "${CONDA_PREFIX}/lib")

message(${CMAKE_PREFIX_PATH})

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
include_directories(include)

set(KERNEL_SRC)
file(GLOB KERNEL_SRC kernel/*.cu)

add_executable(test test.cpp VSAttn.cpp ${KERNEL_SRC})

target_include_directories(test PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(test ${TORCH_LIBRARIES} ${Python_LIBRARIES})