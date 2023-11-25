# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("torch_onnx_deployment")

# 添加 onnxruntime 库
set(ONNXRUNTIME_ROOT "D:/CppLibraries/onnxruntime-win-x64-gpu-1.13.1")
list(APPEND CMAKE_MODULE_PATH "${ONNXRUNTIME_ROOT}/cmake")
find_package(ONNXRUNTIME REQUIRED)

# 添加 opencv 库
set(OpenCV_DIR  "D:/CppLibraries/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 包含子项目。
add_subdirectory ("onnx_inference")