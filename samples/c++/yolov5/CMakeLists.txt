project(test_yolov5)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.22)

set(OBJ test_yolov5.cpp yolov5.cpp)

set(ONNXRUNTIME_ROOTDIR /home/kelamini/workspace/onnxruntime)

include_directories(
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
    ${ONNXRUNTIME_ROOTDIR}/onnxruntime
)

# 查找 opencv 库
find_package(OpenCV REQUIRED)

# 生成编译文件
ADD_EXECUTABLE(test_yolov5 ${OBJ})

# 链接库
target_link_libraries(test_yolov5 onnxruntime)
target_link_libraries(test_yolov5 ${OpenCV_LIBS})
