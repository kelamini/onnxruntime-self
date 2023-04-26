project(capi_test)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.22)

set(ONNXRUNTIME_ROOTDIR /home/kelamini/workspace/onnxruntime)

set(OBJ demo.cpp face_keypoints.cpp)

# # tensorrt_provider_factory.h contains old APIs of the tensorrt execution provider
# include(CheckIncludeFileCXX)
# CHECK_INCLUDE_FILE_CXX(tensorrt_provider_factory.h HAVE_TENSORRT_PROVIDER_FACTORY_H)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
    ${ONNXRUNTIME_ROOTDIR}/onnxruntime
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED
             NO_MODULE
             PATHS /home/kelamini/workspace/libtorch
             NO_DEFAULT_PATH)

ADD_EXECUTABLE(capi_test  ${OBJ})

# if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
#   target_compile_definitions(capi_test PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
# endif()

target_link_libraries(capi_test onnxruntime)
target_link_libraries(capi_test ${OpenCV_LIBS})
target_link_libraries(capi_test ${TORCH_LIBRARIES})
