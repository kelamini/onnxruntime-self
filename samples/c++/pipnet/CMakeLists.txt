project(capi_test)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.13)

set(ONNXRUNTIME_ROOTDIR /home/kelamini/workspace/onnxruntime)

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

ADD_EXECUTABLE(capi_test  test1.cpp)
# if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
#   target_compile_definitions(capi_test PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
# endif()
target_link_libraries(capi_test onnxruntime)
