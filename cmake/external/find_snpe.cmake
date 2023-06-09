if (NOT SNPE_ARCH_ABI)
    if(MSVC)
        string(TOLOWER ${CMAKE_GENERATOR_PLATFORM} GEN_PLATFORM)
        message(STATUS "Building MSVC for architecture ${CMAKE_SYSTEM_PROCESSOR} with CMAKE_GENERATOR_PLATFORM as ${GEN_PLATFORM}")
        if (${GEN_PLATFORM} STREQUAL "arm64")
            set(SNPE_ARCH_ABI aarch64-windows-vc19)
        else()
            set(SNPE_ARCH_ABI x86_64-windows-vc19)
        endif()
    else()
        if (CMAKE_SYSTEM_NAME STREQUAL "Android")
            set(SNPE_ARCH_ABI aarch64-android-clang6.0)
        elseif (LINUX)
            if (${GEN_PLATFORM} STREQUAL "x64")
                set(SNPE_ARCH_ABI x86_64-linux-clang)
            else()
                set(SNPE_ARCH_ABI aarch64-linux-gcc4.9)
            endif()
        endif()
    endif()
    list(APPEND onnxruntime_LINK_DIRS ${SNPE_ROOT}/lib/${SNPE_ARCH_ABI})
endif()
file(TO_CMAKE_PATH ${SNPE_ROOT} SNPE_ROOT)
get_filename_component(SNPE_CMAKE_DIR ${SNPE_ROOT} ABSOLUTE)
file(TO_CMAKE_PATH "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}" SNPE_LIB_DIR)
file(TO_NATIVE_PATH ${SNPE_LIB_DIR} SNPE_NATIVE_DIR)
message(STATUS "Looking for SNPE library in ${SNPE_NATIVE_DIR}")
find_library(SNPE NAMES snpe SNPE libSNPE.so PATHS "${SNPE_NATIVE_DIR}" "${SNPE_ROOT}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)

file(GLOB SNPE_SO_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.so" "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.dll")
# add dsp skel files to distribution
file(GLOB SNPE_DSP_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/dsp/*.so")
list(APPEND SNPE_SO_FILES ${QCDK_FILES} ${SNPE_DSP_FILES})

if(NOT SNPE OR NOT SNPE_SO_FILES)
  message(ERROR "Snpe not found in ${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI} for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()

set(SNPE_NN_LIBS ${SNPE})
if(ANDROID)
  # Use libc++_shared.so from SNPE SDK
  list(APPEND SNPE_NN_LIBS libc++_shared.so)
endif()

message(STATUS "SNPE library at ${SNPE}")
message(STATUS "SNPE so/dlls in ${SNPE_SO_FILES}")