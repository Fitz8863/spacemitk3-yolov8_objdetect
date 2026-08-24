cmake_minimum_required(VERSION 3.16)
project(yolov8_k3_opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# OpenCV and GStreamer are discovered through normal system search paths.
# For a non-standard SDK, pass -DOpenCV_DIR=... and/or -DSPACEMIT_ORT_ROOT=....
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
    gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)

# SpaceMIT ONNX Runtime + EP; default to system prefixes, with an explicit
# opt-in prefix for boards where the SDK is installed outside /usr or /lib.
set(SPACEMIT_ORT_ROOT "" CACHE PATH
    "SpaceMIT ONNX Runtime prefix containing include/ and lib/")

if(SPACEMIT_ORT_ROOT)
    find_path(SPACEMIT_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        PATHS "${SPACEMIT_ORT_ROOT}/include" NO_DEFAULT_PATH)
    find_path(SPACEMIT_ORT_ENV_INCLUDE_DIR NAMES spacemit_ort_env.h
        PATHS "${SPACEMIT_ORT_ROOT}/include" NO_DEFAULT_PATH)
    find_library(SPACEMIT_ORT_LIBRARY NAMES onnxruntime
        PATHS "${SPACEMIT_ORT_ROOT}/lib"
              "${SPACEMIT_ORT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    find_library(SPACEMIT_EP_LIBRARY NAMES spacemit_ep
        PATHS "${SPACEMIT_ORT_ROOT}/lib"
              "${SPACEMIT_ORT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
else()
    find_path(SPACEMIT_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    find_path(SPACEMIT_ORT_ENV_INCLUDE_DIR NAMES spacemit_ort_env.h)
    find_library(SPACEMIT_ORT_LIBRARY NAMES onnxruntime)
    find_library(SPACEMIT_EP_LIBRARY NAMES spacemit_ep)
endif()

if(NOT SPACEMIT_ORT_INCLUDE_DIR OR NOT SPACEMIT_ORT_ENV_INCLUDE_DIR OR
   NOT SPACEMIT_ORT_LIBRARY OR NOT SPACEMIT_EP_LIBRARY)
    message(FATAL_ERROR
        "SpaceMIT ONNX Runtime development files were not found.\n"
        "Install the SDK in a system prefix or configure with:\n"
        "  -DSPACEMIT_ORT_ROOT=<spacemit-onnxruntime-prefix>")
endif()

add_executable(yolov8_camera
    src/main.cpp
    src/gstreamer_camera.cpp
    src/opencl_preprocess.cpp
    src/yolov8_detector.cpp)

target_include_directories(yolov8_camera PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SPACEMIT_ORT_INCLUDE_DIR}
    ${SPACEMIT_ORT_ENV_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(yolov8_camera PRIVATE
    ${OpenCV_LIBS}
    OpenCL::OpenCL
    ${SPACEMIT_ORT_LIBRARY}
    ${SPACEMIT_EP_LIBRARY}
    Threads::Threads
    PkgConfig::GSTREAMER
    dl)

set_target_properties(yolov8_camera PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "")

target_compile_options(yolov8_camera PRIVATE -O3 -ffast-math -Wall -Wextra -Wpedantic)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCL + GStreamer camera + OpenCL GPU preprocessing project")
message(STATUS "SpaceMIT ORT include dir: ${SPACEMIT_ORT_INCLUDE_DIR}")
message(STATUS "SpaceMIT ORT library: ${SPACEMIT_ORT_LIBRARY}")
message(STATUS "SpaceMIT EP library: ${SPACEMIT_EP_LIBRARY}")
