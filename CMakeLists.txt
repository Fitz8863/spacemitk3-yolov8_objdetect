cmake_minimum_required(VERSION 3.16)
project(yolov8_k3_opencv_rvv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# On K3, explicitly select the SpacemiT RVV OpenCV package:
#   cmake -S . -B build -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
# If OpenCV_DIR is omitted, CMake uses its normal package search path.
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(Threads REQUIRED)

add_executable(yolov8_camera
    src/main.cpp
    src/gstreamer_camera.cpp
    src/opencv_rvv_preprocess.cpp
    src/yolov8_detector.cpp
)

target_include_directories(yolov8_camera PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(yolov8_camera PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    spacemit_ep
    Threads::Threads
    dl
)

target_compile_options(yolov8_camera PRIVATE -O3 -ffast-math -Wall -Wextra -Wpedantic)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV library dirs: ${OpenCV_LIBRARY_DIRS}")
message(STATUS "GStreamer camera + OpenCV RVV preprocessing project")
