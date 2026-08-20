#pragma once

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

// OpenCV-RVV CPU preprocessor for the K3 GStreamer camera path.
// The camera/decoder stage supplies a tightly packed NV12 cv::Mat with shape
// [height * 3 / 2, width]. This class produces [1,3,640,640] RGB FP32 CHW
// normalized to [0,1], with the same letterbox geometry as the OpenCL path.
class OpenCvRvvPreprocessor {
public:
    OpenCvRvvPreprocessor();
    ~OpenCvRvvPreprocessor();

    struct Result {
        std::shared_ptr<std::vector<float>> data;
        // Source pixels per model pixel, matching the detector's coordinate
        // restoration convention.
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        double ms = 0.0;
    };

    // The OpenCV RVV implementation is selected by the OpenCV package passed
    // to CMake (normally /opt/opencv-spacemit/lib/cmake/opencv4).
    bool init(int out_width = 640, int out_height = 640);

    // Accepts a tightly packed NV12 image: rows=height*3/2, cols=width,
    // type=CV_8UC1. The method also accepts a non-contiguous Mat and honors
    // its row stride.
    Result preprocess(const cv::Mat& nv12);

    const char* device_name() const { return device_name_.c_str(); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string device_name_;
};
