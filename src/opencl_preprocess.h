#pragma once

#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

class OpenClPreprocessor {
public:
    OpenClPreprocessor();
    struct Result {
        std::shared_ptr<std::vector<float>> data;
        float scale = 1.0f; // source pixels per model pixel
        int pad_x = 0;
        int pad_y = 0;
        double ms = 0.0;
    };

    ~OpenClPreprocessor();
    bool init(int out_width = 640, int out_height = 640);
    // NV12 is a CV_8UC1 matrix with height * 3 / 2 rows. The host copy is
    // split into Y/U/V OpenCL images; color conversion, resize, letterbox and
    // CHW packing run in the GPU kernel.
    Result preprocess(const cv::Mat& nv12);
    const char* device_name() const { return device_name_.c_str(); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string device_name_;
};
