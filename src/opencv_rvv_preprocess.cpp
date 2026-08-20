#include "opencv_rvv_preprocess.h"

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint8_t kPaddingY = 114;
constexpr uint8_t kPaddingUV = 128;
constexpr double kInv255 = 1.0 / 255.0;

int even_at_least_two(int value) {
    return std::max(2, value & ~1);
}

struct LetterboxGeometry {
    float scale = 1.0f;
    int resized_width = 0;
    int resized_height = 0;
    int pad_x = 0;
    int pad_y = 0;
};

LetterboxGeometry calculate_geometry(int src_width, int src_height,
                                      int dst_width, int dst_height) {
    if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) {
        throw std::runtime_error("invalid preprocess dimensions");
    }
    // Same convention as the previous OpenCL implementation: scale is source
    // pixels per model pixel. For 1280x720 -> 640x640 it is 2.0, resulting in
    // a 640x360 image centered with 140-pixel top/bottom padding.
    const float scale = std::max(src_width / static_cast<float>(dst_width),
                                 src_height / static_cast<float>(dst_height));
    const int resized_width = even_at_least_two(
        static_cast<int>(std::lround(src_width / scale)));
    const int resized_height = even_at_least_two(
        static_cast<int>(std::lround(src_height / scale)));
    if (resized_width > dst_width || resized_height > dst_height) {
        throw std::runtime_error("letterbox resized image exceeds destination");
    }
    return {scale, resized_width, resized_height,
            (dst_width - resized_width) / 2,
            (dst_height - resized_height) / 2};
}
}  // namespace

struct OpenCvRvvPreprocessor::Impl {
    int out_w = 640;
    int out_h = 640;

    // Scratch buffers are reused by the single preprocessing worker. The
    // tensor is allocated per frame because it remains owned by the queue.
    cv::Mat resized_y;
    cv::Mat resized_uv;
    cv::Mat letterbox_y;
    cv::Mat letterbox_uv;
    cv::Mat rgb;
    std::array<cv::Mat, 3> channels;

    void ensure_buffers(const LetterboxGeometry& g) {
        resized_y.create(g.resized_height, g.resized_width, CV_8UC1);
        resized_uv.create(g.resized_height / 2, g.resized_width / 2, CV_8UC2);
        letterbox_y.create(out_h, out_w, CV_8UC1);
        letterbox_uv.create(out_h / 2, out_w / 2, CV_8UC2);
        rgb.create(out_h, out_w, CV_8UC3);
    }

    std::shared_ptr<std::vector<float>> rgb_to_tensor(const cv::Mat& image) {
        if (image.type() != CV_8UC3 || image.cols != out_w || image.rows != out_h) {
            throw std::runtime_error("unexpected RGB image shape in OpenCV preprocessor");
        }

        const size_t plane = static_cast<size_t>(out_w) * out_h;
        auto output = std::make_shared<std::vector<float>>(3 * plane);
        cv::split(image, channels.data());
        for (int c = 0; c < 3; ++c) {
            cv::Mat dst(out_h, out_w, CV_32FC1,
                        output->data() + static_cast<size_t>(c) * plane);
            channels[c].convertTo(dst, CV_32F, kInv255, 0.0);
        }
        return output;
    }
};

OpenCvRvvPreprocessor::OpenCvRvvPreprocessor() = default;
OpenCvRvvPreprocessor::~OpenCvRvvPreprocessor() = default;

bool OpenCvRvvPreprocessor::init(int out_width, int out_height) {
    try {
        if (out_width <= 0 || out_height <= 0 || (out_width & 1) || (out_height & 1)) {
            throw std::runtime_error("OpenCV preprocess output dimensions must be positive and even");
        }
        impl_ = std::make_unique<Impl>();
        impl_->out_w = out_width;
        impl_->out_h = out_height;
        cv::setUseOptimized(true);
        device_name_ = "OpenCV RVV";
        std::cout << "OpenCV preprocess: RVV backend (OpenCV " << CV_VERSION
                  << ") output=" << out_width << "x" << out_height << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCV RVV init failed: " << e.what() << "\n";
        impl_.reset();
        return false;
    }
}

OpenCvRvvPreprocessor::Result OpenCvRvvPreprocessor::preprocess(const cv::Mat& nv12) {
    if (!impl_) {
        throw std::runtime_error("OpenCV RVV preprocessor is not initialized");
    }
    if (nv12.empty() || nv12.type() != CV_8UC1 || (nv12.cols & 1) ||
        (nv12.rows % 3) != 0) {
        throw std::runtime_error("expected NV12 CV_8UC1 Mat with even width and 3/2 height");
    }

    const auto t0 = Clock::now();
    const int width = nv12.cols;
    const int height = (nv12.rows * 2) / 3;
    if ((height & 1) || height <= 0) {
        throw std::runtime_error("NV12 height must be positive and even");
    }
    const LetterboxGeometry g = calculate_geometry(width, height,
                                                    impl_->out_w, impl_->out_h);
    if ((g.pad_x & 1) || (g.pad_y & 1)) {
        throw std::runtime_error("NV12 letterbox offsets must be even");
    }

    // Keep the Y and interleaved UV planes as views. No full-resolution RGB
    // conversion is done before resize, which is the important RVV fast path.
    const cv::Mat y(height, width, CV_8UC1, nv12.data, nv12.step);
    const cv::Mat uv(height / 2, width / 2, CV_8UC2,
                     nv12.data + static_cast<size_t>(height) * nv12.step,
                     nv12.step);
    impl_->ensure_buffers(g);
    cv::resize(y, impl_->resized_y, impl_->resized_y.size(), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::resize(uv, impl_->resized_uv, impl_->resized_uv.size(), 0.0, 0.0,
               cv::INTER_LINEAR);

    impl_->letterbox_y.setTo(cv::Scalar(kPaddingY));
    impl_->letterbox_uv.setTo(cv::Scalar(kPaddingUV, kPaddingUV));
    impl_->resized_y.copyTo(impl_->letterbox_y(
        cv::Rect(g.pad_x, g.pad_y, g.resized_width, g.resized_height)));
    impl_->resized_uv.copyTo(impl_->letterbox_uv(
        cv::Rect(g.pad_x / 2, g.pad_y / 2,
                 g.resized_width / 2, g.resized_height / 2)));

    cv::cvtColorTwoPlane(impl_->letterbox_y, impl_->letterbox_uv,
                         impl_->rgb, cv::COLOR_YUV2RGB_NV12);

    Result result;
    result.data = impl_->rgb_to_tensor(impl_->rgb);
    result.scale = g.scale;
    result.pad_x = g.pad_x;
    result.pad_y = g.pad_y;
    result.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return result;
}
