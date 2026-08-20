#include "yolov8_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr int kModelWidth = 640;
constexpr int kModelHeight = 640;
constexpr int kBoxChannels = 4;
constexpr int kExpectedClasses = 6;

float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

std::vector<Detection> nms_class_aware(std::vector<Detection> candidates,
                                        float iou_threshold, int max_detections) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<Detection> kept;
    kept.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(max_detections)));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (candidate.class_id == selected.class_id && iou(candidate, selected) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (static_cast<int>(kept.size()) >= max_detections) break;
        }
    }
    return kept;
}

void print_shape(const char* label, const std::vector<int64_t>& shape) {
    std::cout << label << " [";
    for (std::size_t i = 0; i < shape.size(); ++i) std::cout << (i ? "," : "") << shape[i];
    std::cout << "]";
}
}  // namespace

struct Yolov8Detector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolov8-k3"};
    std::unique_ptr<Ort::Session> session;
};

Yolov8Detector::Yolov8Detector() = default;
Yolov8Detector::~Yolov8Detector() = default;

bool Yolov8Detector::init(const std::string& model_path, int intra_threads,
                          const std::string& ep_affinity) {
    try {
        impl_ = std::make_unique<Impl>();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(std::max(1, intra_threads));
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::unordered_map<std::string, std::string> ep_options;
        const int ep_threads = std::max(1, intra_threads);
        ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(ep_threads);
        if (!ep_affinity.empty()) {
            ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = ep_affinity;
            std::cout << "SpaceMIT EP affinity: " << ep_affinity << "\n";
        }
        Ort::SessionOptionsSpaceMITEnvInit(options, ep_options);
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = impl_->session->GetInputNameAllocated(0, allocator).get();
        output_name_ = impl_->session->GetOutputNameAllocated(0, allocator).get();
        input_shape_ = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Model loaded with SpaceMIT EP: " << model_path << "\n  ";
        print_shape(("input " + input_name_).c_str(), input_shape_);
        std::cout << " ";
        print_shape(("output " + output_name_).c_str(), output_shape_);
        std::cout << "\n";

        if (!input_shape_.empty() && input_shape_.size() != 4)
            throw std::runtime_error("unexpected input rank; expected rank 4");
        if (!input_shape_.empty()) {
            if (input_shape_[1] > 0 && input_shape_[1] != 3)
                throw std::runtime_error("unexpected input channels; expected 3");
            if (input_shape_[2] > 0 && input_shape_[2] != kModelHeight)
                throw std::runtime_error("unexpected input height; expected 640");
            if (input_shape_[3] > 0 && input_shape_[3] != kModelWidth)
                throw std::runtime_error("unexpected input width; expected 640");
        }
        if (!output_shape_.empty() && output_shape_.size() != 3)
            throw std::runtime_error("unexpected YOLOv8 output rank; expected rank 3");
        session_opaque_ = impl_->session.get();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Model init failed: " << e.what() << "\n";
        impl_.reset();
        session_opaque_ = nullptr;
        return false;
    }
}

std::vector<Detection> Yolov8Detector::infer(const float* data, std::size_t count,
                                             float conf_threshold, float scale,
                                             int pad_x, int pad_y, int image_width,
                                             int image_height, float iou_threshold,
                                             int max_detections) {
    if (!impl_ || !impl_->session) throw std::runtime_error("detector is not initialized");
    if (!data || count != static_cast<std::size_t>(3 * kModelWidth * kModelHeight))
        throw std::runtime_error("unexpected input tensor size; expected 3*640*640 floats");

    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, kModelHeight, kModelWidth};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(data), count,
                                                        input_shape.data(), input_shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                       output_names, 1);
    if (outputs.empty() || !outputs[0].IsTensor()) throw std::runtime_error("YOLOv8 output is not a tensor");

    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = output_info.GetShape();
    if (shape.size() != 3) throw std::runtime_error("YOLOv8 output must have rank 3");
    const float* values = outputs[0].GetTensorData<float>();

    // Ultralytics YOLOv8 nms=False exports [1, 4+nc, 8400]. Accept the
    // transposed [1,8400,4+nc] form too, so the executable reports a useful
    // error only for genuinely incompatible models.
    const auto valid_layout = [](int64_t channel_dim, int64_t prediction_dim) {
        return channel_dim >= kBoxChannels + 1 && prediction_dim > 0;
    };
    bool channels_first = valid_layout(shape[1], shape[2]);
    const bool rows_first = valid_layout(shape[2], shape[1]);
    if (!channels_first && !rows_first) {
        throw std::runtime_error("YOLOv8 output must contain box and class channels");
    }
    // Prefer [1,C,N] when both dimensions happen to look plausible. This is
    // the native Ultralytics export and avoids silently transposing a model
    // whose prediction count is unusually small.
    if (channels_first && rows_first) {
        channels_first = shape[1] <= shape[2];
    }
    const std::size_t channels = static_cast<std::size_t>(channels_first ? shape[1] : shape[2]);
    const std::size_t predictions = static_cast<std::size_t>(channels_first ? shape[2] : shape[1]);
    const int num_classes = static_cast<int>(channels) - kBoxChannels;

    static int debug_frames = 0;
    if (debug_frames < 3) {
        std::cerr << "YOLOv8 output debug: layout=" << (channels_first ? "[1,C,N]" : "[1,N,C]")
                  << " channels=" << channels << " predictions=" << predictions
                  << " classes=" << num_classes << "\n";
        ++debug_frames;
    }

    auto at = [&](std::size_t p, std::size_t c) -> float {
        return channels_first ? values[c * predictions + p] : values[p * channels + c];
    };
    std::vector<Detection> candidates;
    candidates.reserve(predictions / 4);
    for (std::size_t p = 0; p < predictions; ++p) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            const float score = at(p, static_cast<std::size_t>(kBoxChannels + c));
            if (std::isfinite(score) && score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        if (best_class < 0 || best_score < conf_threshold) continue;

        // The exported YOLOv8 graph has already applied DFL and decoded each
        // row to xywh in 640x640 model coordinates; only letterbox undo is
        // needed here. Do not apply sigmoid again to the class score.
        const float cx = at(p, 0);
        const float cy = at(p, 1);
        const float w = at(p, 2);
        const float h = at(p, 3);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
            w <= 0.0f || h <= 0.0f) continue;
        Detection d;
        d.x1 = (cx - 0.5f * w - static_cast<float>(pad_x)) * scale;
        d.y1 = (cy - 0.5f * h - static_cast<float>(pad_y)) * scale;
        d.x2 = (cx + 0.5f * w - static_cast<float>(pad_x)) * scale;
        d.y2 = (cy + 0.5f * h - static_cast<float>(pad_y)) * scale;
        d.x1 = std::clamp(d.x1, 0.0f, static_cast<float>(image_width));
        d.y1 = std::clamp(d.y1, 0.0f, static_cast<float>(image_height));
        d.x2 = std::clamp(d.x2, 0.0f, static_cast<float>(image_width));
        d.y2 = std::clamp(d.y2, 0.0f, static_cast<float>(image_height));
        d.confidence = best_score;
        d.class_id = best_class;
        if (d.x2 > d.x1 && d.y2 > d.y1) candidates.push_back(d);
    }
    return nms_class_aware(std::move(candidates), iou_threshold, std::max(1, max_detections));
}
