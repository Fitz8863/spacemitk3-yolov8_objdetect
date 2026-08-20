#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
};

// YOLOv8 detector for the Ultralytics nms=False ONNX output.
// The expected output is [1, 4 + num_classes, num_predictions] (for this
// model [1,10,8400]); the decoder restores boxes to the camera image and runs
// class-aware NMS outside the ONNX graph.
class Yolov8Detector {
public:
    Yolov8Detector();
    ~Yolov8Detector();
    Yolov8Detector(const Yolov8Detector&) = delete;
    Yolov8Detector& operator=(const Yolov8Detector&) = delete;

    bool init(const std::string& model_path, int intra_threads = 1,
              const std::string& ep_affinity = {});
    std::vector<Detection> infer(const float* data, std::size_t count,
                                 float conf_threshold, float scale,
                                 int pad_x, int pad_y,
                                 int image_width, int image_height,
                                 float iou_threshold = 0.45f,
                                 int max_detections = 300);
    bool ready() const { return session_opaque_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void* session_opaque_ = nullptr;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};
