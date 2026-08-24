#pragma once

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>

// A camera frame may directly view the mapped GstBuffer. `owner` keeps the
// GstSample and its mapping alive until all OpenCV consumers release the frame.
// If the producer layout is not a single compatible NV12 view, read() falls
// back to a tightly packed copy and leaves owner empty.
struct GstreamerFrame {
    cv::Mat nv12;
    std::shared_ptr<void> owner;
    bool zero_copy = false;
};

class GstreamerMjpegCamera {
public:
    GstreamerMjpegCamera() = default;
    ~GstreamerMjpegCamera();
    GstreamerMjpegCamera(const GstreamerMjpegCamera&) = delete;

    bool open(int camera_index, const std::string& device, int width, int height,
              int fps, int focus = 0, int zoom = 181, int max_frames = 0);
    bool read(GstreamerFrame& frame, int timeout_ms = 1000);
    // Compatibility API: this necessarily makes a detached copy because a
    // bare cv::Mat cannot retain the GstBuffer mapping lifetime.
    bool read(cv::Mat& nv12, int timeout_ms = 1000);
    void close();
    bool isOpen() const;
    const std::string& device() const { return device_; }
    const std::string& pipeline() const { return pipeline_description_; }
    const std::string& decoder() const { return decoder_; }
    int negotiated_fps() const { return negotiated_fps_; }

private:
    bool configureControls() const;
    bool openPipeline(const std::string& decoder, int candidate_fps,
                      std::string& error);
    bool pullFrame(GstreamerFrame& frame, int timeout_ms, std::string& error);
    bool switchToSoftware();
    bool takeBusError(std::string& error, bool& eos);
    void destroyPipeline(bool send_eos);
    static bool setControl(const std::string& device, unsigned id, int value,
                           const char* name);

    GstElement* gst_pipeline_ = nullptr;
    GstAppSink* appsink_ = nullptr;
    std::string device_;
    std::string pipeline_description_;
    std::string decoder_;
    int width_ = 0;
    int height_ = 0;
    int requested_fps_ = 0;
    int negotiated_fps_ = 0;
    int focus_ = 0;
    int zoom_ = 181;
    int max_frames_ = 0;
    bool hardware_decoder_ = false;
    bool eos_ = false;
    bool runtime_fallback_attempted_ = false;
    bool zero_copy_reported_ = false;
    bool copy_reported_ = false;
};
