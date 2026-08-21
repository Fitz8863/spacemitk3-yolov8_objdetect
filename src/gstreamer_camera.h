#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

// OpenCV's VideoCapture owns the GStreamer pipeline. The pipeline keeps JPEG
// capture and MJPEG decode stay inside GStreamer. spacemitdec is preferred when
// a V4L2 M2M decoder exists; jpegdec is the safe software fallback.
class GstreamerMjpegCamera {
public:
    GstreamerMjpegCamera() = default;
    ~GstreamerMjpegCamera();
    GstreamerMjpegCamera(const GstreamerMjpegCamera&) = delete;

    bool open(int camera_index, const std::string& device, int width, int height,
              int fps, int focus = 0, int zoom = 181);
    bool read(cv::Mat& nv12, int timeout_ms = 1000);
    void close();
    bool isOpen() const;
    const std::string& device() const { return device_; }
    const std::string& pipeline() const { return pipeline_; }
    const std::string& decoder() const { return decoder_; }
    int negotiated_fps() const { return negotiated_fps_; }

private:
    bool configureControls() const;
    static bool setControl(const std::string& device, unsigned id, int value,
                           const char* name);

    cv::VideoCapture capture_;
    std::string device_;
    std::string pipeline_;
    std::string decoder_;
    int width_ = 0;
    int height_ = 0;
    int requested_fps_ = 0;
    int negotiated_fps_ = 0;
    int focus_ = 0;
    int zoom_ = 181;
};
