#include "gstreamer_camera.h"

#include <opencv2/videoio.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {
int xioctl(int fd, unsigned long request, void* arg) {
    int rc;
    do { rc = ::ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR);
    return rc;
}
}

GstreamerMjpegCamera::~GstreamerMjpegCamera() { close(); }

bool GstreamerMjpegCamera::setControl(const std::string& device, unsigned id,
                                       int value, const char* name) {
    if (value < 0) return true;
    const int fd = ::open(device.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::cerr << "Cannot open " << device << " for " << name << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    v4l2_control ctl{};
    ctl.id = id;
    ctl.value = value;
    const bool ok = xioctl(fd, VIDIOC_S_CTRL, &ctl) == 0;
    if (!ok) {
        std::cerr << "Optional " << name << "=" << value << " failed: "
                  << std::strerror(errno) << "\n";
    }
    ::close(fd);
    return ok;
}

bool GstreamerMjpegCamera::configureControls() const {
    // Keep the same defaults as ultralytics/detect.py: fixed focus and zoomed
    // image. Autofocus must be disabled before Focus Absolute is written.
    if (focus_ >= 0 && !setControl(device_, V4L2_CID_FOCUS_AUTO, 0, "focus_auto"))
        return false;
    if (!setControl(device_, V4L2_CID_FOCUS_ABSOLUTE, focus_, "focus")) return false;
    if (!setControl(device_, V4L2_CID_ZOOM_ABSOLUTE, zoom_, "zoom")) return false;
    return true;
}

bool GstreamerMjpegCamera::open(int camera_index, const std::string& device,
                                 int width, int height, int fps, int focus, int zoom) {
    close();
    device_ = device.empty() ? ("/dev/video" + std::to_string(camera_index)) : device;
    width_ = width;
    height_ = height;
    requested_fps_ = fps;
    focus_ = focus;
    zoom_ = zoom;
    if (width_ <= 0 || height_ <= 0 || requested_fps_ <= 0) {
        std::cerr << "Invalid camera geometry or fps\n";
        return false;
    }
    if (!configureControls()) return false;

    // The C920 advertises 24 fps rather than an exact 25 fps mode at 1280x720.
    // Try the requested caps first, then the camera's nearest advertised mode.
    const int candidates[] = {requested_fps_, requested_fps_ == 25 ? 24 : 0};
    for (int candidate : candidates) {
        if (candidate <= 0) continue;
        // NV12 is intentional: asking for BGR makes videoconvert run in this
        // board's riscv64 GStreamer build and can crash. OpenCV receives the
        // 1.5*height NV12 image and converts it only for display.
        pipeline_ = "v4l2src device=" + device_ + " io-mode=2 ! "
                    "image/jpeg,width=" + std::to_string(width_) +
                    ",height=" + std::to_string(height_) +
                    ",framerate=" + std::to_string(candidate) + "/1 ! "
                    "spacemitdec code-type=9 ! "
                    "video/x-raw,format=NV12 ! "
                    "appsink drop=true max-buffers=1 enable-last-sample=false sync=false";
        std::cerr << "Opening OpenCV GStreamer pipeline: " << pipeline_ << "\n";
        if (capture_.open(pipeline_, cv::CAP_GSTREAMER)) {
            cv::Mat first;
            if (capture_.read(first) && !first.empty() && first.type() == CV_8UC1 &&
                first.cols == width_ && first.rows == height_ * 3 / 2) {
                negotiated_fps_ = candidate;
                std::cerr << "GStreamer camera opened: " << device_ << " "
                          << width_ << "x" << height_ << "@" << negotiated_fps_
                          << " decoder=spacemitdec output=NV12\n";
                return true;
            }
            capture_.release();
        }
    }
    std::cerr << "Could not open a K3 GStreamer MJPEG pipeline for " << device_ << "\n";
    return false;
}

bool GstreamerMjpegCamera::read(cv::Mat& nv12, int timeout_ms) {
    (void)timeout_ms; // VideoCapture/appsink performs the blocking wait.
    nv12.release();
    if (!capture_.isOpened()) return false;
    cv::Mat captured;
    if (!capture_.read(captured)) return false;
    if (captured.empty() || captured.type() != CV_8UC1 || captured.cols != width_ ||
        captured.rows < height_ * 3 / 2) {
        std::cerr << "Unexpected GStreamer frame: " << captured.cols << "x" << captured.rows
                  << " type=" << captured.type() << ", expected " << width_ << "x"
                  << height_ * 3 / 2 << " CV_8UC1 NV12\n";
        return false;
    }
    // VideoCapture's GStreamer backend may return a Mat backed by the appsink/
    // decoder buffer. Copy it before the Mat is queued or used by another
    // thread; otherwise holding a few frames can prevent spacemitdec from
    // returning its finite VPU output buffers (queueBuffer EINVAL).
    // Some GStreamer/OpenCV builds preserve the decoder's padded stride in
    // the Mat. Normalize to tightly packed NV12 before the next stage.
    nv12.create(height_ * 3 / 2, width_, CV_8UC1);
    for (int row = 0; row < height_ * 3 / 2; ++row)
        std::memcpy(nv12.ptr(row), captured.ptr(row), static_cast<size_t>(width_));
    return true;
}

void GstreamerMjpegCamera::close() {
    if (capture_.isOpened()) capture_.release();
    pipeline_.clear();
    negotiated_fps_ = 0;
}

bool GstreamerMjpegCamera::isOpen() const { return capture_.isOpened(); }
