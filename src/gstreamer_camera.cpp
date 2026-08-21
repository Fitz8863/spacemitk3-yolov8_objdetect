#include "gstreamer_camera.h"

#include <opencv2/videoio.hpp>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <initializer_list>
#include <sstream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {
int xioctl(int fd, unsigned long request, void* arg) {
    int rc;
    do { rc = ::ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR);
    return rc;
}

bool supportsPixelFormat(int fd, v4l2_buf_type type,
                         std::initializer_list<std::uint32_t> formats) {
    v4l2_fmtdesc format{};
    format.type = type;
    for (format.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &format) == 0; ++format.index) {
        for (const std::uint32_t expected : formats) {
            if (format.pixelformat == expected) return true;
        }
    }
    return false;
}

bool findV4l2M2mDecoder(std::string& description) {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path video4linux("/sys/class/video4linux");
    if (!fs::exists(video4linux, error)) return false;

    for (const auto& entry : fs::directory_iterator(video4linux, error)) {
        if (error) break;
        const std::string node = "/dev/" + entry.path().filename().string();
        const int fd = ::open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) continue;

        v4l2_capability capability{};
        const bool queried = xioctl(fd, VIDIOC_QUERYCAP, &capability) == 0;
        if (!queried) {
            ::close(fd);
            continue;
        }

        const std::uint32_t caps =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                ? capability.device_caps
                : capability.capabilities;
        const bool multiplanar = (caps & V4L2_CAP_VIDEO_M2M_MPLANE) != 0;
        const bool single_planar = (caps & V4L2_CAP_VIDEO_M2M) != 0;
        const bool decodes_jpeg_to_nv12 =
            (multiplanar &&
             supportsPixelFormat(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
                                 {V4L2_PIX_FMT_JPEG, V4L2_PIX_FMT_MJPEG}) &&
             supportsPixelFormat(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                                 {V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_NV12M})) ||
            (single_planar &&
             supportsPixelFormat(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT,
                                 {V4L2_PIX_FMT_JPEG, V4L2_PIX_FMT_MJPEG}) &&
             supportsPixelFormat(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                 {V4L2_PIX_FMT_NV12}));
        ::close(fd);
        if (!decodes_jpeg_to_nv12) continue;

        std::ostringstream stream;
        stream << node << " driver="
               << reinterpret_cast<const char*>(capability.driver)
               << " card=" << reinterpret_cast<const char*>(capability.card);
        description = stream.str();
        return true;
    }
    return false;
}
}  // namespace

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

    // spacemitdec currently crashes inside the vendor MPP library when the
    // Linlon V5 V4L2 M2M decoder did not bind. Detect a usable M2M node before
    // constructing that pipeline, so the application can safely fall back to
    // GStreamer's software JPEG decoder instead of receiving SIGSEGV.
    std::string m2m_description;
    const bool hardware_decoder_available = findV4l2M2mDecoder(m2m_description);
    if (hardware_decoder_available) {
        std::cerr << "V4L2 M2M decoder detected: " << m2m_description
                  << "; preferring spacemitdec\n";
    } else {
        std::cerr << "No usable V4L2 M2M decoder detected; skipping spacemitdec "
                     "and falling back to jpegdec + videoconvert\n";
    }

    // The C920 advertises 24 fps rather than an exact 25 fps mode at 1280x720.
    // Go directly to the valid mode here: a failed 25 fps negotiation can leave
    // the vendor decoder teardown path unhealthy before the 24 fps retry.
    const bool c920_720p25 =
        width_ == 1280 && height_ == 720 && requested_fps_ == 25;
    if (c920_720p25) {
        std::cerr << "Camera mode 1280x720@25 maps to advertised 24 FPS; "
                     "trying 24 FPS directly\n";
    }
    const int candidates[] = {
        c920_720p25 ? 24 : requested_fps_,
        (!c920_720p25 && requested_fps_ == 25) ? 24 : 0
    };
    const char* decoders[] = {"spacemitdec", "jpegdec"};
    for (const char* decoder : decoders) {
        const bool hardware = std::strcmp(decoder, "spacemitdec") == 0;
        if (hardware && !hardware_decoder_available) continue;
        if (!hardware && hardware_decoder_available) {
            std::cerr << "spacemitdec pipeline unavailable; trying software JPEG decode\n";
        }

        for (int candidate : candidates) {
            if (candidate <= 0) continue;
            const std::string decoder_pipeline = hardware
                ? "spacemitdec code-type=9 ! "
                : "jpegdec ! videoconvert ! ";
            pipeline_ = "v4l2src device=" + device_ + " io-mode=2 ! "
                        "image/jpeg,width=" + std::to_string(width_) +
                        ",height=" + std::to_string(height_) +
                        ",framerate=" + std::to_string(candidate) + "/1 ! " +
                        decoder_pipeline +
                        "video/x-raw,format=NV12 ! "
                        "appsink drop=true max-buffers=1 enable-last-sample=false sync=false";
            std::cerr << "Opening OpenCV GStreamer pipeline: " << pipeline_ << "\n";
            if (capture_.open(pipeline_, cv::CAP_GSTREAMER)) {
                cv::Mat first;
                if (capture_.read(first) && !first.empty() && first.type() == CV_8UC1 &&
                    first.cols == width_ && first.rows == height_ * 3 / 2) {
                    negotiated_fps_ = candidate;
                    decoder_ = decoder;
                    std::cerr << "GStreamer camera opened: " << device_ << " "
                              << width_ << "x" << height_ << "@" << negotiated_fps_
                              << " decoder=" << decoder_ << " output=NV12\n";
                    return true;
                }
                capture_.release();
            }
        }
    }
    std::cerr << "Could not open a GStreamer MJPEG pipeline for " << device_ << "\n";
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
    decoder_.clear();
    negotiated_fps_ = 0;
}

bool GstreamerMjpegCamera::isOpen() const { return capture_.isOpened(); }
