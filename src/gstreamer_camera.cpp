#include "gstreamer_camera.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <fcntl.h>
#include <gst/video/video.h>
#include <iostream>
#include <initializer_list>
#include <linux/videodev2.h>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

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
        for (const std::uint32_t expected : formats)
            if (format.pixelformat == expected) return true;
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
        if (xioctl(fd, VIDIOC_QUERYCAP, &capability) != 0) {
            ::close(fd);
            continue;
        }
        const std::uint32_t caps =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                ? capability.device_caps : capability.capabilities;
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
        stream << node << " driver=" << reinterpret_cast<const char*>(capability.driver)
               << " card=" << reinterpret_cast<const char*>(capability.card);
        description = stream.str();
        return true;
    }
    return false;
}

void ensureGstreamerInitialized() {
    static const bool initialized = [] {
        gst_init(nullptr, nullptr);
        return true;
    }();
    (void)initialized;
}

std::string gstErrorMessage(GError* error) {
    const std::string result = error && error->message ? error->message : "unknown GStreamer error";
    if (error) g_error_free(error);
    return result;
}

struct GstMappedFrameOwner {
    GstSample* sample = nullptr;
    GstVideoFrame video_frame{};
    bool mapped = false;

    ~GstMappedFrameOwner() {
        if (mapped) gst_video_frame_unmap(&video_frame);
        if (sample) gst_sample_unref(sample);
    }
};
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
    if (!ok)
        std::cerr << "Optional " << name << "=" << value << " failed: "
                  << std::strerror(errno) << "\n";
    ::close(fd);
    return ok;
}

bool GstreamerMjpegCamera::configureControls() const {
    if (focus_ >= 0 && !setControl(device_, V4L2_CID_FOCUS_AUTO, 0, "focus_auto")) return false;
    if (!setControl(device_, V4L2_CID_FOCUS_ABSOLUTE, focus_, "focus")) return false;
    if (!setControl(device_, V4L2_CID_ZOOM_ABSOLUTE, zoom_, "zoom")) return false;
    return true;
}

bool GstreamerMjpegCamera::openPipeline(const std::string& decoder, int candidate_fps,
                                        std::string& error) {
    ensureGstreamerInitialized();
    const bool hardware = decoder == "spacemitdec";
    const std::string decoder_pipeline = hardware
        ? "spacemitdec code-type=9 ! " : "jpegdec ! videoconvert ! ";
    pipeline_description_ = "v4l2src device=" + device_ + " io-mode=2" +
                (max_frames_ > 0 ? " num-buffers=" + std::to_string(max_frames_ + 1) : "") +
                " ! image/jpeg,width=" + std::to_string(width_) +
                ",height=" + std::to_string(height_) +
                ",framerate=" + std::to_string(candidate_fps) + "/1 ! " +
                decoder_pipeline +
                "video/x-raw,format=NV12 ! appsink name=appsink max-buffers=1 "
                "drop=true enable-last-sample=false sync=false";
    std::cerr << "Opening native GStreamer pipeline: " << pipeline_description_ << "\n";

    GError* parse_error = nullptr;
    GstElement* parsed = gst_parse_launch(pipeline_description_.c_str(), &parse_error);
    if (!parsed) {
        error = gstErrorMessage(parse_error);
        pipeline_description_.clear();
        return false;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(parsed), "appsink");
    if (!sink || !GST_IS_APP_SINK(sink)) {
        error = "appsink element was not created";
        if (sink) gst_object_unref(sink);
        gst_object_unref(parsed);
        return false;
    }
    gst_pipeline_ = parsed;
    appsink_ = GST_APP_SINK(sink);
    gst_app_sink_set_max_buffers(appsink_, 1);
    gst_app_sink_set_drop(appsink_, TRUE);

    const GstStateChangeReturn state = gst_element_set_state(gst_pipeline_, GST_STATE_PLAYING);
    if (state == GST_STATE_CHANGE_FAILURE) {
        error = "pipeline could not enter PLAYING";
        destroyPipeline(false);
        return false;
    }
    GstState current = GST_STATE_NULL;
    GstState pending = GST_STATE_NULL;
    if (gst_element_get_state(gst_pipeline_, &current, &pending, 3 * GST_SECOND) == GST_STATE_CHANGE_FAILURE) {
        error = "pipeline state transition failed";
        destroyPipeline(false);
        return false;
    }
    decoder_ = decoder;
    hardware_decoder_ = hardware;
    negotiated_fps_ = candidate_fps;
    eos_ = false;
    return true;
}

bool GstreamerMjpegCamera::takeBusError(std::string& error, bool& eos) {
    if (!gst_pipeline_) return false;
    GstBus* bus = gst_element_get_bus(gst_pipeline_);
    GstMessage* message = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!message) {
        gst_object_unref(bus);
        return false;
    }
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
        eos = true;
    } else {
        GError* gerror = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &gerror, &debug);
        error = gstErrorMessage(gerror);
        if (debug) g_free(debug);
    }
    gst_message_unref(message);
    gst_object_unref(bus);
    return true;
}

bool GstreamerMjpegCamera::pullFrame(GstreamerFrame& frame, int timeout_ms,
                                      std::string& error) {
    frame.nv12.release();
    frame.owner.reset();
    frame.zero_copy = false;
    if (!gst_pipeline_ || !appsink_ || eos_) return false;
    GstSample* sample = gst_app_sink_try_pull_sample(
        appsink_, static_cast<GstClockTime>(std::max(1, timeout_ms)) * GST_MSECOND);
    if (!sample) {
        bool saw_eos = false;
        if (takeBusError(error, saw_eos) && saw_eos) eos_ = true;
        if (error.empty() && !eos_) error = "timed out waiting for a decoded frame";
        return false;
    }
    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo info{};
    const bool valid_caps = caps && gst_video_info_from_caps(&info, caps) &&
                            GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_NV12 &&
                            GST_VIDEO_INFO_WIDTH(&info) == width_ &&
                            GST_VIDEO_INFO_HEIGHT(&info) == height_;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstVideoFrame mapped{};
    if (!valid_caps || !buffer || !gst_video_frame_map(&mapped, &info, buffer, GST_MAP_READ)) {
        error = "decoded sample is not a valid NV12 frame";
        gst_sample_unref(sample);
        return false;
    }

    const guint8* y = static_cast<const guint8*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 0));
    const guint8* uv = static_cast<const guint8*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 1));
    const gint y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 0);
    const gint uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 1);
    const bool compatible_single_view =
        y && uv && y_stride >= width_ && uv_stride == y_stride &&
        uv == y + static_cast<ptrdiff_t>(height_) * y_stride;

    if (compatible_single_view) {
        auto owner = std::make_shared<GstMappedFrameOwner>();
        owner->sample = sample;
        owner->video_frame = mapped;
        owner->mapped = true;
        frame.nv12 = cv::Mat(height_ * 3 / 2, width_, CV_8UC1,
                             const_cast<guint8*>(y), static_cast<size_t>(y_stride));
        frame.owner = std::move(owner);
        frame.zero_copy = true;
        if (!zero_copy_reported_) {
            std::cerr << "[Camera] NV12 path: zero-copy GstBuffer -> OpenCV Mat header "
                         "(owner retained until consumers release the frame)\n";
            zero_copy_reported_ = true;
        }
        return true;
    }

    // Some GStreamer/VPU allocations expose separate planes or incompatible
    // padding. Keep the safe fallback for those layouts.
    frame.nv12.create(height_ * 3 / 2, width_, CV_8UC1);
    for (int row = 0; row < height_; ++row)
        std::memcpy(frame.nv12.ptr(row), y + static_cast<ptrdiff_t>(row) * y_stride, width_);
    for (int row = 0; row < height_ / 2; ++row)
        std::memcpy(frame.nv12.ptr(height_ + row),
                    uv + static_cast<ptrdiff_t>(row) * uv_stride, width_);
    gst_video_frame_unmap(&mapped);
    gst_sample_unref(sample);
    if (!copy_reported_) {
        std::cerr << "[Camera] NV12 path: safe copy fallback because GstBuffer planes "
                     "are not one compatible OpenCV view\n";
        copy_reported_ = true;
    }
    return true;
}

bool GstreamerMjpegCamera::switchToSoftware() {
    if (!hardware_decoder_ || runtime_fallback_attempted_) return false;
    runtime_fallback_attempted_ = true;
    std::cerr << "[Decoder] Hardware decoder runtime failure; falling back to software decoder jpegdec\n";
    const int fps = negotiated_fps_ > 0 ? negotiated_fps_ : requested_fps_;
    destroyPipeline(true);
    std::string error;
    if (!openPipeline("jpegdec", fps, error)) {
        std::cerr << "[Decoder] Software decoder fallback failed: " << error << "\n";
        return false;
    }
    GstreamerFrame first;
    if (!pullFrame(first, 3000, error)) {
        std::cerr << "[Decoder] Software decoder fallback produced no valid frame: " << error << "\n";
        destroyPipeline(true);
        return false;
    }
    hardware_decoder_ = false;
    decoder_ = "jpegdec";
    std::cerr << "[Decoder] Runtime fallback to software decoder succeeded\n";
    return true;
}

bool GstreamerMjpegCamera::open(int camera_index, const std::string& device,
                                int width, int height, int fps, int focus, int zoom,
                                int max_frames) {
    close();
    device_ = device.empty() ? ("/dev/video" + std::to_string(camera_index)) : device;
    width_ = width; height_ = height; requested_fps_ = fps;
    focus_ = focus; zoom_ = zoom; max_frames_ = max_frames;
    runtime_fallback_attempted_ = false;
    zero_copy_reported_ = false;
    copy_reported_ = false;
    if (width_ <= 0 || height_ <= 0 || requested_fps_ <= 0) {
        std::cerr << "Invalid camera geometry or fps\n";
        return false;
    }
    if (!configureControls()) return false;
    std::string m2m_description;
    const bool force_software = std::getenv("SPACEMIT_FORCE_SOFTWARE_DECODER") != nullptr;
    const bool hardware_available = !force_software && findV4l2M2mDecoder(m2m_description);
    if (force_software) {
        std::cerr << "[Decoder] Software decoder forced by SPACEMIT_FORCE_SOFTWARE_DECODER\n";
    } else if (hardware_available) {
        std::cerr << "[Decoder] Hardware decoder candidate detected: " << m2m_description << "; trying spacemitdec\n";
    } else {
        std::cerr << "[Decoder] Hardware decoder unavailable; will use software decoder jpegdec + videoconvert\n";
    }

    const bool c920_720p25 = width_ == 1280 && height_ == 720 && requested_fps_ == 25;
    if (c920_720p25)
        std::cerr << "Camera mode 1280x720@25 maps to advertised 24 FPS; trying 24 FPS directly\n";
    const int candidates[] = {c920_720p25 ? 24 : requested_fps_,
                              (!c920_720p25 && requested_fps_ == 25) ? 24 : 0};
    std::string error;
    if (hardware_available) {
        for (const int candidate : candidates) {
            if (candidate <= 0) continue;
            if (!openPipeline("spacemitdec", candidate, error)) continue;
            GstreamerFrame first;
            if (pullFrame(first, 3000, error)) {
                std::cerr << "GStreamer camera opened: " << device_ << " " << width_ << "x"
                          << height_ << "@" << negotiated_fps_ << " decoder=spacemitdec output=NV12\n"
                          << "[Decoder] Using hardware decoder: spacemitdec (V4L2 M2M/MJPEG -> NV12)\n";
                return true;
            }
            destroyPipeline(true);
        }
        std::cerr << "[Decoder] Hardware decoder spacemitdec failed"
                  << (error.empty() ? "" : ": " + error)
                  << "; falling back to software decoder jpegdec\n";
    } else {
        std::cerr << "[Decoder] Using software fallback because no usable hardware decoder was detected\n";
    }
    for (const int candidate : candidates) {
        if (candidate <= 0) continue;
        if (!openPipeline("jpegdec", candidate, error)) continue;
        GstreamerFrame first;
        if (pullFrame(first, 3000, error)) {
            std::cerr << "GStreamer camera opened: " << device_ << " " << width_ << "x"
                      << height_ << "@" << negotiated_fps_ << " decoder=jpegdec output=NV12\n"
                      << "[Decoder] Using software decoder: jpegdec"
                      << (hardware_available ? " (hardware fallback)" : "") << "\n";
            return true;
        }
        destroyPipeline(true);
    }
    std::cerr << "Could not open a GStreamer MJPEG pipeline for " << device_ << "\n";
    return false;
}

bool GstreamerMjpegCamera::read(GstreamerFrame& frame, int timeout_ms) {
    std::string error;
    static thread_local int consecutive_errors = 0;
    if (pullFrame(frame, timeout_ms, error)) {
        consecutive_errors = 0;
        return true;
    }
    if (eos_) return false;
    if (hardware_decoder_ && ++consecutive_errors >= 3) {
        consecutive_errors = 0;
        if (switchToSoftware()) {
            if (pullFrame(frame, timeout_ms, error)) return true;
        }
    }
    return false;
}

bool GstreamerMjpegCamera::read(cv::Mat& nv12, int timeout_ms) {
    GstreamerFrame frame;
    if (!read(frame, timeout_ms)) return false;
    // A bare cv::Mat has no owner for the mapped GstBuffer, so this legacy
    // overload intentionally detaches the data before returning.
    nv12 = frame.nv12.clone();
    return true;
}

void GstreamerMjpegCamera::destroyPipeline(bool send_eos) {
    if (!gst_pipeline_) return;
    GstBus* bus = gst_element_get_bus(gst_pipeline_);
    if (send_eos && !eos_) {
        gst_element_send_event(gst_pipeline_, gst_event_new_eos());
        GstMessage* drained = gst_bus_timed_pop_filtered(
            bus, 2 * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (drained) gst_message_unref(drained);
    }
    gst_element_set_state(gst_pipeline_, GST_STATE_NULL);
    gst_element_get_state(gst_pipeline_, nullptr, nullptr, 2 * GST_SECOND);
    gst_object_unref(bus);
    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }
    gst_object_unref(gst_pipeline_);
    gst_pipeline_ = nullptr;
    eos_ = false;
}

void GstreamerMjpegCamera::close() {
    // EOS -> wait for the decoder to drain -> NULL prevents appsink teardown
    // from racing the vendor spacemitdec output-buffer return path.
    destroyPipeline(true);
    pipeline_description_.clear();
    decoder_.clear();
    negotiated_fps_ = 0;
    hardware_decoder_ = false;
}

bool GstreamerMjpegCamera::isOpen() const { return gst_pipeline_ != nullptr; }
