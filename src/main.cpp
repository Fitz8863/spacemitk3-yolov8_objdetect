#include "gstreamer_camera.h"
#include "opencl_preprocess.h"
#include "yolov8_detector.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <deque>


namespace {
using Clock = std::chrono::steady_clock;
volatile sig_atomic_t g_signal_stop = 0;
void on_signal(int) { g_signal_stop = 1; }

// Bounded queue: it can keep a few frames, but never grows without bound.
// When full, the oldest frame is discarded so latency cannot accumulate.
template <typename T>
class FrameQueue {
public:
    explicit FrameQueue(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    bool push(std::shared_ptr<T> value) {
        bool dropped = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            if (queue_.size() >= capacity_) {
                queue_.pop_front();
                dropped = true;
            }
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
        return dropped;
    }

    // Return the newest pending item and discard any older pending items.
    // This is the low-latency policy for an inference/display consumer.
    bool popLatest(std::shared_ptr<T>& value, const std::atomic<bool>& abort,
                   bool prebuffer = false, size_t prebuffer_count = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto ready = [&] {
            return closed_ || abort.load() || g_signal_stop != 0 ||
                   (!queue_.empty() && (!prebuffer || queue_.size() >= prebuffer_count));
        };
        // Poll periodically so Ctrl-C can be observed even while a stage is
        // waiting for a frame. A signal handler cannot safely notify this CV;
        // polling also checks g_signal_stop so Ctrl-C wakes every stage.
        // Keep waiting after a spurious/periodic timeout; returning false here
        // would make the consumer exit whenever a frame takes longer than the
        // polling interval to arrive.
        while (!ready()) {
            cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        // If the producer closed normally, drain any frames already queued.
        // On abort, also consume an already available newest frame, then stop
        // once the queue is empty.
        if ((abort.load() || g_signal_stop != 0) && queue_.empty()) return false;
        if (queue_.empty()) return false;
        value = std::move(queue_.back());
        if (queue_.size() > 1) dropped_pending_ += queue_.size() - 1;
        queue_.clear();
        return static_cast<bool>(value);
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t takeDroppedPending() {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = dropped_pending_;
        dropped_pending_ = 0;
        return n;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<T>> queue_;
    size_t capacity_;
    size_t dropped_pending_ = 0;
    bool closed_ = false;
};

struct PreparedFrame {
    uint64_t id = 0;
    int width = 0;
    int height = 0;
    OpenClPreprocessor::Result prep;
    std::shared_ptr<cv::Mat> nv12;
};

struct InferenceResult {
    uint64_t id = 0;
    int width = 0;
    int height = 0;
    std::shared_ptr<cv::Mat> nv12;
    std::vector<Detection> detections;
};

struct Args {
    std::string model = "models/yolov8_best.q.onnx";
    int camera = 1;
    std::string device;
    int width = 1280, height = 720, fps = 25;
    int intra_threads = 1;
    std::string ep_affinity;
    int focus = 0, zoom = 181;
    float conf = 0.25f;
    size_t queue_depth = 3;
    bool no_display = false;
    int max_frames = 0;
    bool self_test = false;
    std::string dump_input;
};

struct Stats {
    std::atomic<uint64_t> prepared{0};
    std::atomic<uint64_t> inferred{0};
    std::atomic<uint64_t> presented{0};
    std::atomic<uint64_t> dropped_pre{0};
    std::atomic<uint64_t> dropped_result{0};
    std::atomic<uint64_t> detected_frames{0};
    std::atomic<uint64_t> detections{0};
    std::mutex timing_mutex;
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double display_ms = 0.0;

    void addPre(double v) { std::lock_guard<std::mutex> l(timing_mutex); pre_ms += v; }
    void addInfer(double v) { std::lock_guard<std::mutex> l(timing_mutex); infer_ms += v; }
    void addDisplay(double v) { std::lock_guard<std::mutex> l(timing_mutex); display_ms += v; }
    void averages(double& p, double& i, double& d) {
        std::lock_guard<std::mutex> l(timing_mutex);
        p = prepared ? pre_ms / static_cast<double>(prepared.load()) : 0.0;
        i = inferred ? infer_ms / static_cast<double>(inferred.load()) : 0.0;
        d = presented ? display_ms / static_cast<double>(presented.load()) : 0.0;
    }
};

static void usage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --model PATH       ONNX model\n"
              << "  --camera N         V4L2 camera index (default 1)\n"
              << "  --device PATH      explicit V4L2 node, overrides --camera\n"
              << "  --width N --height N --fps N\n"
              << "  --conf FLOAT       confidence threshold (default 0.25)\n"
              << "  --queue-depth N    keep up to N frames per pipeline queue (default 3)\n"
              << "  --focus N          fixed manual focus (default 0; -1 unchanged)\n"
              << "  --zoom N           zoom absolute value (default 181; -1 unchanged)\n"
              << "  --intra-threads N  SpaceMIT EP threads (default 1)\n"
              << "  --ep-affinity LIST bind EP threads to cores, e.g. 8;9;10;11\n"
              << "  --no-display       run pipeline without window\n"
              << "  --max-frames N     stop after N frames enter preprocess (0=unlimited)\n"
              << "  --dump-input PATH  dump first preprocessed tensor as float32\n"
              << "  --self-test        initialize OpenCL GPU and model, run one inference\n";
}

static bool parse(int argc, char** argv, Args& a) {
    auto need = [&](int& i) -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--help" || k == "-h") { usage(argv[0]); return false; }
        const char* v = nullptr;
        if (k == "--model" && (v = need(i))) a.model = v;
        else if (k == "--camera" && (v = need(i))) a.camera = std::stoi(v);
        else if (k == "--device" && (v = need(i))) a.device = v;
        else if (k == "--width" && (v = need(i))) a.width = std::stoi(v);
        else if (k == "--height" && (v = need(i))) a.height = std::stoi(v);
        else if (k == "--fps" && (v = need(i))) a.fps = std::stoi(v);
        else if (k == "--conf" && (v = need(i))) a.conf = std::stof(v);
        else if (k == "--queue-depth" && (v = need(i))) a.queue_depth = static_cast<size_t>(std::stoul(v));
        else if (k == "--focus" && (v = need(i))) a.focus = std::stoi(v);
        else if (k == "--zoom" && (v = need(i))) a.zoom = std::stoi(v);
        else if (k == "--intra-threads" && (v = need(i))) a.intra_threads = std::stoi(v);
        else if (k == "--ep-affinity" && (v = need(i))) a.ep_affinity = v;
        else if (k == "--max-frames" && (v = need(i))) a.max_frames = std::stoi(v);
        else if (k == "--no-display") a.no_display = true;
        else if (k == "--dump-input" && (v = need(i))) a.dump_input = v;
        else if (k == "--self-test") a.self_test = true;
        else { std::cerr << "Unknown or incomplete option: " << k << "\n"; usage(argv[0]); return false; }
    }
    a.queue_depth = std::clamp<size_t>(a.queue_depth, 1, 8);
    if (a.intra_threads < 1) {
        std::cerr << "--intra-threads must be >= 1\n";
        return false;
    }
    if (!a.ep_affinity.empty()) {
        std::size_t count = 1;
        for (char c : a.ep_affinity) {
            if (c == ';') ++count;
            else if (c < '0' || c > '9') {
                std::cerr << "--ep-affinity must be a semicolon-separated list of core IDs, "
                             "for example 8;9;10;11\n";
                return false;
            }
        }
        if (a.ep_affinity.front() == ';' || a.ep_affinity.back() == ';' ||
            a.ep_affinity.find(";;") != std::string::npos) {
            std::cerr << "--ep-affinity contains an empty core ID\n";
            return false;
        }
        if (count != static_cast<std::size_t>(a.intra_threads)) {
            std::cerr << "--ep-affinity contains " << count
                      << " core IDs, but --intra-threads is " << a.intra_threads
                      << "; the counts must match\n";
            return false;
        }
    }
    return true;
}

// Ultralytics-style vivid palette. OpenCV colors are BGR.
static const std::array<cv::Scalar, 6> kClassColors = {
    cv::Scalar(56, 56, 255),    // red       #FF3838
    cv::Scalar(151, 157, 255),  // light red #FF9D97
    cv::Scalar(31, 112, 255),   // orange    #FF701F
    cv::Scalar(29, 178, 255),   // amber     #FFB21D
    cv::Scalar(49, 210, 207),   // yellow    #CFD231
    cv::Scalar(10, 249, 72),    // green     #48F90A
};

static void draw_detections(cv::Mat& bgr, const std::vector<Detection>& ds) {
    for (const auto& d : ds) {
        cv::Rect r(static_cast<int>(d.x1), static_cast<int>(d.y1),
                   std::max(1, static_cast<int>(d.x2 - d.x1)),
                   std::max(1, static_cast<int>(d.y2 - d.y1)));
        const size_t class_index =
            static_cast<size_t>(std::max(0, d.class_id)) % kClassColors.size();
        const cv::Scalar& color = kClassColors[class_index];

        cv::rectangle(bgr, r, color, 2, cv::LINE_AA);
        std::ostringstream label;
        label << (d.class_id + 1) << " " << std::fixed << std::setprecision(2)
              << d.confidence;
        int baseline = 0;
        const auto size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                          .6, 1, &baseline);
        const int label_top = std::max(0, r.y - size.height - baseline - 8);
        const int label_bottom = std::max(size.height + baseline + 2, r.y);
        cv::rectangle(bgr, cv::Point(r.x, label_top),
                      cv::Point(r.x + size.width + 6, label_bottom),
                      color, cv::FILLED);

        // Pick a contrasting label color based on the palette luminance.
        const int brightness = static_cast<int>(0.114 * color[0] +
                                                0.587 * color[1] +
                                                0.299 * color[2]);
        const cv::Scalar text_color = brightness > 160
            ? cv::Scalar(0, 0, 0)
            : cv::Scalar(255, 255, 255);
        cv::putText(bgr, label.str(), cv::Point(r.x + 3, label_bottom - 4),
                    cv::FONT_HERSHEY_SIMPLEX, .6, text_color, 1, cv::LINE_AA);
    }
}


} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Args a;
    if (!parse(argc, argv, a)) return argc > 1 ? 2 : 0;
    if (!std::filesystem::exists(a.model)) {
        std::cerr << "Model not found: " << a.model << "\n";
        return 2;
    }

    OpenClPreprocessor pre;
    if (!pre.init()) return 4;
    Yolov8Detector detector;
    if (!detector.init(a.model, a.intra_threads, a.ep_affinity)) return 5;
    if (a.self_test) {
        try {
            // Exercise the actual OpenCL NV12 -> tensor path as well as the
            // SpaceMIT EP model path. This catches kernel/image/queue errors
            // that a model-only zero-tensor test cannot detect.
            constexpr int synthetic_width = 1280;
            constexpr int synthetic_height = 720;
            cv::Mat synthetic_nv12(synthetic_height * 3 / 2, synthetic_width,
                                   CV_8UC1, cv::Scalar(128));
            const auto prep_result = pre.preprocess(synthetic_nv12);
            if (!prep_result.data || prep_result.data->size() != 3 * 640 * 640) {
                throw std::runtime_error("OpenCL self-test returned an invalid tensor");
            }
            for (float value : *prep_result.data) {
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                    throw std::runtime_error("OpenCL self-test returned invalid tensor values");
                }
            }
            const auto ds = detector.infer(prep_result.data->data(), prep_result.data->size(),
                                           a.conf, prep_result.scale, prep_result.pad_x,
                                           prep_result.pad_y, synthetic_width, synthetic_height);
            std::cout << "Self-test passed: OpenCL preprocess " << prep_result.ms
                      << " ms, " << ds.size() << " detections.\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Self-test failed: " << e.what() << "\n";
            return 6;
        }
    }

    GstreamerMjpegCamera camera;
    if (!camera.open(a.camera, a.device, a.width, a.height, a.fps,
                     a.focus, a.zoom)) {
        std::cerr << "Camera open failed.\n";
        return 3;
    }
    if (!a.no_display) {
        try {
            cv::namedWindow("yolov8-k3", cv::WINDOW_NORMAL);
            cv::resizeWindow("yolov8-k3", a.width, a.height);
        } catch (const cv::Exception& e) {
            std::cerr << "Display initialization failed: " << e.what() << "\n";
            camera.close();
            return 7;
        }
    }

    std::atomic<bool> abort{false};
    std::atomic<bool> preprocess_done{false};
    std::atomic<bool> inference_done{false};
    FrameQueue<PreparedFrame> prepared_queue(a.queue_depth);
    FrameQueue<InferenceResult> result_queue(a.queue_depth);
    Stats stats;
    const auto start = Clock::now();

    // Thread 1: OpenCV VideoCapture -> GStreamer -> K3 spacemitdec -> NV12,
    // followed by OpenCL GPU preprocessing. appsink keeps only the newest frame.
    std::thread preprocess_thread([&] {
        uint64_t id = 0;
        int timeout_count = 0;
        while (!abort.load() &&
               (a.max_frames <= 0 || static_cast<int>(id) < a.max_frames)) {
            auto nv12 = std::make_shared<cv::Mat>();
            if (!camera.read(*nv12, 1000)) {
                if (abort.load()) break;
                if (++timeout_count >= 10) {
                    std::cerr << "Camera read timeout/error in GStreamer stage\n";
                    abort.store(true);
                    break;
                }
                continue;
            }
            timeout_count = 0;
            auto packet = std::make_shared<PreparedFrame>();
            packet->id = id++;
            packet->width = nv12->cols;
            packet->height = (nv12->rows * 2) / 3;
            packet->nv12 = std::move(nv12);
            try {
                const auto t0 = Clock::now();
                packet->prep = pre.preprocess(*packet->nv12);
                if (!a.dump_input.empty() && packet->id == 0) {
                    std::ofstream dump(a.dump_input, std::ios::binary);
                    if (!dump) throw std::runtime_error("cannot open --dump-input path");
                    dump.write(reinterpret_cast<const char*>(packet->prep.data->data()),
                               static_cast<std::streamsize>(packet->prep.data->size() * sizeof(float)));
                }
                stats.addPre(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                if (prepared_queue.push(std::move(packet))) stats.dropped_pre.fetch_add(1);
                stats.dropped_pre.fetch_add(prepared_queue.takeDroppedPending());
                stats.prepared.fetch_add(1);
            } catch (const std::exception& e) {
                std::cerr << "Preprocess stage failed: " << e.what() << "\n";
                abort.store(true);
                break;
            }
        }
        preprocess_done.store(true);
        prepared_queue.close();
    });

    // Thread 2: one SpaceMIT EP session; only this thread touches detector.
    std::thread inference_thread([&] {
        uint64_t id = 0;
        std::shared_ptr<PreparedFrame> packet;
        while (prepared_queue.popLatest(packet, abort)) {
            if (!packet) continue;
            try {
                const auto t0 = Clock::now();
                auto result = std::make_shared<InferenceResult>();
                result->id = packet->id;
                result->width = packet->width;
                result->height = packet->height;
                result->nv12 = packet->nv12;
                result->detections = detector.infer(packet->prep.data->data(), packet->prep.data->size(), a.conf,
                                                     packet->prep.scale, packet->prep.pad_x, packet->prep.pad_y,
                                                     packet->width, packet->height);
                stats.addInfer(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                stats.inferred.fetch_add(1);
                if (!result->detections.empty()) stats.detected_frames.fetch_add(1);
                stats.detections.fetch_add(result->detections.size());
                if (result_queue.push(std::move(result))) stats.dropped_result.fetch_add(1);
                stats.dropped_result.fetch_add(result_queue.takeDroppedPending());
                ++id;
            } catch (const std::exception& e) {
                std::cerr << "Inference stage failed: " << e.what() << "\n";
                abort.store(true);
                break;
            }
        }
        inference_done.store(true);
        result_queue.close();
    });

    // Thread 3 is the display stage logically; it runs on the main/UI thread
    // because OpenCV HighGUI on this board must own the X11 event loop here.
    bool first_display = true;
    uint64_t shown = 0;
    auto last_report = start;
    while (true) {
        if (g_signal_stop) abort.store(true);
        std::shared_ptr<InferenceResult> item;
        if (!result_queue.popLatest(item, abort, first_display, a.queue_depth)) {
            if (g_signal_stop) abort.store(true);
            if (inference_done.load()) break;
            if (abort.load()) break;
            continue;
        }
        first_display = false;
        if (!item) continue;

        const auto t0 = Clock::now();
        if (!a.no_display) {
            cv::Mat bgr;
            if (item->nv12 && !item->nv12->empty()) {
                cv::cvtColor(*item->nv12, bgr, cv::COLOR_YUV2BGR_NV12);
                draw_detections(bgr, item->detections);
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                const double fps = elapsed > 0.0 ? shown / elapsed : 0.0;
                cv::putText(bgr, "DISPLAY " + std::to_string(static_cast<int>(fps)) +
                                      " FPS  DET " + std::to_string(item->detections.size()),
                            {10, 28}, cv::FONT_HERSHEY_SIMPLEX, .75,
                            {0, 255, 255}, 2, cv::LINE_AA);
                cv::imshow("yolov8-k3", bgr);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 27) abort.store(true);
            } else {
                std::cerr << "Display frame transfer/YUV conversion failed\n";
            }
        }
        stats.addDisplay(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        stats.presented.fetch_add(1);
        ++shown;

        const auto now = Clock::now();
        if (now - last_report >= std::chrono::seconds(2)) {
            const double elapsed = std::chrono::duration<double>(now - start).count();
            double p = 0.0, i = 0.0, d = 0.0;
            stats.averages(p, i, d);
            std::cout << "pipeline prepared=" << stats.prepared.load()
                      << " infer=" << stats.inferred.load()
                      << " display=" << stats.presented.load()
                      << " fps_pre=" << stats.prepared.load() / std::max(.001, elapsed)
                      << " fps_infer=" << stats.inferred.load() / std::max(.001, elapsed)
                      << " fps_display=" << stats.presented.load() / std::max(.001, elapsed)
                      << " dropped_pre=" << stats.dropped_pre.load()
                      << " dropped_result=" << stats.dropped_result.load()
                      << " detected_frames=" << stats.detected_frames.load()
                      << " detections=" << stats.detections.load()
                      << " pre_ms=" << p << " infer_ms=" << i << " display_ms=" << d << "\n";
            last_report = now;
        }
        if (abort.load()) break;
    }

    // Ctrl-C/q/exception: stop capture and wake all stages. Normal max-frames
    // drains both queues before reaching this point.
    if (abort.load()) {
        prepared_queue.close();
        result_queue.close();
        camera.close();
    }
    if (preprocess_thread.joinable()) preprocess_thread.join();
    if (inference_thread.joinable()) inference_thread.join();
    camera.close();
    if (!a.no_display) cv::destroyAllWindows();

    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    double p = 0.0, i = 0.0, d = 0.0;
    stats.averages(p, i, d);
    std::cout << "Done. prepared=" << stats.prepared.load()
              << " infer=" << stats.inferred.load()
              << " display=" << stats.presented.load()
              << " elapsed_s=" << elapsed
              << " fps_pre=" << stats.prepared.load() / std::max(.001, elapsed)
              << " fps_infer=" << stats.inferred.load() / std::max(.001, elapsed)
              << " fps_display=" << stats.presented.load() / std::max(.001, elapsed)
              << " dropped_pre=" << stats.dropped_pre.load()
              << " dropped_result=" << stats.dropped_result.load()
              << " detected_frames=" << stats.detected_frames.load()
              << " detections=" << stats.detections.load()
              << " pre_ms=" << p << " infer_ms=" << i << " display_ms=" << d << "\n";
    return 0;
}
