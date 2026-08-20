#include "opencl_preprocess.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* kKernel = R"CLC(
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// Input planes are R/UNORM images. NV12's interleaved UV plane is split on
// the host because this keeps the kernel portable across the K3 OpenCL stack.
__kernel void yuv420_to_yolo(read_only image2d_t yimg, read_only image2d_t uimg,
                             read_only image2d_t vimg, __global float *out,
                             int in_w, int in_h, int out_w, int out_h,
                             int resized_w, int resized_h,
                             float x_scale, float y_scale,
                             int pad_x, int pad_y) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= out_w || y >= out_h) return;

    int rx = x - pad_x;
    int ry = y - pad_y;
    float sx = (((float)rx + 0.5f) * x_scale) - 0.5f;
    float sy = (((float)ry + 0.5f) * y_scale) - 0.5f;
    float r, g, b;
    if (rx < 0 || ry < 0 || rx >= resized_w || ry >= resized_h) {
        r = g = b = 114.0f / 255.0f;
    } else {
        float yy = read_imagef(yimg, smp, (float2)(sx + 0.5f, sy + 0.5f)).x * 255.0f;
        float u = read_imagef(uimg, smp, (float2)(sx * 0.5f + 0.5f, sy * 0.5f + 0.5f)).x * 255.0f;
        float v = read_imagef(vimg, smp, (float2)(sx * 0.5f + 0.5f, sy * 0.5f + 0.5f)).x * 255.0f;
        float c = yy - 16.0f, d = u - 128.0f, e = v - 128.0f;
        r = clamp((1.16438356f * c + 1.79274107f * e) / 255.0f, 0.0f, 1.0f);
        g = clamp((1.16438356f * c - 0.21324861f * d - 0.53290933f * e) / 255.0f, 0.0f, 1.0f);
        b = clamp((1.16438356f * c + 2.11240179f * d) / 255.0f, 0.0f, 1.0f);
    }

    int plane = out_w * out_h;
    int i = y * out_w + x;
    out[i] = r;
    out[i + plane] = g;
    out[i + 2 * plane] = b;
}
)CLC";

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
    const float scale = std::max(src_width / static_cast<float>(dst_width),
                                 src_height / static_cast<float>(dst_height));
    const int resized_width = even_at_least_two(
        static_cast<int>(std::lround(src_width / scale)));
    const int resized_height = even_at_least_two(
        static_cast<int>(std::lround(src_height / scale)));
    if (resized_width > dst_width || resized_height > dst_height) {
        throw std::runtime_error("letterbox resized image exceeds destination");
    }
    const int pad_x = (dst_width - resized_width) / 2;
    const int pad_y = (dst_height - resized_height) / 2;
    // NV12 chroma has 2x2 sampling. Keeping the letterbox origin even makes
    // the luma and UV planes use the same phase, matching the RVV path.
    if ((pad_x & 1) || (pad_y & 1)) {
        throw std::runtime_error("NV12 letterbox offsets must be even");
    }
    return {scale, resized_width, resized_height, pad_x, pad_y};
}
}  // namespace

struct OpenClPreprocessor::Impl {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    int out_w = 640;
    int out_h = 640;
    int in_w = 0;
    int in_h = 0;
    size_t y_size = 0;
    size_t uv_size = 0;
    size_t output_size = 0;

    std::vector<uint8_t> y_host;
    std::vector<uint8_t> u_host;
    std::vector<uint8_t> v_host;
    cl_mem y_buffer = nullptr;
    cl_mem u_buffer = nullptr;
    cl_mem v_buffer = nullptr;
    cl_mem y_image = nullptr;
    cl_mem u_image = nullptr;
    cl_mem v_image = nullptr;
    cl_mem output_buffer = nullptr;

    ~Impl() {
        release_io();
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    static void check(cl_int error, const char* what) {
        if (error != CL_SUCCESS) {
            throw std::runtime_error(std::string(what) + ": " + std::to_string(error));
        }
    }

    static std::string info_string(cl_platform_id id, cl_platform_info param) {
        size_t size = 0;
        check(clGetPlatformInfo(id, param, 0, nullptr, &size), "clGetPlatformInfo size");
        std::string result(size, '\0');
        check(clGetPlatformInfo(id, param, result.size(), result.data(), nullptr),
              "clGetPlatformInfo");
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    }

    void release_io() {
        if (output_buffer) clReleaseMemObject(output_buffer);
        if (v_image) clReleaseMemObject(v_image);
        if (u_image) clReleaseMemObject(u_image);
        if (y_image) clReleaseMemObject(y_image);
        if (v_buffer) clReleaseMemObject(v_buffer);
        if (u_buffer) clReleaseMemObject(u_buffer);
        if (y_buffer) clReleaseMemObject(y_buffer);
        output_buffer = nullptr;
        v_image = nullptr;
        u_image = nullptr;
        y_image = nullptr;
        v_buffer = nullptr;
        u_buffer = nullptr;
        y_buffer = nullptr;
        y_host.clear();
        u_host.clear();
        v_host.clear();
        in_w = 0;
        in_h = 0;
        y_size = 0;
        uv_size = 0;
        output_size = 0;
    }

    void init() {
        cl_uint platform_count = 0;
        check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
        if (platform_count == 0) throw std::runtime_error("no OpenCL platform found");
        std::vector<cl_platform_id> platforms(platform_count);
        check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_uint device_count = 0;
        for (cl_platform_id candidate : platforms) {
            device_count = 0;
            if (clGetDeviceIDs(candidate, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) == CL_SUCCESS &&
                device_count > 0) {
                platform = candidate;
                std::vector<cl_device_id> devices(device_count);
                check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                                     devices.data(), nullptr), "clGetDeviceIDs GPU");
                device = devices.front();
                break;
            }
        }
        if (!device) throw std::runtime_error("no OpenCL GPU device found");

        char name[256] = {};
        check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr),
              "clGetDeviceInfo name");
        std::cout << "OpenCL GPU: " << name
                  << " (platform: " << info_string(platform, CL_PLATFORM_NAME) << ")\n";

        cl_int error = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        check(error, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &error);
        check(error, "clCreateCommandQueue");

        const size_t length = std::strlen(kKernel);
        program = clCreateProgramWithSource(context, 1, &kKernel, &length, &error);
        check(error, "clCreateProgramWithSource");
        error = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  log.size(), log.data(), nullptr);
            throw std::runtime_error("clBuildProgram: " + std::to_string(error) + "\n" + log);
        }
        kernel = clCreateKernel(program, "yuv420_to_yolo", &error);
        check(error, "clCreateKernel");
    }

    void ensure_io(int width, int height) {
        if (width == in_w && height == in_h && y_image && u_image && v_image && output_buffer) {
            return;
        }
        release_io();
        in_w = width;
        in_h = height;
        y_size = static_cast<size_t>(width) * height;
        uv_size = y_size / 4;
        output_size = static_cast<size_t>(3) * out_w * out_h * sizeof(float);
        y_host.resize(y_size);
        u_host.resize(uv_size);
        v_host.resize(uv_size);

        try {
            cl_int error = CL_SUCCESS;
            y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size, nullptr, &error);
            check(error, "clCreateBuffer Y");
            u_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size, nullptr, &error);
            check(error, "clCreateBuffer U");
            v_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size, nullptr, &error);
            check(error, "clCreateBuffer V");

            const cl_image_format format{CL_R, CL_UNORM_INT8};
            auto create_image = [&](cl_mem buffer, int image_width, int image_height,
                                    const char* what) {
                cl_image_desc desc{};
                desc.image_type = CL_MEM_OBJECT_IMAGE2D;
                desc.image_width = static_cast<size_t>(image_width);
                desc.image_height = static_cast<size_t>(image_height);
                desc.image_row_pitch = static_cast<size_t>(image_width);
                desc.buffer = buffer;
                cl_mem image = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc,
                                              nullptr, &error);
                check(error, what);
                return image;
            };
            y_image = create_image(y_buffer, width, height, "clCreateImage Y");
            u_image = create_image(u_buffer, width / 2, height / 2, "clCreateImage U");
            v_image = create_image(v_buffer, width / 2, height / 2, "clCreateImage V");
            output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &error);
            check(error, "clCreateBuffer output");
        } catch (...) {
            release_io();
            throw;
        }
    }
};

OpenClPreprocessor::OpenClPreprocessor() = default;
OpenClPreprocessor::~OpenClPreprocessor() = default;

bool OpenClPreprocessor::init(int out_width, int out_height) {
    try {
        if (out_width <= 0 || out_height <= 0 || (out_width & 1) || (out_height & 1)) {
            throw std::runtime_error("OpenCL output dimensions must be positive and even");
        }
        impl_ = std::make_unique<Impl>();
        impl_->out_w = out_width;
        impl_->out_h = out_height;
        impl_->init();
        char name[256] = {};
        Impl::check(clGetDeviceInfo(impl_->device, CL_DEVICE_NAME, sizeof(name), name, nullptr),
                    "clGetDeviceInfo name");
        device_name_ = name;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL init failed: " << e.what() << "\n";
        impl_.reset();
        return false;
    }
}

OpenClPreprocessor::Result OpenClPreprocessor::preprocess(const cv::Mat& nv12) {
    if (!impl_) throw std::runtime_error("OpenCL preprocessor not initialized");
    if (nv12.empty() || nv12.type() != CV_8UC1 || nv12.rows * 2 % 3 != 0) {
        throw std::runtime_error("OpenCL NV12 input must be non-empty CV_8UC1 with height 3/2*h");
    }

    const int width = nv12.cols;
    const int height = nv12.rows * 2 / 3;
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) {
        throw std::runtime_error("OpenCL NV12 input has invalid even dimensions");
    }
    if (nv12.step < static_cast<size_t>(width)) {
        throw std::runtime_error("OpenCL NV12 input stride is smaller than width");
    }

    const auto t0 = std::chrono::steady_clock::now();
    const LetterboxGeometry geometry = calculate_geometry(width, height, impl_->out_w, impl_->out_h);
    impl_->ensure_io(width, height);

    for (int row = 0; row < height; ++row) {
        std::memcpy(impl_->y_host.data() + static_cast<size_t>(row) * width,
                    nv12.ptr(row), static_cast<size_t>(width));
    }
    const uint8_t* uv_src = nv12.ptr(height);
    for (int row = 0; row < height / 2; ++row) {
        const uint8_t* row_src = uv_src + static_cast<size_t>(row) * nv12.step;
        for (int col = 0, j = 0; col < width; col += 2, ++j) {
            impl_->u_host[static_cast<size_t>(row) * (width / 2) + j] = row_src[col];
            impl_->v_host[static_cast<size_t>(row) * (width / 2) + j] = row_src[col + 1];
        }
    }

    cl_int error = clEnqueueWriteBuffer(impl_->queue, impl_->y_buffer, CL_FALSE, 0,
                                        impl_->y_size, impl_->y_host.data(), 0, nullptr, nullptr);
    Impl::check(error, "clEnqueueWriteBuffer Y");
    error = clEnqueueWriteBuffer(impl_->queue, impl_->u_buffer, CL_FALSE, 0,
                                 impl_->uv_size, impl_->u_host.data(), 0, nullptr, nullptr);
    Impl::check(error, "clEnqueueWriteBuffer U");
    error = clEnqueueWriteBuffer(impl_->queue, impl_->v_buffer, CL_FALSE, 0,
                                 impl_->uv_size, impl_->v_host.data(), 0, nullptr, nullptr);
    Impl::check(error, "clEnqueueWriteBuffer V");

    auto set_arg = [&](cl_uint index, size_t size, const void* value, const char* what) {
        Impl::check(clSetKernelArg(impl_->kernel, index, size, value), what);
    };
    set_arg(0, sizeof(cl_mem), &impl_->y_image, "clSetKernelArg Y");
    set_arg(1, sizeof(cl_mem), &impl_->u_image, "clSetKernelArg U");
    set_arg(2, sizeof(cl_mem), &impl_->v_image, "clSetKernelArg V");
    set_arg(3, sizeof(cl_mem), &impl_->output_buffer, "clSetKernelArg output");
    set_arg(4, sizeof(int), &width, "clSetKernelArg input width");
    set_arg(5, sizeof(int), &height, "clSetKernelArg input height");
    set_arg(6, sizeof(int), &impl_->out_w, "clSetKernelArg output width");
    set_arg(7, sizeof(int), &impl_->out_h, "clSetKernelArg output height");
    set_arg(8, sizeof(int), &geometry.resized_width, "clSetKernelArg resized width");
    set_arg(9, sizeof(int), &geometry.resized_height, "clSetKernelArg resized height");
    const float x_scale = width / static_cast<float>(geometry.resized_width);
    const float y_scale = height / static_cast<float>(geometry.resized_height);
    set_arg(10, sizeof(float), &x_scale, "clSetKernelArg x scale");
    set_arg(11, sizeof(float), &y_scale, "clSetKernelArg y scale");
    set_arg(12, sizeof(int), &geometry.pad_x, "clSetKernelArg pad x");
    set_arg(13, sizeof(int), &geometry.pad_y, "clSetKernelArg pad y");

    const size_t global[] = {static_cast<size_t>(impl_->out_w),
                             static_cast<size_t>(impl_->out_h)};
    Impl::check(clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 2, nullptr, global,
                                       nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel");

    auto output = std::make_shared<std::vector<float>>(
        static_cast<size_t>(3) * impl_->out_w * impl_->out_h);
    // A blocking read waits for all earlier commands in this in-order queue;
    // an extra clFinish here only adds synchronization overhead.
    Impl::check(clEnqueueReadBuffer(impl_->queue, impl_->output_buffer, CL_TRUE, 0,
                                    impl_->output_size, output->data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer output");

    Result result;
    result.data = std::move(output);
    result.scale = geometry.scale;
    result.pad_x = geometry.pad_x;
    result.pad_y = geometry.pad_y;
    result.ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}
