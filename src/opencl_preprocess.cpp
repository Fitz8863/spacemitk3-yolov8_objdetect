#include "opencl_preprocess.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* kKernel = R"CLC(
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// Input planes are R/UNORM images. The NV12 host path splits interleaved UV
// into U and V before creating these temporary images, so this kernel is also
// shared with the former DRM-PRIME YUV420 path.
__kernel void yuv420_to_yolo(read_only image2d_t yimg, read_only image2d_t uimg,
                             read_only image2d_t vimg, __global float *out,
                             int in_w, int in_h, int out_w, int out_h,
                             float scale, int pad_x, int pad_y) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= out_w || y >= out_h) return;
    float sx = (((float)x + 0.5f - (float)pad_x) * scale) - 0.5f;
    float sy = (((float)y + 0.5f - (float)pad_y) * scale) - 0.5f;
    float r, g, b;
    if (sx < -0.5f || sy < -0.5f || sx > (float)in_w - 0.5f || sy > (float)in_h - 0.5f) {
        r = g = b = 114.0f / 255.0f;
    } else {
        float yy = read_imagef(yimg, smp, (float2)(sx + 0.5f, sy + 0.5f)).x * 255.0f;
        float u = read_imagef(uimg, smp, (float2)(sx * 0.5f + 0.5f, sy * 0.5f + 0.5f)).x * 255.0f;
        float v = read_imagef(vimg, smp, (float2)(sx * 0.5f + 0.5f, sy * 0.5f + 0.5f)).x * 255.0f;
        float c = yy - 16.0f, d = u - 128.0f, e = v - 128.0f;
        r = clamp((1.16438356f*c + 1.79274107f*e) / 255.0f, 0.0f, 1.0f);
        g = clamp((1.16438356f*c - 0.21324861f*d - 0.53290933f*e) / 255.0f, 0.0f, 1.0f);
        b = clamp((1.16438356f*c + 2.11240179f*d) / 255.0f, 0.0f, 1.0f);
    }
    int plane = out_w * out_h, i = y * out_w + x;
    out[i] = r; out[i + plane] = g; out[i + 2 * plane] = b;
}
)CLC";
}

struct OpenClPreprocessor::Impl {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    int out_w = 640;
    int out_h = 640;

    ~Impl() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    static void check(cl_int e, const char* what) {
        if (e != CL_SUCCESS)
            throw std::runtime_error(std::string(what) + ": " + std::to_string(e));
    }

    void init() {
        cl_int e = clGetPlatformIDs(1, &platform, nullptr);
        check(e, "clGetPlatformIDs");
        e = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        check(e, "clGetDeviceIDs GPU");

        char name[256] = {};
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "OpenCL GPU: " << name << "\n";

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &e);
        check(e, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &e);
        check(e, "clCreateCommandQueue");

        const size_t len = std::strlen(kKernel);
        program = clCreateProgramWithSource(context, 1, &kKernel, &len, &e);
        check(e, "clCreateProgramWithSource");
        e = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (e != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::string log(n, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            throw std::runtime_error("clBuildProgram: " + std::to_string(e) + "\n" + log);
        }
        kernel = clCreateKernel(program, "yuv420_to_yolo", &e);
        check(e, "clCreateKernel");
    }

    std::pair<cl_mem, cl_mem> make_image(const std::vector<uint8_t>& bytes,
                                         int width, int height) {
        cl_int e = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       bytes.size(), const_cast<uint8_t*>(bytes.data()), &e);
        check(e, "clCreateBuffer input plane");
        cl_image_format format{CL_R, CL_UNORM_INT8};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = static_cast<size_t>(width);
        desc.image_height = static_cast<size_t>(height);
        desc.image_row_pitch = static_cast<size_t>(width);
        desc.buffer = buffer;
        cl_mem image = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &e);
        if (e != CL_SUCCESS) {
            clReleaseMemObject(buffer);
            check(e, "clCreateImage input plane");
        }
        return {buffer, image};
    }
};

OpenClPreprocessor::OpenClPreprocessor() = default;
OpenClPreprocessor::~OpenClPreprocessor() = default;

bool OpenClPreprocessor::init(int out_width, int out_height) {
    try {
        impl_ = std::make_unique<Impl>();
        impl_->out_w = out_width;
        impl_->out_h = out_height;
        impl_->init();
        char name[256] = {};
        clGetDeviceInfo(impl_->device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
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
    if (nv12.empty() || nv12.type() != CV_8UC1 || nv12.rows * 2 % 3 != 0)
        throw std::runtime_error("OpenCL NV12 input must be non-empty CV_8UC1 with height 3/2*h");

    const int width = nv12.cols;
    const int height = nv12.rows * 2 / 3;
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1))
        throw std::runtime_error("OpenCL NV12 input has invalid even dimensions");
    if (nv12.step < static_cast<size_t>(width))
        throw std::runtime_error("OpenCL NV12 input stride is smaller than width");

    const auto t0 = std::chrono::steady_clock::now();
    const size_t y_bytes = static_cast<size_t>(width) * height;
    const size_t uv_bytes = y_bytes / 2;
    std::vector<uint8_t> y(y_bytes), u(uv_bytes / 2), v(uv_bytes / 2);
    for (int row = 0; row < height; ++row)
        std::memcpy(y.data() + static_cast<size_t>(row) * width, nv12.ptr(row), width);
    const uint8_t* uv_src = nv12.ptr(height);
    for (int row = 0; row < height / 2; ++row) {
        const uint8_t* row_src = uv_src + static_cast<size_t>(row) * nv12.step;
        for (int col = 0, j = 0; col < width; col += 2, ++j) {
            u[static_cast<size_t>(row) * (width / 2) + j] = row_src[col];
            v[static_cast<size_t>(row) * (width / 2) + j] = row_src[col + 1];
        }
    }

    const float scale = std::max(width / static_cast<float>(impl_->out_w),
                                 height / static_cast<float>(impl_->out_h));
    const int rw = static_cast<int>(std::round(width / scale));
    const int rh = static_cast<int>(std::round(height / scale));
    const int pad_x = (impl_->out_w - rw) / 2;
    const int pad_y = (impl_->out_h - rh) / 2;
    auto output = std::make_shared<std::vector<float>>(3 * impl_->out_w * impl_->out_h);

    auto yp = impl_->make_image(y, width, height);
    auto up = impl_->make_image(u, width / 2, height / 2);
    auto vp = impl_->make_image(v, width / 2, height / 2);
    cl_int e = CL_SUCCESS;
    cl_mem out = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                output->size() * sizeof(float), output->data(), &e);
    Impl::check(e, "clCreateBuffer output");

    e = 0;
    e |= clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &yp.second);
    e |= clSetKernelArg(impl_->kernel, 1, sizeof(cl_mem), &up.second);
    e |= clSetKernelArg(impl_->kernel, 2, sizeof(cl_mem), &vp.second);
    e |= clSetKernelArg(impl_->kernel, 3, sizeof(cl_mem), &out);
    e |= clSetKernelArg(impl_->kernel, 4, sizeof(int), &width);
    e |= clSetKernelArg(impl_->kernel, 5, sizeof(int), &height);
    e |= clSetKernelArg(impl_->kernel, 6, sizeof(int), &impl_->out_w);
    e |= clSetKernelArg(impl_->kernel, 7, sizeof(int), &impl_->out_h);
    e |= clSetKernelArg(impl_->kernel, 8, sizeof(float), &scale);
    e |= clSetKernelArg(impl_->kernel, 9, sizeof(int), &pad_x);
    e |= clSetKernelArg(impl_->kernel, 10, sizeof(int), &pad_y);
    Impl::check(e, "clSetKernelArg");

    const size_t global[] = {static_cast<size_t>(impl_->out_w), static_cast<size_t>(impl_->out_h)};
    e = clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 2, nullptr, global,
                               nullptr, 0, nullptr, nullptr);
    if (e == CL_SUCCESS) e = clFinish(impl_->queue);
    if (e == CL_SUCCESS)
        e = clEnqueueReadBuffer(impl_->queue, out, CL_TRUE, 0,
                                output->size() * sizeof(float), output->data(),
                                0, nullptr, nullptr);

    clReleaseMemObject(out);
    clReleaseMemObject(yp.second); clReleaseMemObject(yp.first);
    clReleaseMemObject(up.second); clReleaseMemObject(up.first);
    clReleaseMemObject(vp.second); clReleaseMemObject(vp.first);
    Impl::check(e, "OpenCL NV12 preprocess");

    Result result;
    result.data = std::move(output);
    result.scale = scale;
    result.pad_x = pad_x;
    result.pad_y = pad_y;
    result.ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}
