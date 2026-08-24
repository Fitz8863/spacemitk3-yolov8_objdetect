# SpaceMIT K3 YOLOv8 摄像头推理

这个目录实现了 YOLOv8 在 SpaceMIT K3 板端的实时摄像头推理，摄像头读取、MJPEG 解码、OpenCV-RVV 前处理、队列、显示方式与
`spacemitk3_yolo26_detect` 仓库的 `gstreamer-opencv_rvv-k3` 分支保持一致。

## 数据流

```text
USB 摄像头 V4L2 MJPEG 1280x720@25
  -> GStreamer v4l2src
  -> 优先 spacemitdec code-type=9（K3 VPU 硬件解码）
     无可用 V4L2 M2M 解码节点时自动回退 jpegdec + videoconvert
  -> appsink NV12（只保留最新帧）
  -> GstreamerFrame：优先以 GstBuffer 映射内存创建 OpenCV Mat header（安全零拷贝）
     不兼容的 plane/stride 布局自动回退为紧凑 NV12 深拷贝
  -> OpenCV-RVV：Y/UV resize + letterbox + NV12->RGB + HWC->CHW + FP32/255
  -> SpaceMIT ONNX Runtime EP
  -> YOLOv8 raw output [1, 4+nc, N]
  -> xywh 解码 + 置信度筛选 + class-aware NMS
  -> OpenCV HighGUI 显示
```

当前模型 `models/yolov8_best.q.onnx` 的文件元数据表明它是 Ultralytics YOLOv8s-relu、6 类骰子模型，导出参数为 `nms=False`、`imgsz=[640,640]`、`end2end=False`。因此它不是 YOLO26 的 `[1,300,6]` end-to-end 输出，程序会在 CPU 侧执行 YOLOv8 的外部解码和 NMS。

类别 ID `0..5` 在显示时对应骰子面 `1..6`。

## 编译

请在 K3 板端 `<project-root>` 执行：

```bash
cd <project-root>
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

如果 SpaceMIT SDK 暂时解压在项目外的自定义目录，使用相同的源码和下面的参数即可（路径仅作为示例）：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$VENDOR_PREFIX/lib/cmake/opencv4" \
  -DSPACEMIT_ORT_ROOT="$SPACEMIT_ORT_PREFIX"
cmake --build build -j"$(nproc)"
```

默认不指定 `OpenCV_DIR`，CMake 会优先使用系统 OpenCV。只有当 OpenCV 安装在非标准前缀时，才额外传入该板子的路径，例如 `-DOpenCV_DIR="$VENDOR_PREFIX/lib/cmake/opencv4"`。

`SPACEMIT_ORT_ROOT` 同样应指向目标板上的 SpaceMIT ONNX Runtime 安装前缀，目录中应包含 `include/spacemit_ort_env.h`、`lib/libonnxruntime.so` 和 `lib/libspacemit_ep.so`。如果 SDK 已安装到系统目录，使用 `/usr`；如果是临时解压目录或自定义前缀，只修改这个 CMake 参数，不要修改源码。

## 运行

### 模型和 SpaceMIT EP 自测（不访问摄像头）

```bash
cd <project-root>
./build/yolov8_camera \
  --model models/yolov8_best.q.onnx \
  --self-test --no-display
```

### 摄像头显示

```bash
cd <project-root>
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/1000

./build/yolov8_camera \
  --model models/yolov8_best.q.onnx \
  --camera 1 \
  --width 1280 --height 720 --fps 25 \
  --intra-threads 4 \
  --ep-affinity "8;9;10;11" \
  --queue-depth 3 \
  --conf 0.25
```

参考分支中的摄像头可能只接受 1280x720@24；程序会先尝试请求的 FPS，`--fps 25` 失败时自动回退到 24 FPS。

### 无显示端到端测试

```bash
./build/yolov8_camera \
  --model models/yolov8_best.q.onnx \
  --camera 1 \
  --width 1280 --height 720 --fps 25 \
  --intra-threads 4 \
  --queue-depth 3 \
  --conf 0.25 \
  --no-display --max-frames 30
```

也可以显式指定设备：

```bash
./build/yolov8_camera --model models/yolov8_best.q.onnx --device /dev/video1
```

## 重要参数

```text
--model PATH       ONNX 模型
--camera N         使用 /dev/videoN，默认 1
--device PATH      显式指定 V4L2 节点，覆盖 --camera
--width N --height N --fps N
--conf FLOAT       置信度阈值，默认 0.25
--queue-depth N    每级队列深度，默认 3；满时丢旧帧降低延迟
--focus N          手动对焦，默认 0；-1 表示不改动
--zoom N           绝对变焦，默认 181；-1 表示不改动
--intra-threads N  SpaceMIT EP 线程数，默认 1
--ep-affinity LIST  EP 线程绑定的 AI 核，数量必须等于 --intra-threads
--no-display       不创建 HighGUI 窗口
--max-frames N     处理 N 帧后退出
--dump-input PATH  保存首帧 640x640 FP32 CHW 输入
--self-test        执行 NV12->OpenCV-RVV 前处理并完成一次推理
```

## 实现边界

- 摄像头阶段由原生 GStreamer 管理：优先使用 `v4l2src ! image/jpeg ! spacemitdec code-type=9 ! video/x-raw,format=NV12 ! appsink`。如果没有检测到可用的 V4L2 M2M 节点、硬件 pipeline 打开失败，或运行时连续读取失败，则打印原因并自动使用 `jpegdec ! videoconvert ! video/x-raw,format=NV12` 软件解码。
- 启动时会明确打印 `[Decoder] Using hardware decoder: spacemitdec ...` 或 `[Decoder] Using software decoder: jpegdec ...`；运行时回退会打印 `Hardware decoder runtime failure` 和 `Runtime fallback to software decoder succeeded`。
- NV12 默认尽量走浅拷贝：`GstVideoFrame` 映射后，OpenCV `cv::Mat` 只创建 header，不复制像素；`GstreamerFrame::owner` 持有 `GstSample` 和映射状态，并随帧经过前处理、推理、显示队列，直到最后一个消费者释放后才 unmap/unref。
- 零拷贝只在两个 plane 能表示为一个兼容的 NV12 视图时启用：Y/UV stride 满足要求，且 UV 紧接在 Y plane 后面；如果 VPU/GStreamer 给出分离 plane 或不兼容 padding，则自动逐行深拷贝，并打印 `safe copy fallback`。因此不要把裸 `cv::Mat` 长期保存为 GstBuffer 的替代品；旧的 `read(cv::Mat&)` 兼容接口会主动 `clone()`。
- 队列深度必须保持有限（默认 3），因为零拷贝会短暂持有 VPU/GStreamer buffer；关闭时先停止采集线程、等待工作线程退出并清空应用队列，再向 GStreamer pipeline 发送 EOS、等待 decoder drain，最后按 `PLAYING -> NULL` 释放，避免 appsink/VPU 输出 buffer 仍在队列中时销毁 channel。
- `FrameQueue` 使用 100 ms 轮询等待并检查 Ctrl-C；超时不会被误判为流结束，队列中已有帧会在关闭时先排空。
- 前处理保持参考分支的 Y/UV 分平面 resize、114/128 NV12 letterbox、NV12 转 RGB、CHW 和 `/255`。
- 推理线程只访问一个 ORT session；显示在主线程执行，保持 HighGUI 事件循环安全。
- YOLOv8 解码当前支持 `[1,C,N]` 和 `[1,N,C]` 两种三维输出布局；对当前模型预期为 `[1,10,8400]`，即 4 个框通道加 6 个类别通道。
- YOLOv8 输出的框按 `cx,cy,w,h`、类别分数已在导出图中完成 DFL/激活，程序不会再次对类别分数做 sigmoid；随后撤销 letterbox 并做按类别 NMS。

## 验证状态（2026-08-24）

已在一块 K3 板端验证：

- CMake 配置成功，使用 OpenCV `4.10.0`、板端 riscv64 编译器和 SpaceMIT ORT 依赖；
- C++ 编译链接成功，生成 `build/yolov8_camera`；
- `--self-test --no-display` 成功；运行时模型输入为 `[1,3,640,640]`，输出为 `[1,10,8400]`；
- `/dev/video1` 成功通过 `spacemitdec code-type=9` 解码 1280x720 MJPEG，摄像头实际协商为 24 FPS；
- 新版安全零拷贝路径已验证：硬件解码短测打印 `[Camera] NV12 path: zero-copy GstBuffer -> OpenCV Mat header`，`--no-display --max-frames 30` 正常退出，`prepared=30`、`infer=29`、`display=27`，平均前处理约 `6.38 ms`，推理约 `28.50 ms`，检测到 `348` 个框；
- 强制软件解码 `SPACEMIT_FORCE_SOFTWARE_DECODER=1` 也打印同样的 NV12 零拷贝路径，`--max-frames 10` 正常退出：`prepared=10`、`infer=9`、`display=7`；
- 输出日志确认实际使用 `[1,C,N]`、`C=10`、`N=8400` 的 YOLOv8 解码路径。

仍需注意：

- 25 FPS caps 首次协商失败后会自动回退到 24 FPS，这是当前摄像头能力表现；
- 2026-08-24 的板端短测和 Ctrl-C 测试中，原先的 `queueBuffer ... Invalid argument` 已不再出现；程序均正常返回。驱动仍可能打印一次 `V4L2_EVENT_EOS event is not support yet`，这是板端 MPP/V4L2 驱动对 EOS 事件的已知提示，不是应用 buffer 未清空，也未导致退出失败；
- 可用 `SPACEMIT_FORCE_SOFTWARE_DECODER=1` 强制验证软件解码路径，例如 `SPACEMIT_FORCE_SOFTWARE_DECODER=1 ./build/yolov8_camera ...`；
- 已验证无显示硬件解码端到端短测和 Ctrl-C 退出：退出时仍可能出现驱动的 `V4L2_EVENT_EOS event is not support yet` 提示，但未出现 `queueBuffer ... Invalid argument`，且程序完成 `Done.`；这属于板端 MPP/V4L2 驱动的 EOS 事件提示，不是应用队列未清空。
- 当前板端实际日志示例：
  ```text
  [Decoder] Using hardware decoder: spacemitdec (V4L2 M2M/MJPEG -> NV12)
  [Camera] NV12 path: zero-copy GstBuffer -> OpenCV Mat header (owner retained until consumers release the frame)
  ```
  软件路径示例：
  ```text
  [Decoder] Software decoder forced by SPACEMIT_FORCE_SOFTWARE_DECODER
  [Decoder] Using software decoder: jpegdec
  ```
- HighGUI 显示代码保持参考分支方式，仍建议在目标显示环境进行长时间稳定性测试。
