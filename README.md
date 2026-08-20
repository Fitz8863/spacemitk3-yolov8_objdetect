# SpaceMIT K3 YOLOv8 摄像头推理

这个目录实现了 YOLOv8 在 SpaceMIT K3 板端的实时摄像头推理，摄像头读取、MJPEG 硬件解码、OpenCL GPU 前处理、队列、显示方式与
`spacemitk3_yolo26_detect` 仓库的 `gstreamer-opencl-k3` 分支保持一致。

## 数据流

```text
USB 摄像头 V4L2 MJPEG 1280x720@25
  -> GStreamer v4l2src
  -> spacemitdec code-type=9（K3 VPU 硬件解码）
  -> appsink NV12（只保留最新帧）
  -> OpenCL GPU：Y/UV 上传 + NV12->RGB + resize + letterbox + CHW + FP32/255
  -> SpaceMIT ONNX Runtime EP
  -> YOLOv8 raw output [1, 4+nc, N]
  -> xywh 解码 + 置信度筛选 + class-aware NMS
  -> OpenCV HighGUI 显示
```

当前模型 `models/yolov8_best.q.onnx` 的文件元数据表明它是 Ultralytics YOLOv8s-relu、6 类骰子模型，导出参数为 `nms=False`、`imgsz=[640,640]`、`end2end=False`。因此它不是 YOLO26 的 `[1,300,6]` end-to-end 输出，程序会在 CPU 侧执行 YOLOv8 的外部解码和 NMS。

类别 ID `0..5` 在显示时对应骰子面 `1..6`。

## 编译

请在 K3 板端 `/home/spacemit/projects/yolov8` 执行：

```bash
cd /home/spacemit/projects/yolov8
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
cmake --build build -j4
```

如果板端没有 `/opt/opencv-spacemit/lib/cmake/opencv4`，去掉 `-DOpenCV_DIR=...`，以 CMake 实际打印的 OpenCV include/library 路径为准。参考分支使用的是系统/板端 OpenCV 包；应在 K3 板端编译；本工作区的本机环境不作为 riscv64/SpaceMIT 运行时验证依据。

## 运行

### 模型和 OpenCL/SpaceMIT EP 自测（不访问摄像头）

```bash
cd /home/spacemit/projects/yolov8
./build/yolov8_camera \
  --model models/yolov8_best.q.onnx \
  --self-test --no-display
```

### 摄像头显示

```bash
cd /home/spacemit/projects/yolov8
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
--self-test        初始化 OpenCL GPU 和模型，并执行一次真实 NV12 OpenCL 前处理和推理
```

## 实现边界

- 摄像头阶段保持 `v4l2src ! image/jpeg ! spacemitdec code-type=9 ! video/x-raw,format=NV12 ! appsink`，不使用 `videoconvert`。
- NV12 会先复制成紧凑连续内存，避免 GStreamer/VPU buffer 生命周期导致 `queueBuffer ... Invalid argument`。
- 前处理使用 OpenCL GPU kernel 完成 Y/UV 图像采样、NV12 转 RGB、resize、114/128 letterbox、CHW 和 `/255`；主机侧仅负责将 NV12 的 Y/UV 数据上传到 OpenCL。
- 推理线程只访问一个 ORT session；显示在主线程执行，保持 HighGUI 事件循环安全。
- YOLOv8 解码当前支持 `[1,C,N]` 和 `[1,N,C]` 两种三维输出布局；对当前模型预期为 `[1,10,8400]`，即 4 个框通道加 6 个类别通道。
- YOLOv8 输出的框按 `cx,cy,w,h`、类别分数已在导出图中完成 DFL/激活，程序不会再次对类别分数做 sigmoid；随后撤销 letterbox 并做按类别 NMS。

## RVV 分支基线验证（2026-08-20）

父分支 `gstreamer-opencv_rvv-k3` 的基线数据如下；OpenCL 分支应以实际运行日志中的 `OpenCL GPU: ...`、`GStreamer camera opened` 和 `Done.` 行为准。

- CMake 配置成功，使用 OpenCV `4.10.0`、板端 riscv64 编译器和 SpaceMIT ORT 依赖；
- C++ 编译链接成功，生成 `build/yolov8_camera`；
- 模型运行时输入为 `[1,3,640,640]`，输出为 `[1,10,8400]`；
- `/dev/video1` 成功通过 `spacemitdec code-type=9` 解码 1280x720 MJPEG，摄像头实际协商为 24 FPS；
- `--no-display --max-frames 30` 端到端成功：`prepared=30`、`infer=29`、`display=27`，平均前处理约 `6.32 ms`，推理约 `29.48 ms`，检测到 `325` 个框；
- 输出日志确认实际使用 `[1,C,N]`、`C=10`、`N=8400` 的 YOLOv8 解码路径。

## OpenCL 分支验证（2026-08-20）

已在 K3 板端验证当前 `gstreamer-opencl-k3` 工作树：

- OpenCL 设备为 `PowerVR B-Series BXM-4-64`，平台为 `PowerVR`；
- 全量清理编译成功，生成 `build/yolov8_camera`；
- `--self-test --no-display` 成功，真实执行 1280x720 NV12 → OpenCL GPU 前处理 → SpaceMIT EP 推理；
- `--no-display --max-frames 30` 成功：`prepared=30`、`infer=29`、`display=28`，平均前处理约 `8.35 ms`，推理约 `29.62 ms`；
- OpenCL 资源复用后，前处理相较原始 OpenCL 实现的约 `15.82 ms` 明显下降；具体 FPS 会随摄像头、负载和队列丢帧策略变化。

已知现象：

- 退出摄像头管线时，`spacemitdec` 仍可能打印一次 `queueBuffer ... Invalid argument`；该现象在 RVV 基线分支也出现，当前应用能正常退出，尚未将其误报为 OpenCL 前处理故障。

仍需注意：

- 25 FPS caps 首次协商失败后会自动回退到 24 FPS，这是当前摄像头能力表现；
- 短测末尾仍可能出现一次 `spacemitdec queueBuffer ... Invalid argument`，与参考分支观察到的 VPU buffer 生命周期告警一致；程序随后正常退出。建议再做带显示器的长时间稳定性测试；
- 本次已验证无显示端到端链路，HighGUI 显示代码保持参考分支方式，但尚未做长时间带显示器测试。
