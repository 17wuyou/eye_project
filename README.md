# Real-Time AI Vision and Voice Assistant (Python/C++ Hybrid)

这是一个高性能的实时人工智能视觉与语音助手项目。它通过 WebSocket 从客户端接收实时音视频流，并在服务器端进行一系列复杂的 AI 处理，包括语音活动检测 (VAD)、关键词唤醒 (KWS)、说话人日志 (Diarization)、语音识别 (ASR)、人脸识别、关键帧检测以及与大型语言模型 (LLM) 的交互。

为了追求极致的性能，该项目采用了 **Python/C++ 混合编程**的架构。高层业务逻辑、网络通信和第三方 AI 库的调用由 Python 负责，而计算密集型的底层任务（如图像处理和音频分析）则由 C++ 实现，并通过 **pybind11** 暴露给 Python，从而显著降低 CPU 负载，提升处理效率。123

## 🌟 核心功能

*   **实时音视频流处理**: 通过 WebSocket 高效接收和处理来自客户端的音视频数据。
*   **语音处理流水线**:
    *   **语音活动检测 (VAD)**: 实时判断音频流中是否包含人声。
    *   **关键词唤醒 (KWS)**: 使用 `pvporcupine` 监听特定的唤醒词（如“小童”）。
    *   **说话人日志 (Diarization)**: 使用 `pyannote.audio` 区分音频流中的不同说话人。
    *   **语音识别 (ASR)**: 支持 `Whisper` 或 `FunASR` 引擎，将语音转换为文字。
    *   **声纹识别**: 能够识别已注册的用户（"user"）和其他历史说话人。
*   **视觉处理流水线**:
    *   **关键帧检测**: 基于直方图和帧差分析，智能提取场景发生显著变化的视频帧。
    *   **人脸识别**: 使用 `InsightFace` 进行高精度的人脸检测和身份识别，并能自动注册新面孔。
*   **智能交互**:
    *   **LLM 集成**: 在检测到唤醒词后，将 ASR 结果和当前关键帧图像发送给 Google Gemini Pro 模型，生成智能回答。
    *   **TTS 语音合成**: 使用 `edge-tts` 将 LLM 的回答合成为自然语音，并通过 WebSocket 回传给客户端播放。
*   **数据管理与监控**:
    *   **Web 控制面板**: 提供一个 Flask 驱动的 Web 界面，用于监控服务状态和向客户端发送指令。
    *   **媒体文件管理**: 提供一个管理面板，用于查看和重命名已识别的人脸图像和说话人音频样本。
    *   **事件数据集记录**: 自动将“关键帧-ASR字幕”配对保存为结构化的事件数据集，便于后续模型训练。
*   **高性能混合架构**:
    *   **C++ 核心**: `KeyframeDetector` (关键帧检测) 和 `is_audio_active_by_rms` (VAD) 已被重写为 C++，以实现原生性能。
    *   **pybind11 桥梁**: 无缝连接 Python 和 C++，有效释放 Python GIL，实现真正的多线程并行计算。

## 🏛️ 项目架构

项目采用模块化的分层设计：

*   **`app.py`**: 项目入口，负责初始化 Flask 应用、SocketIO、线程池和所有服务。
*   **`modules/`**: 包含项目的主要逻辑模块。
    *   `websocket_callbacks.py`: 处理与客户端 WebSocket 的所有原始数据交互。
    *   `service_management.py`: 负责所有 AI 服务（ASR, Diarization, Face等）的生命周期管理（加载、卸载、重载）。
    *   `audio_processing_logic.py` & `video_processing_logic.py`: 定义核心的音视频处理流水线。
    *   `llm_service.py`: 封装了与 Gemini 和 TTS 的交互逻辑。
    *   `gui_callbacks.py` & `management_callbacks.py`: 处理来自 Web 控制面板的事件。
    *   `state_manager.py`: 全局状态管理中心，保存客户端连接、检测器实例等信息。
*   **根目录服务文件 (`asr_service.py`, `face_service.py`, etc.)**: 定义了各个独立AI功能的类封装。
*   **`cpp_src/`**: 存放所有 C++ 源代码。
    *   `keyframe_detector.cpp`, `audio_utils.cpp`: 高性能 C++ 实现。
    *   `main.cpp`: 使用 pybind11 定义 Python 模块的绑定。
    *   `CMakeLists.txt`: C++ 部分的构建配置文件。
*   **`deps/`**: 存放第三方 C++ 依赖，如 `pybind11`。

## 🚀 快速开始

### 1. 环境依赖

*   **Python**: 3.9+
*   **C++ 编译器**:
    *   **Windows**: Visual Studio 2019 或更高版本 (需安装 "使用C++的桌面开发" 工作负载)。
    *   **Linux**: GCC 9+
    *   **macOS**: Clang/LLVM
*   **CMake**: 3.12+
*   **Git**: 用于克隆仓库和子模块。
*   **FFmpeg**: 用于音频格式转换。请确保其可执行文件路径已添加到系统 `PATH` 环境变量中。
*   **(可选) NVIDIA GPU**:
    *   **CUDA Toolkit**: 11.8+
    *   **cuDNN**

### 2. 安装步骤

#### a. 克隆仓库

```bash
git clone https://github.com/17wuyou/eye_project.git
cd eye_project
```

#### b. 初始化 C++ 依赖 (pybind11)(项目中已安装)

```bash
git submodule add https://github.com/pybind/pybind11.git deps/pybind11
git submodule update --init --recursive
```

#### c. 安装 OpenCV C++ 开发库

本项目 C++ 部分依赖 OpenCV。请从 [OpenCV 官网](https://opencv.org/releases/) 下载适用于您操作系统的最新版本开发库，并将其路径配置到环境变量 `OpenCV_DIR` 中，指向其 `build` 目录。

```bash
# 举例
# 环境变量中，设置 OpenCV_DIR
OpenCV_DIR = <你的路径>\opencv\build
# Path 中添加
%OpenCV_DIR%\x64\vc16\bin # 或者 %OpenCV_DIR%\x64\vc17\bin，看自己的文件夹是vc16还是vc17
```



#### d. 编译 C++ 模块(已有示例的.sln项目和.pyd示例文件，若项目无更改可以直接跳过此步骤)

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目 (CMake)
# Windows 用户可能需要指定生成器, e.g., -G "Visual Studio 17 2022"
cmake ../cpp_src

# 构建项目
# Windows
cmake --build . --config Release
# Linux / macOS
make -j$(nproc)

# 将编译好的模块复制到项目根目录
# 在 Windows 上，它位于 build/Release/ 目录下，是一个 .pyd 文件
# 在 Linux/macOS 上，它直接在 build/ 目录下，是一个 .so 文件
cd ..
# (手动或通过脚本将 .pyd/.so 文件复制到此处)
```

#### e. 创建 Python 虚拟环境并安装依赖

```bash
# 创建虚拟环境（venv虚拟环境已经创建，可以跳过这一步）
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装所有 Python 依赖
pip install -r requirements.txt
```
**注意**: 如果在安装 `requirements.txt` 时遇到依赖冲突，请根据错误提示调整文件中的包版本。

### 3. 配置

在运行前，请检查并配置 `configs.py` 文件：

*   **PICOVOICE_ACCESS_KEY**: 填入你的 PicoVoice AccessKey 用于 KWS。
*   **GEMINI_API_KEY**: 填入你的 Google AI Studio API Key 用于 LLM。
*   **HF_TOKEN (环境变量)**: 建议设置 `HF_TOKEN` 环境变量，用于从 Hugging Face Hub 下载 `pyannote` 模型。
*   **模型路径和数据库路径**: 根据需要调整。

### 4. 运行项目

确保你的**虚拟环境已激活**，并且 C++ 模块 (`.pyd` 或 `.so` 文件) 已位于项目根目录。

```bash
python app.py
```

应用启动后，你将看到类似以下的日志：
```
INFO:__main__:准备启动Flask应用 (带SocketIO) 于 0.0.0.0:5002。
```

### 5. 使用方法

1.  **连接客户端**: 启动你的 Android 客户端或其他 WebSocket 客户端，连接到服务器的 `ws://<server-ip>:5002/ws` 地址。
2.  **监控面板**: 在浏览器中打开 `http://127.0.0.1:5002/` 查看实时日志、视频流和客户端列表。
3.  **管理面板**: 在浏览器中打开 `http://127.0.0.1:5002/management` 查看和管理已识别的人脸和说话人。
4.  **触发交互**: 对着客户端说出唤醒词（如“小童”），然后提出你的问题。服务器将通过 LLM 和 TTS 进行回应。

## 🛠️ 开发与调试

*   **修改 C++ 代码**: 每次修改 `cpp_src` 目录下的 C++ 文件后，都需要重新执行**步骤 d (编译 C++ 模块)** 并将新生成的模块文件复制到根目录。
*   **日志**: 项目使用 Python 的 `logging` 模块。主要的日志文件是 `log.txt`，用于记录详细的延迟和事件跟踪。

---