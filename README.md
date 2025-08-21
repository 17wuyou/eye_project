# 智能眼镜后端服务 (Smart Glasses Backend Service)

这是一个为智能眼镜设备提供后端支持的综合服务项目。它采用C++和Python混合编程模型，以实现高性能的实时数据处理和强大的AI功能。

## 主要功能

* **实时流处理**: 基于C++的WebSocket服务器，用于接收和处理来自眼镜的实时音频和视频流。
* **关键词唤醒 (KWS)**: 使用 PicoVoice Porcupine 在C++端进行低延迟的关键词检测。
* **语音识别 (ASR)**: 使用 OpenAI Whisper 模型进行高精度的语音到文本转换。
* **声纹识别 (Diarization)**: 使用 Pyannote.audio 区分对话中的不同说话人。
* **人脸识别 (Face Recognition)**: 使用 InsightFace 进行人脸检测和身份识别。
* **大语言模型 (LLM)**: 集成 Google Gemini Pro Vision，实现基于视觉和语音的智能问答。
* **语音合成 (TTS)**: 使用 Edge TTS 将AI的回答合成为自然语音。
* **Web管理界面**: 基于 Flask 和 Socket.IO 的前端页面，用于监控和管理后端服务。

## 环境准备 (Prerequisites)

在开始之前，请确保您的开发环境中已安装以下软件：

1.  **Visual Studio 2022**: 需安装 “使用C++的桌面开发” 工作负载。
2.  **CMake**: 版本 3.15 或更高。
3.  **Python**: 版本 3.12 (64-bit)。
4.  **vcpkg**: C++ 包管理器，并已正确配置。

## 安装与构建流程

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd EyeTeach
    ```

2.  **创建Python虚拟环境**
    强烈建议在项目根目录下创建一个虚拟环境，以隔离依赖。
    ```bash
    python -m venv glassEnv
    ```

3.  **激活虚拟环境**
    ```powershell
    # 在 PowerShell 中
    .\glassEnv\Scripts\Activate.ps1
    ```

4.  **安装Python依赖库**
    在**已激活**的虚拟环境中，安装所有必需的Python库。
    ```bash
    # 首先安装PyTorch
    pip install torch torchvision toraudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    # 然后安装其余库
    pip install -r requirements.txt
    ```

5.  **【核心步骤】配置C++中的Python路径**
    由于本项目的C++核心 (`CoreEngine.exe`) 需要直接嵌入并调用Python，您必须在C++代码中手动指定正确的Python路径。

    * **打开源文件**: `src/main.cpp`
    * **定位到配置区域**: 找到文件开头的路径设置部分。
    * **根据您本机的路径进行修改**:

    您需要修改以下几行代码中的**绝对路径**，使其指向您自己电脑上的对应位置：

    ```cpp
    // 1️⃣ 指定您的主Python安装目录
    // 找到您电脑上安装Python 3.12的根目录
    Py_SetPythonHome(L"C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312");

    // ...

    // 3️⃣ 添加Python的搜索路径
    PyRun_SimpleString(
        "import sys\n"
        // 路径a: 您的主Python的Lib目录
        "sys.path.append(r'C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312\\Lib')\n"
        // 路径b: 您的主Python的site-packages目录
        "sys.path.append(r'C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages')\n"
        // 路径c: 您为此项目创建的虚拟环境的site-packages目录
        "sys.path.append(r'D:\\eye8.18\\EyeTeach\\glassEnv\\Lib\\site-packages')\n"
        // 路径d: 您本项目的根目录
        "sys.path.append(r'D:\\eye8.18\\EyeTeach')\n"
    );
    ```
    **示例**: 如果您的用户是 `newUser`，Python安装在 `C:\Python312`，项目放在 `C:\Projects\EyeTeach`，那么修改后的代码应该像这样：
    ```cpp
    Py_SetPythonHome(L"C:\\Python312");
    
    PyRun_SimpleString(
        "import sys\n"
        "sys.path.append(r'C:\\Python312\\Lib')\n"
        "sys.path.append(r'C:\\Python312\\Lib\\site-packages')\n"
        "sys.path.append(r'C:\\Projects\\EyeTeach\\glassEnv\\Lib\\site-packages')\n"
        "sys.path.append(r'C:\\Projects\\EyeTeach')\n"
    );
    ```
    **修改完成后，请务必保存 `main.cpp` 文件。**

6.  **编译C++代码**
    确保您仍处于**已激活**的虚拟环境中，运行以下命令：
    ```cmd
    # 生成项目文件 (请确保vcpkg路径正确)
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"

    # 编译项目
    cmake --build build
    ```

## 运行程序

1.  **准备DLL文件**
    编译成功后，将 `vendor\porcupine\lib\windows\amd64\libpv_porcupine.dll` 文件复制到 `build\Debug` 目录下，与 `CoreEngine.exe` 放在一起。

2.  **启动核心引擎**
    在**已激活**的虚拟环境中，运行可执行文件：
    ```cmd
    cd build\Debug
    CoreEngine.exe
    ```
    如果一切顺利，您将看到服务器成功启动并监听端口的日志信息。
