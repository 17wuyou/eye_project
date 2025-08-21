# 🚀 快速开始（修订·避坑版）

以下内容针对 **Windows 64-bit** 用户，按顺序执行可最大限度避免「DLL 找不到」「CMake 找不到 Python」等常见坑。

---

## a. 克隆仓库

```bash
git clone https://github.com/17wuyou/eye_project.git
cd eye_project

## b. 初始化 C++ 依赖（pybind11）

> 本项目 `deps` 目录已内置 pybind11，通常 **无需手动操作**。  
> 如需强制更新，可取消下一行注释：

```bash
# git submodule update --init --recursive

## c. 安装并配置 OpenCV C++ 开发库（关键步骤）

> C++ 模块依赖的 **OpenCV 版本** 必须与 **Python 版本及编译器严格匹配**，否则会出现「DLL load failed」。

### 1. 下载  
从 [OpenCV 官网](https://opencv.org/releases/) 下载 **Windows 最新版开发库**。

### 2. 解压  
双击 `.exe` 安装包，将内容解压到 **无中文、无空格** 的稳定路径，例如：  
D:\opencv

### 3. 设置系统环境变量  
> 修改后 **务必** 关闭并重新打开终端，使其生效。

| 变量名 | 值 | 用途 |
| --- | --- | --- |
| **OpenCV_DIR** | `D:\opencv\build` | 让 CMake 在编译时找到库 |
| **Path** | `%OpenCV_DIR%\x64\vc16\bin` | 让 Windows 在运行时找到 `.dll` |

> ✅ 请确认 `D:\opencv\build\x64\vc16` 存在；Python 3.11 需配合 **vc16**。

## d. 编译 C++ 模块

> **前置条件**  
> 请先完成步骤 **e**（创建并激活 Python 3.11 虚拟环境），再进行本节操作。

| 步骤 | 命令 / 说明 |
| --- | --- |
| **1. 清理旧缓存** | ```cmd<br>rmdir /S /Q build<br>``` |
| **2. 新建构建目录** | ```cmd<br>mkdir build<br>cd build<br>``` |
| **3. 配置项目（CMake）** | 为避免 CMake 选错 Python，强制指定解释器路径：<br>```cmd<br>cmake -DPython_EXECUTABLE="D:/AnacondaDowload/envs/eye_env_311/python.exe" ../cpp_src<br>```<br>（请将路径换成 `where python` 得到的实际路径） |
| **4. 构建** | ```cmd<br>cmake --build . --config Release<br>``` |
| **5. 返回根目录** | ```cmd<br>cd ..<br>``` |
| **6. 复制产物** | 将 `build\Release\*.pyd`（如 `my_project_cpp.cp311-win_amd64.pyd`）**手动复制到项目根目录**。 |

> **终极避坑方案**  
> 再把 `D:\opencv\build\x64\vc16\bin\opencv_world*.dll` 一并复制到项目根目录，可彻底避免运行时 `ImportError: DLL load failed`。


## e. 创建 Python 虚拟环境并安装依赖

> 强烈建议使用 **Python 3.11 (64-bit)**，与已验证的 vc16 工具链保持一致。

| 步骤 | 命令 |
| --- | --- |
| **1. 创建虚拟环境**（推荐 Conda） | ```cmd<br>conda create -n eye_env_311 python=3.11<br>``` |
| **2. 激活虚拟环境** | ```cmd<br>conda activate eye_env_311<br>``` |
| **3. 升级 pip** | ```cmd<br>pip install --upgrade pip<br>``` |
| **4. 安装 Python 依赖** | ```cmd<br>pip install -r requirements.txt<br>``` |

> ⚠️ 若安装过程中出现依赖冲突，请根据报错提示手动调整 `requirements.txt` 中的版本号。
