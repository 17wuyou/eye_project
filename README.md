# 安装步骤（修订与避坑版）

> ⚠️ 操作前请先完整阅读本指南，避免因版本或路径问题导致的常见错误。

---

## a. 克隆仓库

```bash
git clone https://github.com/17wuyou/eye_project.git
cd eye_project
b. 初始化 C++ 依赖（pybind11）
提示：本项目 deps/ 目录已内置 pybind11，通常无需手动执行。
如确实需要自行更新，可取消下一行注释后运行：
bash
复制
# git submodule update --init --recursive
c. 安装并配置 OpenCV C++ 开发库（关键步骤）
本项目 C++ 部分依赖 OpenCV，版本需与你的 Python 版本 + 编译器 严格匹配。
1. 下载
从 OpenCV 官网 获取 Windows 最新版本开发库（.exe 安装包）。
2. 解压
运行 .exe，将 OpenCV 解压到 无中文、无空格 的稳定路径，例如：
D:\opencv
3. 配置环境变量（成败关键）
表格
复制
变量名	值示例	说明
OpenCV_DIR	D:\opencv\build	供 CMake 编译时 查找库
Path	%OpenCV_DIR%\x64\vc16\bin	供 运行时 查找 .dll
Python 3.11 对应 vc16；如文件夹为 vc17 请相应调整。
⚠️ 重要：修改环境变量后，必须关闭并重新打开终端/IDE，否则不会生效！
d. 编译 C++ 模块
⚠️ 前置条件：请先完成下一步 (e) 创建并激活 Python 3.11 虚拟环境，再执行本节命令！
bash
复制
# 1. 清理旧缓存并创建构建目录
rmdir /S /Q build
mkdir build
cd build

# 2. 配置项目（CMake）
#    为避免 CMake 选错 Python，强制指定解释器路径：
#    先在激活的环境里执行 `where python` 获取路径，再替换下方命令
cmake -DPython_EXECUTABLE="D:/AnacondaDowload/envs/eye_env_311/python.exe" ../cpp_src

# 3. 构建项目
cmake --build . --config Release

# 4. 回到项目根目录
cd ..
构建完成后
手动将 build\Release\*.pyd（如 my_project_cpp.cp311-win_amd64.pyd）复制到项目根目录。
终极避坑：把
D:\opencv\build\x64\vc16\bin\opencv_worldXXX.dll
一并复制到项目根目录，与 .pyd 同目录，可彻底避免 ImportError: DLL load failed。
e. 创建 Python 虚拟环境并安装依赖
表格
复制
推荐版本	说明
Python 3.11 64-bit	与我们验证过的 vc16 工具链完全兼容
bash
复制
# 1. 创建虚拟环境（推荐 Conda）
conda create -n eye_env_311 python=3.11

# 2. 激活虚拟环境
conda activate eye_env_311

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装所有 Python 依赖
pip install -r requirements.txt
若出现依赖冲突，按提示在 requirements.txt 中调整版本后重新安装即可。
