2. 安装步骤（修订与避坑版）
a. 克隆仓库
bash
git clone https://github.com/17wuyou/eye_project.git
cd eye_project
b. 初始化 C++ 依赖 (pybind11)
注：此项目的 deps 目录中已包含 pybind11，通常无需手动执行此步骤

bash
# git submodule update --init --recursive
c. 安装并配置 OpenCV C++ 开发库 (关键步骤)
本项目 C++ 部分依赖 OpenCV，其版本必须与您的 Python 版本及编译器严格匹配。

下载：从 OpenCV 官网下载适用于 Windows 的最新版本开发库。
解压：运行下载的 .exe 文件，将其解压到一个稳定的、不含中文或空格的路径（例如 D:\opencv）。
设置环境变量：这是成功运行的关键。
OpenCV_DIR: 此变量用于编译时让 CMake 找到库
Path: 此变量用于运行时让 Windows 找到所需的 .dll 文件

bash
# 举例:
# 在系统环境变量中，新建一个条目
OpenCV_DIR = D:\opencv\build

# 在系统环境变量 Path 的列表中，新建一条
# (Python 3.11 需配合 vc16，请确保 D:\opencv\build\x64 目录下存在 vc16 文件夹)
%OpenCV_DIR%\x64\vc16\bin

重要：修改环境变量后，必须关闭并重新打开您的命令行 / 终端窗口才能生效。
d. 编译 C++ 模块
注意：请确保已完成下一步 (e)，并在已激活的 Python 3.11 虚拟环境中执行此操作。

bash
# 1. 清理旧缓存并创建构建目录 (如果存在build文件夹)
rmdir /S /Q build
mkdir build
cd build

# 2. 配置项目 (CMake)
# 为避免CMake找到错误的Python版本，我们强制指定解释器路径
# a. 先在激活的环境中用 `where python` 命令找到路径
# b. 然后在下方命令中替换为您自己的路径
cmake -DPython_EXECUTABLE="D:/AnacondaDowload/envs/eye_env_311/python.exe" ../cpp_src

# 3. 构建项目
cmake --build . --config Release

# 4. 回到根目录
cd ..

构建完成后，必须手动将 build\Release 目录下的 .pyd 文件（例如 my_project_cpp.cp311-win_amd64.pyd）复制到项目根目录。

终极避坑方案：为彻底避免运行时 ImportError: DLL load failed 的问题，强烈建议将 D:\opencv\build\x64\vc16\bin 目录下的 opencv_worldXXX.dll 文件也复制到项目根目录，和 .pyd 文件放在一起。
e. 创建 Python 虚拟环境并安装依赖
强烈建议使用 Python 3.11 (64 位) 版本，以保证与我们验证过的 vc16 工具链兼容。

bash
# 1. 创建虚拟环境 (推荐使用 Conda)
conda create -n eye_env_311 python=3.11

# 2. 激活虚拟环境
conda activate eye_env_311

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装所有 Python 依赖
pip install -r requirements.txt





注意：如果在安装 requirements.txt 时遇到依赖冲突，请根据错误提示调整文件中的包版本。
