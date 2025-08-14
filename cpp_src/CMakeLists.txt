cmake_minimum_required(VERSION 3.12)
project(my_project_cpp)

# 找到 Python 解释器和开发库
find_package(Python REQUIRED COMPONENTS Interpreter Development)
# 找到 OpenCV 库
find_package(OpenCV REQUIRED)

# --- 修正部分 ---
# 使用 CMAKE_CURRENT_SOURCE_DIR (指向 cpp_src) 和 CMAKE_BINARY_DIR (指向 build)
# 来构建 pybind11 的源目录和构建目录的路径。

# pybind11 的源文件路径
set(PYBIND11_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../deps/pybind11)

# 为 pybind11 指定一个在 build 目录下的构建子目录
set(PYBIND11_BINARY_DIR ${CMAKE_BINARY_DIR}/pybind11_build)

# 使用带有源目录和二进制目录的 add_subdirectory 命令
add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})
# --- 修正结束 ---


# 将我们的源文件添加到变量中
set(SOURCES
    main.cpp
    keyframe_detector.cpp
    audio_utils.cpp
)

# 创建 Python 扩展模块
# pybind11_add_module 现在应该能被识别了
pybind11_add_module(my_project_cpp SHARED ${SOURCES})

# 链接 OpenCV
target_link_libraries(my_project_cpp PRIVATE ${OpenCV_LIBS})

# 包含必要的头文件目录
target_include_directories(my_project_cpp PRIVATE
    ${Python_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    .  # 包含当前目录 (cpp_src)，以便 .h 文件能被找到
)

# 设置 C++ 标准
set_target_properties(my_project_cpp PROPERTIES CXX_STANDARD 17)