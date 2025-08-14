cmake_minimum_required(VERSION 3.12)
project(my_project_cpp)

# 找到 Python 解释器和开发库
find_package(Python REQUIRED COMPONENTS Interpreter Development)
# 找到 OpenCV 库
find_package(OpenCV REQUIRED)

# 添加 pybind11
add_subdirectory(../deps/pybind11)

# 将我们的源文件添加到变量中
set(SOURCES
    main.cpp
    keyframe_detector.cpp
    audio_utils.cpp
)

# 创建 Python 扩展模块
# my_project_cpp 是模块名，在 Python 中将通过 from my_project_cpp import ... 来使用
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