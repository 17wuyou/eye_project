cmake_minimum_required(VERSION 3.15)
project(CoreEngine VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 1. 查找依赖库 ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(unofficial-uwebsockets REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED) 

# 添加 nlohmann_json 的包含目录
include_directories(vendor/nlohmann_json/include)

# --- Porcupine 库设置 ---
set(PORCUPINE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/porcupine/include)
set(PORCUPINE_DLL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/porcupine/lib/windows/amd64/libpv_porcupine.dll)
set(PORCUPINE_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/porcupine/lib/windows/amd64/libpv_porcupine.lib)

add_library(porcupine_lib INTERFACE)
target_include_directories(porcupine_lib INTERFACE "${PORCUPINE_INCLUDE_DIR}")
target_link_libraries(porcupine_lib INTERFACE "${PORCUPINE_LIB_PATH}")


# --- 2. 定义编译目标 ---
add_executable(CoreEngine
    src/main.cpp
    src/Client.cpp
    src/ClientManager.cpp
    src/WebSocketServer.cpp
    src/CoreEngine.cpp
    src/KeyframeDetector.cpp
    src/KwsService.cpp
    src/pybind_bridge.cpp # ✅ 修复：将桥接代码直接编译进主程序
)

add_custom_command(
    TARGET CoreEngine 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PORCUPINE_DLL_PATH}"
        "$<TARGET_FILE_DIR:CoreEngine>"
    COMMENT "Copying libpv_porcupine.dll to output directory"
)

pybind11_add_module(my_project_bridge SHARED
    src/pybind_bridge.cpp
)


# --- 3. 为目标设置属性 ---

# --- 为 CoreEngine 目标 ---
target_include_directories(CoreEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${PORCUPINE_INCLUDE_DIR}
)

target_compile_definitions(CoreEngine PRIVATE "__attribute__(x)=")

target_link_libraries(CoreEngine PRIVATE
    unofficial::uwebsockets::uwebsockets
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    ${OpenCV_LIBS}
    Python3::Python
    porcupine_lib
    Ole32.lib
    # ✅ 修复：此处不再需要链接 my_project_bridge，因为它的源文件已包含在上面
)

# --- 为 my_project_bridge 目标 ---
target_include_directories(my_project_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(my_project_bridge PRIVATE pybind11::module)