// src/pybind_bridge.cpp
#include "pybind_bridge.h"
#include <iostream>

// --- C++ 侧的 AIServiceManager 实现 ---
AIServiceManager::AIServiceManager() {
    // 构造函数：初始化Python服务
    py::gil_scoped_acquire acquire; // 获取Python全局解释器锁(GIL)
    try {
        // 导入您的Python服务管理模块
        auto service_management = py::module_::import("modules.service_management");
        
        // 调用Python函数来初始化所有服务
        std::cout << "[C++] Calling Python: initialize_services()..." << std::endl;
        service_management.attr("initialize_services")();
        std::cout << "[C++] Python services initialized." << std::endl;

        // 获取初始化后的 asr_service_instance 实例
        asr_service_ = service_management.attr("asr_service_instance");

        if (asr_service_.is_none()) {
            throw std::runtime_error("ASR service instance is None after initialization.");
        }
        std::cout << "[C++] Successfully acquired Python ASR Service instance." << std::endl;

    } catch (py::error_already_set &e) {
        std::cerr << "[C++ FATAL] Failed to initialize Python AI services: " << e.what() << std::endl;
        // 在实际应用中，这里应该导致程序退出或进入安全模式
    }
}

AIServiceManager::~AIServiceManager() {
    // 析构函数：释放Python对象
    py::gil_scoped_acquire acquire;
    asr_service_ = py::none();
    std::cout << "[C++] Python ASR Service instance released." << std::endl;
}

std::string AIServiceManager::transcribe(const std::string& audio_bytes, int sample_rate) {
    if (asr_service_.is_none()) {
        return "Error: ASR Service not loaded.";
    }

    // 每次调用Python代码都需要获取GIL，因为处理线程和主线程可能不同
    py::gil_scoped_acquire acquire; 
    try {
        // 将C++的std::string（字节序列）转换为Python的bytes对象
        py::bytes audio_py_bytes(audio_bytes);
        
        // 调用Python对象的 `transcribe_bytes` 方法
        // asr_service_.attr("...") 会查找并返回该名称的属性/方法
        py::object result_py_obj = asr_service_.attr("transcribe_bytes")(audio_py_bytes, sample_rate);
        
        // 将返回的Python字符串对象转换为C++的std::string
        return result_py_obj.cast<std::string>();

    } catch (py::error_already_set &e) {
        std::string error_msg = "[C++ Error] Python ASR call failed: ";
        error_msg += e.what();
        std::cerr << error_msg << std::endl;
        return error_msg;
    }
}


// --- pybind11 模块定义 ---
// 注意：这个模块主要是为了让Python可以测试C++（如test_bridge.py），
// 但在我们的混合架构中，主要用途是让C++能调用Python。
// 我们在此处仍然定义一个模块，但主程序不直接使用它。
// 上面的AIServiceManager类才是C++调用Python的关键。

std::string say_hello_from_cpp() {
    return "Hello from C++!";
}

PYBIND11_MODULE(my_project_bridge, m) {
    m.doc() = "pybind11 bridge for my_project";
    m.def("say_hello_from_cpp", &say_hello_from_cpp, "A function that says hello from C++");
}