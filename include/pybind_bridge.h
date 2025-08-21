// include/pybind_bridge.h
#pragma once
#include <string>
#include <pybind11/embed.h>

namespace py = pybind11;

class AIServiceManager {
public:
    AIServiceManager();
    ~AIServiceManager();

    // 调用Python ASR服务进行语音识别
    std::string transcribe(const std::string& audio_bytes, int sample_rate);

private:
    py::object asr_service_; // 持有Python asr_service_instance对象
    py::object llm_service_; // 为未来扩展预留
    // ... 其他服务
};