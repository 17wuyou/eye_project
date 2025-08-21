#include "KwsService.h"
#include <cstdint>
#include <iostream>

// 包含Porcupine的C语言头文件
extern "C" {
    #include "pv_porcupine.h"
}

// 构造函数：初始化Porcupine引擎
KwsService::KwsService(
    const std::string& access_key,
    const std::string& model_path,
    const std::vector<std::string>& keyword_paths,
    const std::vector<float>& sensitivities)
    : handle_(nullptr), sample_rate_(0), frame_length_(0) {

    // 将std::vector<std::string> 转换为 const char* const*
    std::vector<const char*> keyword_paths_c;
    for (const auto& path : keyword_paths) {
        keyword_paths_c.push_back(path.c_str());
    }

    pv_status_t status = pv_porcupine_init(
        access_key.c_str(),
        model_path.c_str(),
        static_cast<int32_t>(keyword_paths.size()),
        keyword_paths_c.data(),
        sensitivities.data(),
        &handle_
    );

    if (status != PV_STATUS_SUCCESS) {
        // 如果初始化失败，抛出异常并附带错误信息
        throw std::runtime_error(
            "Failed to initialize Porcupine: " + std::string(pv_status_to_string(status))
        );
    }
    
    // 初始化成功后，获取采样率和帧长度
    sample_rate_ = pv_sample_rate();
    frame_length_ = pv_porcupine_frame_length();

    std::cout << "[KwsService] Porcupine engine initialized successfully. "
              << "Sample Rate: " << sample_rate_ << ", Frame Length: " << frame_length_ << std::endl;
}

// 析构函数：释放资源
KwsService::~KwsService() {
    if (handle_ != nullptr) {
        pv_porcupine_delete(handle_);
        handle_ = nullptr;
        std::cout << "[KwsService] Porcupine engine resources released." << std::endl;
    }
}

// 获取引擎采样率
int KwsService::get_sample_rate() const {
    return sample_rate_;
}

// 获取引擎期望的帧长度
int KwsService::get_frame_length() const {
    return frame_length_;
}

// 处理音频块
bool KwsService::process(const std::vector<char>& pcm_chunk) {
    if (handle_ == nullptr || pcm_chunk.empty()) {
        return false;
    }

    // 将char*转换为const int16_t*
    const int16_t* pcm = reinterpret_cast<const int16_t*>(pcm_chunk.data());
    const size_t pcm_sample_count = pcm_chunk.size() / sizeof(int16_t);

    // Porcupine需要按固定帧长度处理
    for (size_t i = 0; i + frame_length_ <= pcm_sample_count; i += frame_length_) {
        const int16_t* frame = pcm + i;
        int32_t keyword_index = -1;
        
        pv_status_t status = pv_porcupine_process(handle_, frame, &keyword_index);
        if (status != PV_STATUS_SUCCESS) {
            std::cerr << "[KwsService] Error during processing: " << pv_status_to_string(status) << std::endl;
            return false;
        }

        if (keyword_index != -1) {
            // 检测到关键词！
            return true;
        }
    }

    return false; // 未检测到关键词
}