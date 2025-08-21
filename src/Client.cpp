#include "Client.h"
#include "KwsService.h"
#include "KeyframeDetector.h"
#include <iostream>
#include <numeric>
#include <cmath>

// 构造函数实现
Client::Client(const std::string& uuid, const std::string& kws_model_path, const std::vector<std::string>& kws_keyword_paths)
    : uuid_(uuid),
      vad_is_active_(false),
      vad_consecutive_silent_chunks_(0),
      dialogue_last_speech_time_(std::chrono::steady_clock::now()),
      dialogue_accumulated_chunks_count_(0),
      kws_was_triggered_(false),
      dialogue_start_client_timestamp_(0.0)
{
    // 初始化KWS服务
    try {
        // TODO: 从配置中获取AccessKey和灵敏度
        std::string access_key = "7M0CcTw9MNobXpzy+zGgFIEqAaTXF+XmkvTTyi/zH04VWmcdp+csNg=="; 
        std::vector<float> sensitivities = {0.5f};
        kws_service_ = std::make_unique<KwsService>(access_key, kws_model_path, kws_keyword_paths, sensitivities);
    } catch (const std::exception& e) {
        std::cerr << "[Client] Failed to initialize KwsService for " << uuid_ << ": " << e.what() << std::endl;
        kws_service_ = nullptr;
    }

    // 初始化KeyframeDetector
    try {
        // TODO: 从配置中获取关键帧检测参数
        keyframe_detector_ = std::make_unique<KeyframeDetector>();
    } catch (const std::exception& e) {
        std::cerr << "[Client] Failed to initialize KeyframeDetector for " << uuid_ << ": " << e.what() << std::endl;
        keyframe_detector_ = nullptr;
    }
    
    std::cout << "[Log] Client " << uuid_ << " has been initialized with KWS and KeyframeDetector." << std::endl;
}

// 析构函数实现
Client::~Client() {
    std::cout << "[Log] Client " << uuid_ << " resources are being cleaned up." << std::endl;
    // std::unique_ptr 会自动调用 KwsService 和 KeyframeDetector 的析构函数，释放资源
}

// VAD检测方法实现
bool Client::isAudioActive(const std::vector<char>& audio_chunk) {
    if (audio_chunk.empty() || AUDIO_SAMPLE_WIDTH != 2) {
        return false;
    }
    const int16_t* pcm_data = reinterpret_cast<const int16_t*>(audio_chunk.data());
    size_t sample_count = audio_chunk.size() / sizeof(int16_t);

    if (sample_count == 0) return false;

    double sum_sq = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum_sq += static_cast<double>(pcm_data[i]) * pcm_data[i];
    }
    double rms = std::sqrt(sum_sq / sample_count);
    return rms > RMS_VAD_THRESHOLD;
}

// 重置累积器方法实现
void Client::resetAccumulator() {
    dialogue_buffer_.clear();
    dialogue_accumulated_chunks_count_ = 0;
    kws_was_triggered_ = false;
    dialogue_start_client_timestamp_ = 0.0;
    vad_is_active_ = false;
    vad_consecutive_silent_chunks_ = 0;
    // SUGGESTION: 重置计时器以避免旧时间戳导致立即触发
    dialogue_last_speech_time_ = std::chrono::steady_clock::now();
}

// 音频处理核心状态机实现
std::optional<std::pair<std::vector<char>, bool>> Client::processAudioChunk(const std::vector<char>& audio_chunk, double client_timestamp) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    bool is_active = isAudioActive(audio_chunk);
    auto now = std::chrono::steady_clock::now();

    if (is_active) {
        if (kws_service_ && !kws_was_triggered_) {
            // TODO: 在此添加重采样逻辑，如果KWS采样率与输入不同
            if (kws_service_->process(audio_chunk)) {
                kws_was_triggered_ = true;
                std::cout << "[KWS] Keyword detected for client " << uuid_ << "!" << std::endl;
            }
        }
        
        if (!vad_is_active_) {
            vad_is_active_ = true;
            if (dialogue_buffer_.empty()) {
                dialogue_start_client_timestamp_ = client_timestamp;
            }
        }
        dialogue_buffer_.push_back(audio_chunk);
        dialogue_accumulated_chunks_count_++;
        dialogue_last_speech_time_ = now;
        vad_consecutive_silent_chunks_ = 0;
    } else { // is not active
        if (vad_is_active_) {
            vad_consecutive_silent_chunks_++;
            if (vad_consecutive_silent_chunks_ <= SHORT_SILENCE_PADDING_CHUNKS) {
                dialogue_buffer_.push_back(audio_chunk);
                dialogue_accumulated_chunks_count_++;
            } else {
                vad_is_active_ = false;
            }
        }
    }

    long long silence_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - dialogue_last_speech_time_).count();

    bool should_process = false;
    if (!dialogue_buffer_.empty()) {
        // SUGGESTION: 简化触发逻辑以完全匹配Python版本，主要依赖静音超时和最大块数
        if (silence_duration_ms > DIALOGUE_SILENCE_TIMEOUT_MS && !vad_is_active_) {
            should_process = true;
        } else if (dialogue_accumulated_chunks_count_ >= MAX_ACCUMULATED_CHUNKS) {
            should_process = true;
        }
    }

    if (should_process) {
        std::vector<char> full_dialogue;
        size_t total_size = 0;
        for (const auto& chunk : dialogue_buffer_) {
            total_size += chunk.size();
        }
        full_dialogue.reserve(total_size);
        for (const auto& chunk : dialogue_buffer_) {
            full_dialogue.insert(full_dialogue.end(), chunk.begin(), chunk.end());
        }

        // 捕获KWS触发状态，因为resetAccumulator会重置它
        bool triggered_state = kws_was_triggered_;
        
        resetAccumulator();
        
        // 返回包含完整音频和KWS状态的pair
        return std::make_pair(full_dialogue, triggered_state);
    }

    return std::nullopt;
}

// 获取关键帧检测器实例
KeyframeDetector* Client::getKeyframeDetector() {
    return keyframe_detector_.get();
}