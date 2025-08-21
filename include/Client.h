#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include <optional>
#include <memory>
#include <utility> // For std::pair

// 前向声明，以避免在头文件中引入重量级的依赖，减少编译时间
class KwsService;
class KeyframeDetector;

/**
 * @class Client
 * @brief 代表一个连接到服务器的客户端及其所有相关状态。
 *
 * 这个类封装了单个客户端的所有实时处理逻辑，包括VAD、KWS、
 * 音频对话累积和关键帧检测。它是线程安全的。
 */
class Client {
public:
    /**
     * @brief 构造函数，初始化客户端所有状态和子服务。
     * @param uuid 客户端的唯一标识符。
     * @param kws_model_path Porcupine通用模型文件的路径。
     * @param kws_keyword_paths 关键词模型文件的路径列表。
     */
    explicit Client(
        const std::string& uuid, 
        const std::string& kws_model_path, 
        const std::vector<std::string>& kws_keyword_paths);

    /**
     * @brief 析构函数，确保所有资源被正确释放。
     */
    ~Client();

    /**
     * @brief 处理一个传入的音频块。
     *
     * 这是音频处理的核心状态机。它执行VAD、KWS和对话累积。
     * 当一个完整的对话片段（因静音或超时）结束时，它会返回该片段的
     * 所有音频数据以及该片段是否由KWS触发的状态。
     *
     * @param audio_chunk 16-bit PCM音频数据块。
     * @param client_timestamp 客户端发送此数据块时的时间戳。
     * @return 如果对话片段已准备好被处理，则返回包含{完整音频, KWS触发状态}的optional；否则返回std::nullopt。
     */
    std::optional<std::pair<std::vector<char>, bool>> processAudioChunk(
        const std::vector<char>& audio_chunk, 
        double client_timestamp);

    /**
     * @brief 获取此客户端关联的关键帧检测器实例。
     * @return 指向KeyframeDetector对象的指针。
     */
    KeyframeDetector* getKeyframeDetector();

    // 禁止拷贝和赋值，确保每个Client实例的唯一性
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

private:
    /**
     * @brief VAD（语音活动检测）逻辑，判断音频块是否包含语音。
     */
    bool isAudioActive(const std::vector<char>& audio_chunk);
    
    /**
     * @brief 重置音频累积器的状态，在处理完一个对话后调用。
     */
    void resetAccumulator();

    // --- 状态变量 ---
    std::string uuid_;
    bool vad_is_active_;
    int vad_consecutive_silent_chunks_;
    std::deque<std::vector<char>> dialogue_buffer_;
    std::chrono::steady_clock::time_point dialogue_last_speech_time_;
    int dialogue_accumulated_chunks_count_;
    bool kws_was_triggered_;
    double dialogue_start_client_timestamp_;
    
    // 用于保护所有成员变量的互斥锁，确保线程安全
    std::mutex data_mutex_;

    // --- 配置参数 (硬编码，未来应从配置文件加载) ---
    // FIX: 将VAD阈值从120.0修正为10.0，以匹配Python项目的配置
    const double RMS_VAD_THRESHOLD = 10.0;
    // NOTE: 以下参数与Python configs.py 中的值对应
    const int DIALOGUE_SILENCE_TIMEOUT_MS = 1500;
    const int MAX_ACCUMULATED_CHUNKS = 150; // 对应python: int((30 * 1000) / 200)
    const int SHORT_SILENCE_PADDING_CHUNKS = 2;
    const int AUDIO_SAMPLE_WIDTH = 2; // 16-bit

    // --- 子服务实例 ---
    std::unique_ptr<KwsService> kws_service_;
    std::unique_ptr<KeyframeDetector> keyframe_detector_;
};