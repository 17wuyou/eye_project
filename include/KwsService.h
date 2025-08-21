#pragma once

#include <string>
#include <vector>
#include <stdexcept>

// 前向声明来自 pv_porcupine.h 的C结构体，避免在头文件中暴露C API细节
struct pv_porcupine;

class KwsService {
public:
    /**
     * @brief 构造函数，初始化Porcupine引擎。
     * @param access_key PicoVoice AccessKey.
     * @param model_path Porcupine通用模型文件路径(.pv).
     * @param keyword_paths 关键词模型文件路径列表(.ppn).
     * @param sensitivities 对应关键词的灵敏度列表(0.0 to 1.0).
     * @throw std::runtime_error 如果初始化失败。
     */
    KwsService(
        const std::string& access_key,
        const std::string& model_path,
        const std::vector<std::string>& keyword_paths,
        const std::vector<float>& sensitivities);

    /**
     * @brief 析构函数，自动释放Porcupine引擎资源。
     */
    ~KwsService();

    /**
     * @brief 处理一个音频块以检测关键词。
     * @param pcm_chunk 16-bit little-endian PCM音频块。
     * @return 如果检测到关键词则返回true，否则返回false。
     */
    bool process(const std::vector<char>& pcm_chunk);

    // 获取引擎的采样率
    int get_sample_rate() const;

    // 获取引擎期望的帧长度
    int get_frame_length() const;

private:
    // 删除拷贝构造函数和赋值操作，防止意外复制
    KwsService(const KwsService&) = delete;
    KwsService& operator=(const KwsService&) = delete;

    // 指向Porcupine C语言库引擎实例的指针
    pv_porcupine* handle_;
    int sample_rate_;
    int frame_length_;
};