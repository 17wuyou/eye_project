#include "audio_utils.h"
#include <cmath> // For std::sqrt
#include <numeric> // For std::accumulate (though a manual loop is clearer here)

bool is_audio_active_by_rms(const std::vector<unsigned char>& audio_chunk_bytes, int sample_width, double rms_threshold) {
    if (audio_chunk_bytes.empty()) {
        return false;
    }

    // 根据样本宽度计算样本数量
    const size_t num_samples = audio_chunk_bytes.size() / sample_width;
    if (num_samples == 0) {
        return false;
    }

    // 使用 double 避免平方后溢出
    double sum_sq = 0.0;

    if (sample_width == 2) {
        // 将字节缓冲区重新解释为 16 位有符号整数
        const int16_t* samples = reinterpret_cast<const int16_t*>(audio_chunk_bytes.data());
        for (size_t i = 0; i < num_samples; ++i) {
            // 在乘法前转换为 double
            double sample_val = static_cast<double>(samples[i]);
            sum_sq += sample_val * sample_val;
        }
    }
    else if (sample_width == 1) {
        // 8 位无符号整数，需要转换为 [-128, 127] 范围
        const uint8_t* samples = audio_chunk_bytes.data();
        for (size_t i = 0; i < num_samples; ++i) {
            // 转换为 int16_t 再转为 double
            double sample_val = static_cast<double>(static_cast<int16_t>(samples[i]) - 128);
            sum_sq += sample_val * sample_val;
        }
    }
    else {
        // 不支持的样本宽度
        return false;
    }

    double rms = std::sqrt(sum_sq / num_samples);
    return rms > rms_threshold;
}