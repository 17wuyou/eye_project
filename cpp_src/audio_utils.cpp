#include "audio_utils.h"
#include <cmath> // For std::sqrt
#include <numeric> // For std::accumulate (though a manual loop is clearer here)

bool is_audio_active_by_rms(const std::vector<unsigned char>& audio_chunk_bytes, int sample_width, double rms_threshold) {
    if (audio_chunk_bytes.empty()) {
        return false;
    }

    // ����������ȼ�����������
    const size_t num_samples = audio_chunk_bytes.size() / sample_width;
    if (num_samples == 0) {
        return false;
    }

    // ʹ�� double ����ƽ�������
    double sum_sq = 0.0;

    if (sample_width == 2) {
        // ���ֽڻ��������½���Ϊ 16 λ�з�������
        const int16_t* samples = reinterpret_cast<const int16_t*>(audio_chunk_bytes.data());
        for (size_t i = 0; i < num_samples; ++i) {
            // �ڳ˷�ǰת��Ϊ double
            double sample_val = static_cast<double>(samples[i]);
            sum_sq += sample_val * sample_val;
        }
    }
    else if (sample_width == 1) {
        // 8 λ�޷�����������Ҫת��Ϊ [-128, 127] ��Χ
        const uint8_t* samples = audio_chunk_bytes.data();
        for (size_t i = 0; i < num_samples; ++i) {
            // ת��Ϊ int16_t ��תΪ double
            double sample_val = static_cast<double>(static_cast<int16_t>(samples[i]) - 128);
            sum_sq += sample_val * sample_val;
        }
    }
    else {
        // ��֧�ֵ��������
        return false;
    }

    double rms = std::sqrt(sum_sq / num_samples);
    return rms > rms_threshold;
}