#pragma once

#include <vector>
#include <cstdint>

/**
 * @brief 通过 RMS（均方根）值检测音频块是否处于活动状态。
 *
 * 这是 Python 版本 is_audio_active_by_rms 的高性能 C++ 实现。
 *
 * @param audio_chunk_bytes 包含 16位或8位 PCM 音频数据的字节向量。
 * @param sample_width 每个样本的字节数（支持 1 或 2）。
 * @param rms_threshold 判断音频是否活跃的能量阈值。
 * @return 如果计算出的 RMS 值大于阈值，则返回 true，否则返回 false。
 */
bool is_audio_active_by_rms(const std::vector<unsigned char>& audio_chunk_bytes, int sample_width, double rms_threshold);