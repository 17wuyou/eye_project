#pragma once

#include <vector>
#include <cstdint>

/**
 * @brief ͨ�� RMS����������ֵ�����Ƶ���Ƿ��ڻ״̬��
 *
 * ���� Python �汾 is_audio_active_by_rms �ĸ����� C++ ʵ�֡�
 *
 * @param audio_chunk_bytes ���� 16λ��8λ PCM ��Ƶ���ݵ��ֽ�������
 * @param sample_width ÿ���������ֽ�����֧�� 1 �� 2����
 * @param rms_threshold �ж���Ƶ�Ƿ��Ծ��������ֵ��
 * @return ���������� RMS ֵ������ֵ���򷵻� true�����򷵻� false��
 */
bool is_audio_active_by_rms(const std::vector<unsigned char>& audio_chunk_bytes, int sample_width, double rms_threshold);