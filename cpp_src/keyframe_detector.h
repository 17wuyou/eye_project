#pragma once

#include <vector>
#include <deque>
#include <chrono>
#include <opencv2/opencv.hpp>

class KeyframeDetector {
public:
    KeyframeDetector(
        double threshold,
        int history_size,
        std::pair<int, int> resize_dim,
        double motion_threshold_percent,
        int motion_diff_threshold_value,
        double cooldown_period,
        bool use_h_channel,
        std::vector<int> hist_channels,
        std::vector<int> hist_size,
        std::vector<float> hist_ranges
    );

    bool is_keyframe(const std::vector<unsigned char>& frame_bytes);

private:
    std::pair<cv::Mat, cv::Mat> _preprocess_frame_and_calc_hist(const std::vector<unsigned char>& frame_bytes);

    // --- 配置参数 ---
    double m_threshold;
    cv::Size m_resize_dim;
    double m_motion_threshold; // 0.0 to 1.0
    int m_motion_diff_value_thresh;
    std::chrono::duration<double> m_cooldown_period;
    bool m_use_h_channel;

    // --- 直方图参数 ---
    std::vector<int> m_hist_channels;
    std::vector<int> m_hist_size;
    std::vector<float> m_hist_ranges;
    // 指针数组，因为 cv::calcHist 需要
    std::vector<const float*> m_hist_ranges_ptr;

    // --- 状态变量 ---
    cv::Mat m_prev_hist;
    cv::Mat m_prev_gray_frame_scaled;
    std::deque<cv::Mat> m_historical_hists;
    int m_history_size;
    std::chrono::time_point<std::chrono::steady_clock> m_last_keyframe_timestamp;
};