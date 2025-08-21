#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <deque>
#include <chrono>

class KeyframeDetector {
public:
    // 构造函数，参数与Python版本完全对应
    KeyframeDetector(
        float threshold = 0.7,
        int history_size_config = 10,
        const cv::Size& resize_dim = cv::Size(0, 0), // (宽, 高), 0表示不缩放
        float motion_threshold_percent = 0.0,
        int motion_diff_threshold_value = 25,
        double cooldown_period_seconds = 2.0,
        bool use_h_channel = true,
        const std::vector<int>& hist_channels_h = {0},
        const std::vector<int>& hist_size_h = {180},
        const std::vector<float>& hist_ranges_h = {0, 180},
        const std::vector<int>& hist_channels_gray = {0},
        const std::vector<int>& hist_size_gray = {256},
        const std::vector<float>& hist_ranges_gray = {0, 256}
    );

    // 判断当前帧是否为关键帧的核心方法
    bool is_keyframe(const std::vector<char>& current_frame_bytes);

private:
    // 预处理帧并计算直方图的私有辅助函数
    std::pair<cv::Mat, cv::Mat> _preprocess_frame_and_calc_hist(const std::vector<char>& frame_bytes);

    // --- 配置参数 ---
    float threshold_;
    int history_size_;
    cv::Size resize_dim_;
    float motion_threshold_; // 已转换为 0.0-1.0 的比例
    int motion_diff_value_thresh_;
    std::chrono::duration<double> cooldown_period_;
    bool use_h_channel_;

    // 直方图计算参数
    std::vector<int> hist_channels_;
    std::vector<int> hist_size_;
    std::vector<float> hist_ranges_;

    // --- 状态变量 ---
    cv::Mat prev_hist_;
    cv::Mat prev_gray_frame_scaled_;
    std::deque<cv::Mat> historical_hists_;
    std::chrono::steady_clock::time_point last_keyframe_timestamp_;
};