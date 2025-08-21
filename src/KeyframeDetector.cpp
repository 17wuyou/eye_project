#include "KeyframeDetector.h"
#include <iostream> // 用于日志输出

// 构造函数实现
KeyframeDetector::KeyframeDetector(
    float threshold, int history_size_config, const cv::Size& resize_dim,
    float motion_threshold_percent, int motion_diff_threshold_value,
    double cooldown_period_seconds, bool use_h_channel,
    const std::vector<int>& hist_channels_h, const std::vector<int>& hist_size_h,
    const std::vector<float>& hist_ranges_h, const std::vector<int>& hist_channels_gray,
    const std::vector<int>& hist_size_gray, const std::vector<float>& hist_ranges_gray)
    : threshold_(threshold),
      history_size_(history_size_config),
      resize_dim_(resize_dim),
      motion_threshold_(motion_threshold_percent / 100.0f),
      motion_diff_value_thresh_(motion_diff_threshold_value),
      cooldown_period_(cooldown_period_seconds),
      use_h_channel_(use_h_channel),
      // 初始化时间戳为过去一个冷却周期，确保第一次调用能立即检测
      last_keyframe_timestamp_(std::chrono::steady_clock::now() - std::chrono::duration_cast<std::chrono::steady_clock::duration>(cooldown_period_))
    //   last_keyframe_timestamp_(std::chrono::steady_clock::now() - cooldown_period_)
{
    if (use_h_channel_) {
        hist_channels_ = hist_channels_h;
        hist_size_ = hist_size_h;
        hist_ranges_ = hist_ranges_h;
    } else {
        hist_channels_ = hist_channels_gray;
        hist_size_ = hist_size_gray;
        hist_ranges_ = hist_ranges_gray;
    }
    
    std::cout << "[KeyframeDetector] Initialized. Threshold < " << threshold_
              << ", History Size: " << history_size_
              << ", Resize Dim: (" << resize_dim_.width << "x" << resize_dim_.height << ")"
              << ", Motion Threshold > " << motion_threshold_ * 100.0f << "%"
              << ", Cooldown: " << cooldown_period_seconds << "s"
              << ", Use H-Channel: " << (use_h_channel_ ? "Yes" : "No") << std::endl;
}

// 预处理和计算直方图的函数实现
std::pair<cv::Mat, cv::Mat> KeyframeDetector::_preprocess_frame_and_calc_hist(const std::vector<char>& frame_bytes) {
    try {
        // 使用IMREAD_COLOR从内存中的字节解码图像
        cv::Mat img_color = cv::imdecode(frame_bytes, cv::IMREAD_COLOR);
        if (img_color.empty()) {
            std::cerr << "[KeyframeDetector] Failed to decode image from bytes." << std::endl;
            return {{}, {}};
        }

        cv::Mat img_scaled_color = img_color;
        if (resize_dim_.width > 0 && resize_dim_.height > 0) {
            cv::resize(img_color, img_scaled_color, resize_dim_, 0, 0, cv::INTER_AREA);
        }

        cv::Mat gray_scaled;
        cv::cvtColor(img_scaled_color, gray_scaled, cv::COLOR_BGR2GRAY);

        cv::Mat target_img_for_hist; // 这里不能初始化为gray_scaled，因为可能被hsv覆盖
        if (use_h_channel_) {
            cv::Mat hsv_scaled;
            cv::cvtColor(img_scaled_color, hsv_scaled, cv::COLOR_BGR2HSV);
            target_img_for_hist = hsv_scaled;
        } else {
            target_img_for_hist = gray_scaled;
        }

        cv::Mat hist;
        // C++ API需要一个指向范围数组的指针数组
        const float* ranges[] = { hist_ranges_.data() };
        cv::calcHist(&target_img_for_hist, 1, hist_channels_.data(), cv::Mat(), hist, 1, hist_size_.data(), ranges, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        
        return {hist, gray_scaled};
    } catch (const cv::Exception& e) {
        std::cerr << "[KeyframeDetector] OpenCV error in preprocessing: " << e.what() << std::endl;
        return {{}, {}};
    }
}

// 判断关键帧的核心逻辑实现
bool KeyframeDetector::is_keyframe(const std::vector<char>& current_frame_bytes) {
    auto current_time = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(current_time - last_keyframe_timestamp_).count() < cooldown_period_.count()) {
        return false; // 处于冷却期
    }

    auto [current_hist, current_gray_scaled] = _preprocess_frame_and_calc_hist(current_frame_bytes);
    if (current_hist.empty() || current_gray_scaled.empty()) {
        return false; // 预处理失败
    }

    // 处理第一帧
    if (prev_hist_.empty() || prev_gray_frame_scaled_.empty()) {
        prev_hist_ = current_hist;
        prev_gray_frame_scaled_ = current_gray_scaled;
        if (history_size_ > 0) {
            historical_hists_.push_back(current_hist);
        }
        last_keyframe_timestamp_ = current_time; // 将第一帧视为"场景开始"并更新时间戳
        return false; // 首帧作为基准，不标记为关键帧
    }

    // 帧差法运动检测
    if (motion_threshold_ > 0.0f) {
        cv::Mat frame_diff, thresh_diff;
        cv::absdiff(prev_gray_frame_scaled_, current_gray_scaled, frame_diff);
        cv::threshold(frame_diff, thresh_diff, motion_diff_value_thresh_, 255, cv::THRESH_BINARY);
        float motion_score = static_cast<float>(cv::countNonZero(thresh_diff)) / static_cast<float>(thresh_diff.size().area());
        
        if (motion_score < motion_threshold_) {
            // FIX: 与Python逻辑保持一致，当运动量不足时，不更新prev_hist_。
            // 仅更新用于下一帧比较的灰度图。
            prev_gray_frame_scaled_ = current_gray_scaled; 
            return false; // 运动量不足，直接判定非关键帧
        }
    }

    // 与前一帧直方图比较
    double correlation_with_prev = cv::compareHist(prev_hist_, current_hist, cv::HISTCMP_CORREL);
    bool final_is_keyframe = false;

    if (correlation_with_prev < threshold_) {
        if (history_size_ == 0 || historical_hists_.empty()) {
            final_is_keyframe = true;
        } else {
            // 与历史队列比较，确保是新场景
            bool is_different_from_all_history = true;
            for (const auto& hist_in_queue : historical_hists_) {
                double correlation_with_historical = cv::compareHist(hist_in_queue, current_hist, cv::HISTCMP_CORREL);
                if (correlation_with_historical >= threshold_) {
                    is_different_from_all_history = false;
                    break;
                }
            }
            if (is_different_from_all_history) {
                final_is_keyframe = true;
            }
        }
    }

    // 更新状态
    if (final_is_keyframe) {
        // std::cout << "[KeyframeDetector] Keyframe DETECTED! Correlation: " << correlation_with_prev << std::endl;
        last_keyframe_timestamp_ = current_time;
        if (history_size_ > 0) {
            historical_hists_.push_back(current_hist);
            // 手动管理队列大小，等同于Python中deque(maxlen=...)的功能
            if (historical_hists_.size() > static_cast<size_t>(history_size_)) {
                historical_hists_.pop_front(); 
            }
        }
    }
    
    // 无论是否是关键帧，都将当前帧作为下一帧的比较基准
    prev_hist_ = current_hist;
    prev_gray_frame_scaled_ = current_gray_scaled;

    return final_is_keyframe;
}