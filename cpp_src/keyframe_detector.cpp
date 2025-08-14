#include "keyframe_detector.h"

KeyframeDetector::KeyframeDetector(
    double threshold, int history_size, std::pair<int, int> resize_dim,
    double motion_threshold_percent, int motion_diff_threshold_value,
    double cooldown_period, bool use_h_channel,
    std::vector<int> hist_channels, std::vector<int> hist_size, std::vector<float> hist_ranges
) : m_threshold(threshold),
m_history_size(history_size),
m_resize_dim(resize_dim.first, resize_dim.second),
m_motion_threshold(motion_threshold_percent / 100.0),
m_motion_diff_value_thresh(motion_diff_threshold_value),
m_cooldown_period(cooldown_period),
m_use_h_channel(use_h_channel),
m_hist_channels(std::move(hist_channels)),
m_hist_size(std::move(hist_size)),
m_hist_ranges(std::move(hist_ranges)),
m_last_keyframe_timestamp(std::chrono::steady_clock::now())
{
    if (m_history_size > 0) {
        m_historical_hists = std::deque<cv::Mat>();
    }
    // cv::calcHist 需要一个 const float* 的数组
    m_hist_ranges_ptr.push_back(m_hist_ranges.data());
}


std::pair<cv::Mat, cv::Mat> KeyframeDetector::_preprocess_frame_and_calc_hist(const std::vector<unsigned char>& frame_bytes) {
    try {
        cv::Mat img_color = cv::imdecode(frame_bytes, cv::IMREAD_COLOR);
        if (img_color.empty()) {
            return { cv::Mat(), cv::Mat() };
        }

        cv::Mat img_scaled_color = img_color;
        if (m_resize_dim.width > 0 && m_resize_dim.height > 0) {
            cv::resize(img_color, img_scaled_color, m_resize_dim, 0, 0, cv::INTER_AREA);
        }

        cv::Mat gray_scaled;
        cv::cvtColor(img_scaled_color, gray_scaled, cv::COLOR_BGR2GRAY);

        cv::Mat hist;
        cv::Mat target_img_for_hist;

        if (m_use_h_channel) {
            cv::Mat hsv_scaled;
            cv::cvtColor(img_scaled_color, hsv_scaled, cv::COLOR_BGR2HSV);
            target_img_for_hist = hsv_scaled;
        }
        else {
            target_img_for_hist = gray_scaled;
        }

        cv::calcHist(&target_img_for_hist, 1, m_hist_channels.data(), cv::Mat(), hist, 1, m_hist_size.data(), m_hist_ranges_ptr.data(), true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        return { hist, gray_scaled };
    }
    catch (const cv::Exception& e) {
        // 在生产环境中，这里应该使用日志库
        // std::cerr << "OpenCV exception in preprocess: " << e.what() << std::endl;
        return { cv::Mat(), cv::Mat() };
    }
}


bool KeyframeDetector::is_keyframe(const std::vector<unsigned char>& current_frame_bytes) {
    auto current_time = std::chrono::steady_clock::now();
    if (current_time - m_last_keyframe_timestamp < m_cooldown_period) {
        return false;
    }

    auto [current_hist, current_gray_scaled] = _preprocess_frame_and_calc_hist(current_frame_bytes);
    if (current_hist.empty() || current_gray_scaled.empty()) {
        return false;
    }

    if (m_prev_hist.empty() || m_prev_gray_frame_scaled.empty()) {
        m_prev_hist = current_hist;
        m_prev_gray_frame_scaled = current_gray_scaled;
        if (m_history_size > 0) {
            m_historical_hists.push_back(current_hist);
        }
        m_last_keyframe_timestamp = current_time;
        return false;
    }

    if (m_motion_threshold > 0) {
        cv::Mat frame_diff, thresh_diff;
        cv::absdiff(m_prev_gray_frame_scaled, current_gray_scaled, frame_diff);
        cv::threshold(frame_diff, thresh_diff, m_motion_diff_value_thresh, 255, cv::THRESH_BINARY);
        double motion_score = static_cast<double>(cv::countNonZero(thresh_diff)) / thresh_diff.size().area();

        if (motion_score < m_motion_threshold) {
            m_prev_gray_frame_scaled = current_gray_scaled;
            m_prev_hist = current_hist; // 如果运动不足，也更新基准以适应渐变
            return false;
        }
    }

    double correlation_with_prev = cv::compareHist(m_prev_hist, current_hist, cv::HISTCMP_CORREL);
    bool final_is_keyframe = false;

    if (correlation_with_prev < m_threshold) {
        if (m_history_size == 0) {
            final_is_keyframe = true;
        }
        else {
            bool is_different_from_all_history = true;
            for (const auto& hist_in_queue : m_historical_hists) {
                double correlation_with_historical = cv::compareHist(hist_in_queue, current_hist, cv::HISTCMP_CORREL);
                if (correlation_with_historical >= m_threshold) {
                    is_different_from_all_history = false;
                    break;
                }
            }
            if (is_different_from_all_history) {
                final_is_keyframe = true;
            }
        }
    }

    if (final_is_keyframe) {
        m_last_keyframe_timestamp = current_time;
        if (m_history_size > 0) {
            if (m_historical_hists.size() >= m_history_size) {
                m_historical_hists.pop_front();
            }
            m_historical_hists.push_back(current_hist);
        }
    }

    m_prev_hist = current_hist;
    m_prev_gray_frame_scaled = current_gray_scaled;

    return final_is_keyframe;
}