# keyframe_detector.py
import cv2
import numpy as np
import logging
from collections import deque
import time # 用于冷却机制

logger = logging.getLogger(__name__)

class KeyframeDetector:
    def __init__(self, threshold=0.7,
                 history_size_config=10,
                 resize_dim=None,
                 motion_threshold_percent=0.0, # 运动像素百分比阈值，0表示禁用帧差
                 motion_diff_threshold_value=25, # 像素差异阈值
                 cooldown_period=2.0,
                 use_h_channel=True,
                 hist_channels_h=None, hist_size_h=None, hist_ranges_h=None,
                 hist_channels_gray=None, hist_size_gray=None, hist_ranges_gray=None):
        """
        初始化增强版关键帧检测器。
        Args:
            threshold (float): 直方图比较的相关性阈值。如果当前帧与参考帧的直方图相关性 *小于* 此阈值，则认为差异显著。
            history_size_config (int): 存储历史场景直方图的队列大小。
            resize_dim (tuple, optional): 图像缩放尺寸 (宽, 高)。None表示不缩放。
            motion_threshold_percent (float): 帧差法中，运动像素占总像素的百分比阈值 (0.0-100.0)。
                                              例如，0.5 表示画面中至少有0.5%的像素发生显著变化。
                                              设为0则禁用帧差法。
            motion_diff_threshold_value (int): 帧差计算时，单个像素被视为“变化”的差异阈值 (0-255)。
            cooldown_period (float): 关键帧检测的冷却时间 (秒)。
            use_h_channel (bool): 是否使用HSV空间的H通道计算直方图。
            hist_channels_h (list, optional): H通道直方图的通道。默认为 [0]。
            hist_size_h (list, optional): H通道直方图的大小。默认为 [180]。
            hist_ranges_h (list, optional): H通道直方图的范围。默认为 [0, 180]。
            hist_channels_gray (list, optional): 灰度直方图的通道。默认为 [0]。
            hist_size_gray (list, optional): 灰度直方图的大小。默认为 [256]。
            hist_ranges_gray (list, optional): 灰度直方图的范围。默认为 [0, 256]。
        """
        self.threshold = threshold
        self.resize_dim = resize_dim
        self.motion_threshold = motion_threshold_percent / 100.0 # 转换为0.0-1.0的比例
        self.motion_diff_value_thresh = motion_diff_threshold_value
        self.cooldown_period = cooldown_period
        self.use_h_channel = use_h_channel

        if self.use_h_channel:
            self.hist_channels = hist_channels_h if hist_channels_h is not None else [0]
            self.hist_size = hist_size_h if hist_size_h is not None else [180]
            self.hist_ranges = hist_ranges_h if hist_ranges_h is not None else [0, 180]
        else:
            self.hist_channels = hist_channels_gray if hist_channels_gray is not None else [0]
            self.hist_size = hist_size_gray if hist_size_gray is not None else [256]
            self.hist_ranges = hist_ranges_gray if hist_ranges_gray is not None else [0, 256]

        self.prev_hist = None
        self.prev_gray_frame_scaled = None # 用于帧差法
        self.last_keyframe_timestamp = 0  # 用于冷却机制

        self.history_size = history_size_config
        if self.history_size > 0:
            self.historical_hists = deque(maxlen=self.history_size)
        else:
            self.historical_hists = None

        logger.info(f"关键帧检测器已初始化: 阈值(相关性) < {self.threshold}, 历史队列大小: {self.history_size}, "
                    f"缩放尺寸: {self.resize_dim}, 运动阈值(比例): > {self.motion_threshold:.4f} (像素差异 > {self.motion_diff_value_thresh}), "
                    f"冷却时间: {self.cooldown_period}s, 使用H通道: {self.use_h_channel}")
        if self.use_h_channel:
            logger.info(f"  H通道直方图参数: 通道={self.hist_channels}, 大小={self.hist_size}, 范围={self.hist_ranges}")
        else:
            logger.info(f"  灰度直方图参数: 通道={self.hist_channels}, 大小={self.hist_size}, 范围={self.hist_ranges}")


    def _preprocess_frame_and_calc_hist(self, frame_bytes):
        """
        解码、预处理 (缩放) 图像帧，计算其灰度版本 (用于帧差) 和目标直方图。
        Args:
            frame_bytes (bytes): 图像帧的字节数据 (例如 JPEG 格式)。
        Returns:
            tuple (np.ndarray, np.ndarray) or (None, None):
                - 计算得到的直方图。
                - 缩放后的灰度图像。
                如果解码或处理失败则返回 (None, None)。
        """
        try:
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            img_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_color is None:
                logger.warning("从字节解码图像失败。")
                return None, None

            # 1. 缩放 (如果配置了)
            img_scaled_color = img_color
            if self.resize_dim and self.resize_dim[0] > 0 and self.resize_dim[1] > 0:
                try:
                    img_scaled_color = cv2.resize(img_color, self.resize_dim, interpolation=cv2.INTER_AREA)
                except Exception as e_resize:
                    logger.warning(f"图像缩放至 {self.resize_dim} 失败: {e_resize}. 使用原始图像。")
                    img_scaled_color = img_color # 出错则用回原图

            # 2. 准备灰度图 (用于帧差)
            gray_scaled = cv2.cvtColor(img_scaled_color, cv2.COLOR_BGR2GRAY)

            # 3. 计算直方图
            target_img_for_hist = gray_scaled # 默认或 use_h_channel=False 时
            if self.use_h_channel:
                hsv_scaled = cv2.cvtColor(img_scaled_color, cv2.COLOR_BGR2HSV)
                target_img_for_hist = hsv_scaled
            
            hist = cv2.calcHist([target_img_for_hist], self.hist_channels, None, self.hist_size, self.hist_ranges)
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            return hist, gray_scaled

        except Exception as e:
            logger.error(f"预处理帧并计算直方图时出错: {e}", exc_info=False) # 减少日志冗余
            return None, None

    def is_keyframe(self, current_frame_bytes):
        """
        根据多种策略判断当前帧是否为关键帧。
        策略包括：冷却、帧差运动检测、与前一帧的直方图差异、与历史场景的直方图差异。
        Args:
            current_frame_bytes (bytes): 当前视频帧的字节数据。
        Returns:
            bool: 如果是关键帧则返回 True，否则返回 False。
        """
        current_time = time.time()

        # 1. 冷却机制检查
        if current_time - self.last_keyframe_timestamp < self.cooldown_period:
            # logger.debug("处于冷却期，跳过关键帧检测。")
            return False

        current_hist, current_gray_scaled = self._preprocess_frame_and_calc_hist(current_frame_bytes)

        if current_hist is None or current_gray_scaled is None:
            logger.debug("当前帧直方图或灰度图计算失败，无法判断是否为关键帧。")
            return False

        # 处理第一帧 (或刚重置后)
        if self.prev_hist is None or self.prev_gray_frame_scaled is None:
            self.prev_hist = current_hist
            self.prev_gray_frame_scaled = current_gray_scaled
            if self.historical_hists is not None:
                self.historical_hists.append(current_hist)
            self.last_keyframe_timestamp = current_time # 认为第一帧是一个"场景的开始"
            logger.debug("首帧或重置后第一帧，设为参考，非严格关键帧（但更新时间戳）。")
            return False # 通常首帧不立即标记为关键帧，而是作为基准

        # 2. 帧差法运动检测 (如果启用)
        if self.motion_threshold > 0:
            frame_diff = cv2.absdiff(self.prev_gray_frame_scaled, current_gray_scaled)
            _, thresh_diff = cv2.threshold(frame_diff, self.motion_diff_value_thresh, 255, cv2.THRESH_BINARY)
            motion_score = np.count_nonzero(thresh_diff) / thresh_diff.size
            
            if motion_score < self.motion_threshold:
                # logger.debug(f"运动量不足 ({motion_score*100:.2f}% < {self.motion_threshold*100:.2f}%)，非关键帧。")
                # 即使运动不足，也更新前一帧的灰度图，以便下一轮比较
                self.prev_gray_frame_scaled = current_gray_scaled
                # 对于直方图，如果运动不足，通常场景变化不大，可以考虑也更新 prev_hist，
                # 或者保持 prev_hist 不变，直到有显著运动+直方图变化。
                # 当前选择：如果运动不足，则认为场景未变，不进行后续直方图比较，prev_hist 也不急于更新为当前帧的，
                # 而是等待一个真正“动起来”的帧。
                # 但如果 prev_hist 长时间不更新，而场景在缓慢变化，可能会导致后续一旦有运动就误判。
                # 一个折中：如果运动不足，但这是连续的第N个运动不足的帧，可能也要更新一下基准。
                # 目前简单处理：运动不足则认为不是关键帧，并更新 prev_gray_frame_scaled。
                # prev_hist 的更新放到后面，如果最终不是关键帧，则 prev_hist 更新为 current_hist。
                return False # 如果运动量不足，直接判定非关键帧
            else:
                 logger.debug(f"帧差检测到足够运动: {motion_score*100:.2f}% >= {self.motion_threshold*100:.2f}%")


        # 3. 与最近的前一帧直方图比较
        correlation_with_prev = cv2.compareHist(self.prev_hist, current_hist, cv2.HISTCMP_CORREL)
        logger.debug(f"与前一帧的直方图相关性: {correlation_with_prev:.4f} (阈值 < {self.threshold})")

        is_kf_candidate_due_to_prev_diff = False
        if correlation_with_prev < self.threshold: # 与前一帧差异显著
            is_kf_candidate_due_to_prev_diff = True
            logger.debug(f"与前一帧差异显著 (相关性 {correlation_with_prev:.4f})。")

        final_is_keyframe = False
        if is_kf_candidate_due_to_prev_diff:
            if self.historical_hists is None or not self.historical_hists: # 如果不使用历史队列或队列为空
                final_is_keyframe = True
                logger.debug("无历史队列或队列为空，候选帧直接成为关键帧。")
            else:
                # 4. 与历史队列中的所有直方图比较，确保与所有近期场景都不同
                is_different_from_all_history = True
                for i, hist_in_queue in enumerate(list(self.historical_hists)): # 使用list复制一份，防止迭代时修改
                    correlation_with_historical = cv2.compareHist(hist_in_queue, current_hist, cv2.HISTCMP_CORREL)
                    # logger.debug(f"  与历史队列帧 {i} 的相关性: {correlation_with_historical:.4f}")
                    if correlation_with_historical >= self.threshold: # 如果与历史中某帧相似
                        is_different_from_all_history = False
                        logger.debug(f"与历史队列帧 {i} 相似 (相关性: {correlation_with_historical:.4f} >= {self.threshold})。非新场景。")
                        break 
                
                if is_different_from_all_history:
                    final_is_keyframe = True
                    logger.debug("与所有历史帧均存在显著差异，确认为关键帧。")
        else:
            # logger.debug("与前一帧不够差异，非关键帧。")
            pass

        # 更新状态
        if final_is_keyframe:
            logger.info(f"检测到关键帧! 原因: 与前帧差异 ({correlation_with_prev:.4f} < {self.threshold}) 且与历史场景均不同。")
            self.last_keyframe_timestamp = current_time # 更新冷却时间戳
            if self.historical_hists is not None:
                self.historical_hists.append(current_hist) # 将新关键帧的直方图加入历史
            self.prev_hist = current_hist
        else:
            # 如果不是关键帧， prev_hist 通常更新为当前帧的直方图，
            # 这样下一帧的比较基准就是当前帧。
            # 这有助于跟踪场景的渐变。
            self.prev_hist = current_hist
            # logger.debug("非关键帧，prev_hist 更新为当前帧直方图。")


        self.prev_gray_frame_scaled = current_gray_scaled # 始终更新上一帧的灰度图

        return final_is_keyframe