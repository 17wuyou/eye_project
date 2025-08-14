# keyframe_detector.py
import logging
import configs

# 同样，安全地尝试导入 C++ 模块
try:
    from my_project_cpp import KeyframeDetector as KeyframeDetectorCpp
    CPP_EXTENSION_LOADED = True
    logging.getLogger(__name__).info("Successfully loaded C++ implementation for KeyframeDetector.")
except ImportError:
    CPP_EXTENSION_LOADED = False
    logging.getLogger(__name__).error(
        "Failed to load C++ implementation for KeyframeDetector. "
        "The KeyframeDetector will be non-functional. "
        "Please ensure 'my_project_cpp' module is compiled and in the project root."
    )
    # 定义一个假的类，以便在导入失败时程序不会崩溃
    class KeyframeDetectorCpp:
        def __init__(self, *args, **kwargs):
            self.is_dummy = True
            logger.critical("Using dummy KeyframeDetector. Keyframe detection will NOT work.")
        def is_keyframe(self, current_frame_bytes):
            return False


logger = logging.getLogger(__name__)

def KeyframeDetector(
    threshold=configs.KEYFRAME_DETECTOR_THRESHOLD,
    history_size_config=configs.KEYFRAME_HISTORY_SIZE,
    resize_dim=configs.KEYFRAME_RESIZE_DIM,
    motion_threshold_percent=configs.KEYFRAME_MOTION_THRESHOLD_PERCENT,
    motion_diff_threshold_value=configs.KEYFRAME_MOTION_DIFF_THRESHOLD_VALUE,
    cooldown_period=configs.KEYFRAME_COOLDOWN_SECONDS,
    use_h_channel=configs.KEYFRAME_USE_H_CHANNEL_FOR_HIST,
    hist_channels_h=configs.KEYFRAME_HIST_CHANNELS_H, 
    hist_size_h=configs.KEYFRAME_HIST_SIZE_H, 
    hist_ranges_h=configs.KEYFRAME_HIST_RANGES_H,
    hist_channels_gray=configs.KEYFRAME_HIST_CHANNELS_GRAY, 
    hist_size_gray=configs.KEYFRAME_HIST_SIZE_GRAY, 
    hist_ranges_gray=configs.KEYFRAME_HIST_RANGES_GRAY
):
    """
    工厂函数，用于创建和初始化 KeyframeDetector 实例。
    它会优先创建 C++ 版本的实例。
    """
    if not CPP_EXTENSION_LOADED:
        logger.error("Cannot create KeyframeDetector instance because C++ module is not loaded.")
        # 返回一个什么都不做的假对象
        return KeyframeDetectorCpp()

    # 将直方图参数打包成一个字典，以匹配 C++ 绑定中构造函数的要求
    hist_params = {
        "hist_channels_h": hist_channels_h,
        "hist_size_h": hist_size_h,
        "hist_ranges_h": hist_ranges_h,
        "hist_channels_gray": hist_channels_gray,
        "hist_size_gray": hist_size_gray,
        "hist_ranges_gray": hist_ranges_gray,
    }

    logger.info(f"Initializing C++ KeyframeDetector with threshold: {threshold}, history: {history_size_config}...")
    
    # 创建 C++ 类的实例
    instance = KeyframeDetectorCpp(
        threshold=threshold,
        history_size_config=history_size_config,
        resize_dim=resize_dim,
        motion_threshold_percent=motion_threshold_percent,
        motion_diff_threshold_value=motion_diff_threshold_value,
        cooldown_period=cooldown_period,
        use_h_channel=use_h_channel,
        hist_params=hist_params
    )
    logger.info("C++ KeyframeDetector instance created successfully.")
    return instance

# 原来的 Python KeyframeDetector 类可以完全删除，因为我们不再使用它。
# 如果需要，也可以像 audio_utils 一样保留它作为备用方案，但对于这么复杂的类，
# 简单的工厂模式通常更清晰。