# audio_utils.py
import logging
import configs

# 尝试导入我们自己编译的 C++ 模块。
# 如果失败，可能是模块未编译或未放在正确的位置。
try:
    from my_project_cpp import is_audio_active_by_rms as is_audio_active_by_rms_cpp
    CPP_EXTENSION_LOADED = True
    logging.getLogger(__name__).info("Successfully loaded C++ implementation for audio_utils.")
except ImportError:
    CPP_EXTENSION_LOADED = False
    logging.getLogger(__name__).error(
        "Failed to load C++ implementation for audio_utils. "
        "Falling back to pure Python version. "
        "Please ensure 'my_project_cpp' module is compiled and in the project root."
    )
    # 导入 NumPy 作为 Python 回退方案的依赖
    import numpy as np


logger = logging.getLogger(__name__)

RMS_VAD_THRESHOLD = configs.RMS_VAD_THRESHOLD  # 能量阈值

def is_audio_active_by_rms(audio_chunk_bytes: bytes, sample_width: int = 2) -> bool:
    """
    通过 RMS (均方根) 值检测音频块是否处于活动状态。
    此函数会优先调用高性能的 C++ 实现，如果 C++ 模块加载失败，则回退到纯 Python 实现。
    """
    if not audio_chunk_bytes:
        return False
    
    if CPP_EXTENSION_LOADED:
        try:
            # 直接调用 C++ 函数
            return is_audio_active_by_rms_cpp(
                audio_chunk_bytes=audio_chunk_bytes,
                sample_width=sample_width,
                rms_threshold=RMS_VAD_THRESHOLD
            )
        except Exception as e:
            logger.error(f"Error calling C++ RMS VAD implementation: {e}", exc_info=True)
            # 可以在这里选择是返回 False 还是尝试回退
            return False
    else:
        # 如果 C++ 模块未加载，执行原始的 Python 逻辑作为回退
        return _is_audio_active_by_rms_python(audio_chunk_bytes, sample_width)

def _is_audio_active_by_rms_python(audio_chunk_bytes: bytes, sample_width: int = 2) -> bool:
    """
    纯 Python 版本的 VAD 实现，作为 C++ 模块加载失败时的备用方案。
    """
    try:
        if sample_width == 2:
            audio_array = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
        elif sample_width == 1:
            audio_array = np.frombuffer(audio_chunk_bytes, dtype=np.uint8)
            audio_array = audio_array.astype(np.int16) - 128
        else:
            logger.warning(f"Unsupported sample_width for Python RMS VAD: {sample_width}.")
            return False

        if audio_array.size == 0:
            return False

        rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
        
        is_active = rms > RMS_VAD_THRESHOLD
        
        # 调试日志
        # logger.info(f"(Python VAD) RMS={rms:.2f}, Threshold={RMS_VAD_THRESHOLD}, Active={is_active}") 
        
        return is_active
    except Exception as e:
        logger.error(f"Error in Python RMS VAD calculation: {e}", exc_info=True)
        return False