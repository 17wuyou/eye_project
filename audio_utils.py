# audio_utils.py
import numpy as np
import logging
import configs
logger = logging.getLogger(__name__)

RMS_VAD_THRESHOLD = configs.RMS_VAD_THRESHOLD  # 能量阈值 - 需要您根据实际情况调整！

def is_audio_active_by_rms(audio_chunk_bytes: bytes, sample_width: int = 2) -> bool:
    if not audio_chunk_bytes:
        return False
    try:
        if sample_width == 2:
            audio_array = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
        elif sample_width == 1:
            audio_array = np.frombuffer(audio_chunk_bytes, dtype=np.uint8)
            audio_array = audio_array.astype(np.int16) - 128
        else:
            logger.warning(f"Unsupported sample_width for RMS VAD: {sample_width}.")
            return False

        if audio_array.size == 0:
            return False

        rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
        
        is_active = rms > RMS_VAD_THRESHOLD
        # 这条日志会非常频繁，调试VAD阈值时开启，平时可以注释掉或改为DEBUG级别
        logger.info(f"RMS VAD Check: RMS={rms:.2f}, Threshold={RMS_VAD_THRESHOLD}, Active={is_active}") 
        
        return is_active
    except Exception as e:
        logger.error(f"Error in RMS VAD calculation: {e}", exc_info=True)
        return False