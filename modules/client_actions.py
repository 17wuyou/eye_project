# modules/client_actions.py
import threading, time, logging
from collections import deque

from . import state_manager
from .performance_utils import initialize_performance_data
import configs

logger = logging.getLogger(__name__)

def initialize_client_locks(client_uuid: str):
    if client_uuid not in state_manager.client_data_locks:
        state_manager.client_data_locks[client_uuid] = {
            "performance_lock": threading.Lock(),
            "event_data_lock": threading.Lock()
        }

def remove_client_locks(client_uuid: str):
    if client_uuid in state_manager.client_data_locks:
        del state_manager.client_data_locks[client_uuid]

def initialize_client_accumulator(client_uuid: str):
    state_manager.client_audio_accumulators[client_uuid] = {
        'vad_is_active': False,
        'vad_consecutive_silent_chunks': 0,
        'dialogue_buffer': deque(),
        'dialogue_last_speech_time_ns': time.monotonic_ns(),
        'dialogue_accumulated_chunks_count': 0,
        'kws_was_triggered': False, # 新增：标记当前对话是否由KWS触发
        'dialogue_start_client_timestamp': None, # 新增: 记录对话开始时第一个音频块的客户端时间戳
    }
    if configs.ENABLE_PERFORMANCE_MONITORING:
        initialize_performance_data(client_uuid)

def initialize_client_event_data(client_uuid: str):
    lock = state_manager.client_data_locks.get(client_uuid, {}).get("event_data_lock")
    if lock:
        with lock:
            state_manager.client_event_data[client_uuid] = {
                "current_event_folder_name": None,
                "current_event_start_time_unix": None,
                "current_event_keyframe_image_bytes": None,
                "current_event_keyframe_timestamp_unix": None,
                "accumulated_asr_for_event": deque(),
                "event_counter": 0
            }
    else:
        logger.error(f"客户端 {client_uuid}: 事件数据锁未找到，无法初始化事件数据。")

def cleanup_client_resources(client_uuid: str, client_desc: str):
    logger.info(f"正在为 {client_desc} 进行资源清理...")

    # KWS 检测器
    if client_uuid in state_manager.client_kws_detectors:
        try:
            state_manager.client_kws_detectors[client_uuid].delete()
        except Exception as e:
            logger.error(f"清理客户端 {client_uuid} 的KWS检测器时出错: {e}")
        del state_manager.client_kws_detectors[client_uuid]
    
    # KWS 和 LLM 相关状态
    for state_dict in [
        state_manager.client_kws_audio_buffer,
        state_manager.llm_query_active,
        state_manager.last_video_frames
    ]:
        if client_uuid in state_dict:
            del state_dict[client_uuid]
            
    # 关键帧检测器和时间
    if client_uuid in state_manager.keyframe_detectors:
        del state_manager.keyframe_detectors[client_uuid]
    if client_uuid in state_manager.last_keyframe_check_times:
        del state_manager.last_keyframe_check_times[client_uuid]
    
    # 人脸检测时间
    if client_uuid in state_manager.last_face_check_times:
        del state_manager.last_face_check_times[client_uuid]

    # 音频累积器和计数器
    if client_uuid in state_manager.client_audio_accumulators:
        del state_manager.client_audio_accumulators[client_uuid]
    if client_uuid in state_manager.app_audio_save_counter:
        del state_manager.app_audio_save_counter[client_uuid]

    # 性能数据 (带锁)
    if configs.ENABLE_PERFORMANCE_MONITORING:
        perf_lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if perf_lock:
            with perf_lock:
                if client_uuid in state_manager.client_performance_data:
                    del state_manager.client_performance_data[client_uuid]
        else:
            logger.warning(f"清理客户端 {client_uuid} 性能数据时：性能锁未找到。")

    # 事件数据 (带锁)
    event_lock = state_manager.client_data_locks.get(client_uuid, {}).get("event_data_lock")
    if event_lock:
        with event_lock:
            if client_uuid in state_manager.client_event_data:
                del state_manager.client_event_data[client_uuid]
    else:
        logger.warning(f"清理客户端 {client_uuid} 事件数据时：事件锁未找到。")

    # 最后，移除锁本身
    remove_client_locks(client_uuid)
    
    logger.info(f"{client_desc} 资源清理完毕。")