# modules/video_processing_logic.py
import time, logging
from typing import Dict, List, Any, Optional

from . import state_manager
from . import performance_utils
from . import android_commands
from . import event_recorder
from . import service_management
# 新增: 导入延迟日志模块
from . import latency_logger

import configs

logger = logging.getLogger(__name__)

# --- 关键帧处理 ---
# 修改: 增加 trace_id 参数
def _run_keyframe_detection_task(socketio: Any, detector_instance: Any, frame_bytes: bytes, client_uuid: str, client_desc: str, trace_id: str):
    """在线程池中运行关键帧检测并处理结果。"""
    latency_logger.log_event(trace_id, client_uuid, 'keyframe_detection_task', 'START')
    kf_detect_start_time = time.perf_counter()
    
    is_kf = detector_instance.is_keyframe(frame_bytes)
    
    kf_detect_duration_ms = (time.perf_counter() - kf_detect_start_time) * 1000
    latency_logger.log_event(
        trace_id, client_uuid, 'keyframe_detection_task', 'END', 
        {'duration_ms': f'{kf_detect_duration_ms:.2f}', 'is_keyframe': is_kf}
    )

    if configs.ENABLE_PERFORMANCE_MONITORING:
        perf_lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if perf_lock:
            with perf_lock:
                if client_uuid in state_manager.client_performance_data:
                    perf_entry = state_manager.client_performance_data[client_uuid]
                    perf_entry["last_kf_detection_duration_ms"] = kf_detect_duration_ms
                    perf_entry["cycle_kf_detection_time_ms_total"] = \
                        perf_entry.get("cycle_kf_detection_time_ms_total", 0.0) + kf_detect_duration_ms
                    perf_entry["cycle_kf_detection_count"] = \
                        perf_entry.get("cycle_kf_detection_count", 0) + 1
        else:
            logger.warning(f"[{client_desc}] 关键帧检测任务: 无法获取性能锁来更新耗时。")

    if is_kf:
        logger.info(f"[{client_desc}] (异步)检测到关键帧! 耗时: {kf_detect_duration_ms:.2f}ms. 准备后续处理。")
        current_keyframe_time_unix = time.time()
        socketio.start_background_task(
            _handle_detected_keyframe_results,
            client_uuid,
            client_desc,
            frame_bytes,
            current_keyframe_time_unix
        )

def _handle_detected_keyframe_results(client_uuid: str, client_desc: str, video_frame_bytes: bytes, keyframe_timestamp_unix: float):
    """
    处理关键帧检测结果：发送命令、管理事件。
    """
    logger.debug(f"[{client_desc}] 开始处理已检测到的关键帧 (时间戳: {keyframe_timestamp_unix})...")

    kf_cmd = {
        "type": "DRAW_TEXT", 
        "text_id": f"kf_{int(keyframe_timestamp_unix*1000)}",
        "text": "关键帧", 
        "position_normalized": [0.5, 0.15],
        "color_rgba": [0,0,255,255], 
        "size_sp": 30, 
        "duration_ms": 1000
    }
    if not android_commands._send_encrypted_command_to_android(client_uuid, kf_cmd):
        logger.warning(f"[{client_desc}] 发送关键帧绘制命令失败。")

    event_recorder.manage_event_on_keyframe(
        client_uuid, 
        client_desc, 
        video_frame_bytes, 
        keyframe_timestamp_unix
    )
    
    logger.debug(f"[{client_desc}] 关键帧处理完毕。")


# --- 人脸处理 ---
# 修改: 增加 trace_id 参数
def _run_face_processing_task(socketio: Any, client_uuid: str, client_desc: str, frame_bytes: bytes, trace_id: str):
    """在线程池中运行人脸检测/识别。"""
    if not service_management.face_service_instance:
        logger.warning(f"[{client_desc}] 人脸服务未初始化，跳过人脸处理。")
        return

    latency_logger.log_event(trace_id, client_uuid, 'face_processing_task', 'START')
    face_proc_start_time = time.perf_counter()
    
    face_results: List[Dict[str, Any]] = []
    try:
        face_results = service_management.face_service_instance.process_frame(frame_bytes)
    except Exception as e_face_proc:
        logger.error(f"[{client_desc}] 人脸处理任务执行中发生错误: {e_face_proc}", exc_info=True)
        
    face_proc_duration_ms = (time.perf_counter() - face_proc_start_time) * 1000
    latency_logger.log_event(
        trace_id, client_uuid, 'face_processing_task', 'END', 
        {'duration_ms': f'{face_proc_duration_ms:.2f}', 'faces_found': len(face_results)}
    )
    logger.info(f"[{client_desc}] 人脸处理 (InsightFace) 完成，耗时: {face_proc_duration_ms:.2f}ms. 检测到 {len(face_results)} 张人脸。")

    socketio.start_background_task(
        _handle_face_processing_results,
        client_uuid,
        client_desc,
        face_results
    )

def _handle_face_processing_results(client_uuid: str, client_desc: str, results: List[Dict[str, Any]]):
    """
    处理人脸识别结果并发送 DRAW_TEXT 命令。
    【已更新】此函数现在只处理人脸名称。
    """
    num_faces_to_display = min(len(results), configs.ANDROID_FACE_INFO_MAX_FACES)

    for i in range(num_faces_to_display):
        res = results[i]
        name = res.get('name', '未知')
        
        # 新的、简化的显示文本
        text_to_display = f"身份: {name}"
        text_id = f"face_info_{client_uuid}_{i}"
        
        # 使用配置中的位置和样式参数
        position_y = configs.ANDROID_FACE_INFO_BASE_Y_NORMALIZED + \
                     (i * configs.ANDROID_FACE_INFO_LINE_HEIGHT_NORMALIZED)
        # 将文本放在左侧，方便阅读
        position_x = 0.1 

        draw_cmd = {
            "type": "DRAW_TEXT", "text_id": text_id, "text": text_to_display,
            "position_normalized": [position_x, position_y],
            "color_rgba": [255, 255, 0, 255], # 颜色保持黄色以醒目
            "size_sp": configs.ANDROID_FACE_INFO_TEXT_SIZE_SP,
            "duration_ms": configs.ANDROID_FACE_INFO_DURATION_MS
        }
        if android_commands._send_encrypted_command_to_android(client_uuid, draw_cmd):
             logger.debug(f"[{client_desc}] 已发送人脸信息: '{text_to_display}'")

    # 清除可能残留的旧的人脸信息文本
    for i in range(num_faces_to_display, configs.ANDROID_FACE_INFO_MAX_FACES):
        text_id = f"face_info_{client_uuid}_{i}"
        clear_cmd = {
            "type": "DRAW_TEXT", "text_id": text_id, "text": "", 
            "position_normalized": [0,0], "color_rgba": [0,0,0,0],
            "size_sp": 1, "duration_ms": 1 # 发送空文本，立即过期
        }
        android_commands._send_encrypted_command_to_android(client_uuid, clear_cmd)