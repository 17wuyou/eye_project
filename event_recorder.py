# modules/event_recorder.py
import os, json, datetime, time, logging
from typing import List, Dict, Any, Optional # 确保 Optional 已导入

from . import state_manager # 用于 client_event_data, client_data_locks
import configs # 用于 DATASET_SAVE_EVENTS, DATASET_ROOT_DIR

logger = logging.getLogger(__name__)

def save_event_data_on_disconnect(client_uuid: str):
    if not configs.DATASET_SAVE_EVENTS:
        return

    event_lock = state_manager.client_data_locks.get(client_uuid, {}).get("event_data_lock")
    if not event_lock:
        logger.error(f"客户端 {client_uuid} 事件数据锁未找到，无法在断开连接时保存最终事件。")
        return

    with event_lock:
        if client_uuid in state_manager.client_event_data:
            event_tracking = state_manager.client_event_data[client_uuid]
            if event_tracking.get("current_event_keyframe_image_bytes") is not None and \
               event_tracking.get("current_event_start_time_unix") is not None and \
               event_tracking.get("current_event_folder_name") is not None:
                
                disconnection_time_unix = time.time()
                logger.info(f"客户端 {client_uuid} 断开连接，保存最终事件 '{event_tracking['current_event_folder_name']}'。")
                _save_event_data_internal(
                    client_uuid=client_uuid,
                    event_folder_name=event_tracking["current_event_folder_name"],
                    event_start_time_unix=event_tracking["current_event_start_time_unix"],
                    event_end_time_unix=disconnection_time_unix,
                    keyframe_image_bytes=event_tracking["current_event_keyframe_image_bytes"],
                    keyframe_timestamp_unix=event_tracking["current_event_keyframe_timestamp_unix"],
                    asr_transcripts=list(event_tracking["accumulated_asr_for_event"]) # 创建副本
                )
            # else:
                # logger.debug(f"客户端 {client_uuid} 断开连接时无当前事件数据可保存。")

def _save_event_data_internal(client_uuid: str, event_folder_name: str,
                              event_start_time_unix: float, event_end_time_unix: float,
                              keyframe_image_bytes: bytes, keyframe_timestamp_unix: float,
                              asr_transcripts: List[Dict[str, Any]]): # 更具体的类型提示
    try:
        client_specific_path = os.path.join(configs.DATASET_ROOT_DIR, f"android_{client_uuid}")
        os.makedirs(client_specific_path, exist_ok=True)
        
        event_path = os.path.join(client_specific_path, event_folder_name)
        os.makedirs(event_path, exist_ok=True)

        keyframe_image_filename = "keyframe.jpg"
        keyframe_image_filepath = os.path.join(event_path, keyframe_image_filename)
        with open(keyframe_image_filepath, "wb") as f:
            f.write(keyframe_image_bytes)

        asr_transcripts_filename = "asr_transcripts.json"
        asr_transcripts_filepath = os.path.join(event_path, asr_transcripts_filename)
        with open(asr_transcripts_filepath, "w", encoding="utf-8") as f:
            json.dump(asr_transcripts, f, ensure_ascii=False, indent=2)

        metadata_filename = "metadata.json"
        metadata_filepath = os.path.join(event_path, metadata_filename)

        def safe_isoformat(unix_ts: Optional[float]) -> Optional[str]:
            if unix_ts is None: return None
            try:
                return datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc).isoformat()
            except: # 时间戳可能存在问题的回退方案
                return str(unix_ts)

        metadata = {
            "event_folder_id": event_folder_name,
            "client_uuid": client_uuid,
            "keyframe_timestamp_unix": keyframe_timestamp_unix,
            "keyframe_timestamp_iso": safe_isoformat(keyframe_timestamp_unix),
            "event_start_time_unix": event_start_time_unix,
            "event_start_time_iso": safe_isoformat(event_start_time_unix),
            "event_end_time_unix": event_end_time_unix,
            "event_end_time_iso": safe_isoformat(event_end_time_unix),
            "event_duration_seconds": round(event_end_time_unix - event_start_time_unix, 3) 
                                      if event_start_time_unix and event_end_time_unix else None,
            "keyframe_image_path_relative": keyframe_image_filename,
            "asr_transcripts_path_relative": asr_transcripts_filename,
            "num_asr_segments": len(asr_transcripts)
        }
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"事件数据已保存至: {event_path}")

    except Exception as e:
        logger.error(f"保存事件数据失败 (客户端 {client_uuid}, 事件文件夹 {event_folder_name}): {e}", exc_info=True)


def record_asr_for_event(client_uuid: str, final_transcripts: List[str]):
    if not configs.DATASET_SAVE_EVENTS:
        return

    event_lock = state_manager.client_data_locks.get(client_uuid, {}).get("event_data_lock")
    if not event_lock:
        logger.error(f"客户端 {client_uuid} 事件数据锁未找到，无法记录ASR事件。")
        return
        
    with event_lock:
        if client_uuid in state_manager.client_event_data:
            event_tracking = state_manager.client_event_data[client_uuid]
            if event_tracking and event_tracking.get("current_event_start_time_unix") is not None:
                asr_event_timestamp_iso = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
                for line in final_transcripts:
                    parts = line.split(":", 1)
                    speaker_tag = parts[0].strip()
                    text_content = parts[1].strip() if len(parts) > 1 else ""
                    if text_content: # 仅当有实际文本内容时才添加
                        event_tracking["accumulated_asr_for_event"].append({
                            "timestamp_iso": asr_event_timestamp_iso,
                            "speaker": speaker_tag,
                            "text": text_content
                        })
            # else:
                # logger.debug(f"客户端 {client_uuid}: 未开始事件或事件跟踪数据不存在，不记录ASR。")

# 当 video_processing_logic 检测到关键帧时将调用此函数
def manage_event_on_keyframe(client_uuid: str, client_desc: str, video_frame_bytes: bytes, keyframe_timestamp_unix: float):
    if not configs.DATASET_SAVE_EVENTS:
        logger.debug(f"[{client_desc}] 事件数据集保存功能未启用，跳过关键帧事件处理。")
        return

    event_lock = state_manager.client_data_locks.get(client_uuid, {}).get("event_data_lock")
    if not event_lock:
        logger.error(f"[{client_desc}] 关键帧事件处理: 事件数据锁未找到。")
        return

    prev_event_to_save_details: Optional[Dict[str, Any]] = None
    current_event_folder_name_for_log: Optional[str] = None

    with event_lock:
        logger.debug(f"[{client_desc}] 获取 event_data_lock 以处理关键帧事件数据...")
        if client_uuid not in state_manager.client_event_data:
            logger.warning(f"[{client_desc}] 关键帧事件处理: 客户端事件数据未找到，无法处理。")
            return

        event_tracking = state_manager.client_event_data[client_uuid]

        # A. 准备保存上一个事件的数据 (如果存在)
        if event_tracking.get("current_event_keyframe_image_bytes") is not None and \
           event_tracking.get("current_event_start_time_unix") is not None:
            
            prev_event_folder_name = event_tracking.get("current_event_folder_name")
            if not prev_event_folder_name: # 回退
                 prev_kf_ts_for_name = event_tracking.get('current_event_keyframe_timestamp_unix', time.time())
                 prev_event_folder_name = (
                    f"event_{event_tracking.get('event_counter', 0):05d}_"
                    f"{datetime.datetime.fromtimestamp(prev_kf_ts_for_name).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                )
            
            prev_event_to_save_details = {
                "client_uuid": client_uuid,
                "event_folder_name": prev_event_folder_name,
                "event_start_time_unix": event_tracking["current_event_start_time_unix"],
                "event_end_time_unix": keyframe_timestamp_unix, # 当前关键帧结束上一个事件
                "keyframe_image_bytes": event_tracking["current_event_keyframe_image_bytes"],
                "keyframe_timestamp_unix": event_tracking["current_event_keyframe_timestamp_unix"],
                "asr_transcripts": list(event_tracking["accumulated_asr_for_event"]) # 复制
            }
            event_tracking["accumulated_asr_for_event"].clear() # 为新事件清空
            logger.debug(f"[{client_desc}] 事件 '{prev_event_folder_name}' 数据准备完毕，锁外保存。")

        # B. 为新事件更新 event_tracking
        event_tracking["event_counter"] = event_tracking.get("event_counter", 0) + 1
        new_event_folder_name = (
            f"event_{event_tracking['event_counter']:05d}_"
            f"{datetime.datetime.fromtimestamp(keyframe_timestamp_unix).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        )
        event_tracking["current_event_folder_name"] = new_event_folder_name
        event_tracking["current_event_start_time_unix"] = keyframe_timestamp_unix
        event_tracking["current_event_keyframe_image_bytes"] = video_frame_bytes
        event_tracking["current_event_keyframe_timestamp_unix"] = keyframe_timestamp_unix
        
        current_event_folder_name_for_log = new_event_folder_name
        logger.debug(f"[{client_desc}] 新事件 '{new_event_folder_name}' 跟踪信息已在锁内更新。")

    # --- 结束临界区 ---
    logger.debug(f"[{client_desc}] 释放 event_data_lock。")

    # C. 在锁外部执行文件保存操作
    if prev_event_to_save_details:
        logger.info(f"[{client_desc}] (锁外) 保存上一个事件数据: '{prev_event_to_save_details['event_folder_name']}'...")
        try:
            _save_event_data_internal(
                client_uuid=prev_event_to_save_details["client_uuid"],
                event_folder_name=prev_event_to_save_details["event_folder_name"],
                event_start_time_unix=prev_event_to_save_details["event_start_time_unix"],
                event_end_time_unix=prev_event_to_save_details["event_end_time_unix"],
                keyframe_image_bytes=prev_event_to_save_details["keyframe_image_bytes"],
                keyframe_timestamp_unix=prev_event_to_save_details["keyframe_timestamp_unix"],
                asr_transcripts=prev_event_to_save_details["asr_transcripts"]
            )
            logger.info(f"[{client_desc}] (锁外) 事件数据 '{prev_event_to_save_details['event_folder_name']}' 保存完毕。")
        except Exception as e_save:
            logger.error(f"[{client_desc}] (锁外) 保存事件 '{prev_event_to_save_details['event_folder_name']}' 失败: {e_save}", exc_info=True)

    if current_event_folder_name_for_log:
        logger.info(f"[{client_desc}] 新事件 '{current_event_folder_name_for_log}' 已正式开始。")