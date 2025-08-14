# modules/performance_utils.py
import time, datetime, logging, threading
from typing import Optional, Dict, Any

from . import state_manager
from .general_utils import parse_iso_timestamp_to_unix
import configs
from werkzeug.serving import is_running_from_reloader
logger = logging.getLogger(__name__)

# 【关键修改】移除 'from app import app_instance_for_perf'
# 我们将完全依赖 set_app_instance 函数来填充这个变量
app_instance_for_perf: Optional[Any] = None

def set_app_instance(app: Any):
    global app_instance_for_perf
    app_instance_for_perf = app

def initialize_performance_data(client_uuid: str):
    if configs.ENABLE_PERFORMANCE_MONITORING:
        lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if not lock:
            logger.warning(f"性能监控: 客户端 {client_uuid} 的性能锁未找到。")
            state_manager.client_data_locks.setdefault(client_uuid, {})['performance_lock'] = threading.Lock()
            lock = state_manager.client_data_locks[client_uuid]['performance_lock']

        with lock:
            state_manager.client_performance_data[client_uuid] = {
                "cycle_client_timestamp_unix": None,
                "cycle_server_recv_wall_time": None,
                "cycle_server_recv_perf_time": None,
                "cycle_accumulated_vad_duration_ms": 0.0,
                "cycle_chunk_count": 0,
                "cycle_kf_detection_time_ms_total": 0.0,
                "cycle_kf_detection_count": 0,
                "last_kf_detection_duration_ms": 0.0,
                "last_processed_client_timestamp_unix": None,
                "last_processed_server_recv_wall_time": None,
                "last_diarization_duration_ms": 0.0,
                "last_asr_total_duration_ms": 0.0,
                "last_accumulated_vad_duration_ms": 0.0,
                "last_total_processing_wall_time_ms": 0.0,
                "latency_client_to_server_recv_ms": None,
                "latency_client_to_server_processed_ms": None,
                "last_report_time": time.monotonic()
            }

def reset_cycle_performance_metrics(client_uuid: str):
    if configs.ENABLE_PERFORMANCE_MONITORING and client_uuid in state_manager.client_performance_data:
        lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if lock:
            with lock:
                if client_uuid in state_manager.client_performance_data:
                    entry = state_manager.client_performance_data[client_uuid]
                    entry["cycle_client_timestamp_unix"] = None
                    entry["cycle_server_recv_wall_time"] = None
                    entry["cycle_server_recv_perf_time"] = None
                    entry["cycle_accumulated_vad_duration_ms"] = 0.0
                    entry["cycle_chunk_count"] = 0
                    entry["cycle_kf_detection_time_ms_total"] = 0.0
                    entry["cycle_kf_detection_count"] = 0
        else:
            logger.warning(f"性能监控: 重置周期指标时客户端 {client_uuid} 的性能锁未找到。")


def log_detailed_performance_for_cycle(client_uuid: str, diar_time_ms: float, asr_time_ms: float, vad_time_ms: float,
                                        total_wall_processing_time_ms: float, client_ts: Optional[float], server_recv_ts: Optional[float],
                                        kf_avg_time_ms: Optional[float]):
    if configs.LOG_DETAILED_COMPONENT_TIMES and configs.ENABLE_PERFORMANCE_MONITORING:
        log_msg = f"性能监控 [{client_uuid}] - 处理周期完成: "
        log_msg += f"VAD总耗时: {vad_time_ms:.2f}ms, "
        log_msg += f"Diarization耗时: {diar_time_ms:.2f}ms, "
        log_msg += f"ASR总耗时: {asr_time_ms:.2f}ms, "
        if kf_avg_time_ms is not None:
            log_msg += f"KF检测平均耗时: {kf_avg_time_ms:.2f}ms, "
        log_msg += f"服务器总处理墙上时间: {total_wall_processing_time_ms:.2f}ms."
        if client_ts and server_recv_ts:
            recv_latency = (server_recv_ts - client_ts) * 1000
            client_ts_dt = "N/A"
            try:
                client_ts_dt = datetime.datetime.fromtimestamp(client_ts).isoformat()
            except:
                pass
            log_msg += f" (客户端发送时间: {client_ts_dt}, 服务器接收延迟: {recv_latency:.2f}ms)"
        logger.info(log_msg)

def _report_performance_summary(socketio: Any):
    if not configs.ENABLE_PERFORMANCE_MONITORING: return
    
    active_clients_uuids = list(state_manager.raw_ws_clients.keys())

    for client_uuid in active_clients_uuids:
        if client_uuid not in state_manager.client_performance_data:
            continue

        lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if not lock:
            logger.warning(f"性能报告: 客户端 {client_uuid} 的性能锁未找到，跳过报告。")
            continue

        with lock:
            if client_uuid not in state_manager.client_performance_data:
                continue
            
            data = state_manager.client_performance_data[client_uuid]
            summary_parts = [f"性能报告 [{client_uuid}]"]
            has_data_to_report = False

            if data.get("last_processed_client_timestamp_unix") is not None or data.get("cycle_kf_detection_count", 0) > 0:
                has_data_to_report = True
                
                client_send_dt_iso = "N/A"
                if data.get("last_processed_client_timestamp_unix") is not None:
                    try:
                        client_send_dt = datetime.datetime.fromtimestamp(data["last_processed_client_timestamp_unix"])
                        client_send_dt_iso = client_send_dt.isoformat()
                    except Exception: pass
                summary_parts.append(f"最新处理音频客户端时间戳: {client_send_dt_iso}")

                if data.get("latency_client_to_server_recv_ms") is not None:
                    summary_parts.append(f"C->S接收延迟: {data['latency_client_to_server_recv_ms']:.2f}ms")
                if data.get("latency_client_to_server_processed_ms") is not None:
                    summary_parts.append(f"C->S处理完成延迟: {data['latency_client_to_server_processed_ms']:.2f}ms")
                
                summary_parts.append(f"VAD耗时: {data.get('last_accumulated_vad_duration_ms', 0):.2f}ms")
                summary_parts.append(f"Diarization耗时: {data.get('last_diarization_duration_ms', 0):.2f}ms")
                summary_parts.append(f"ASR总耗时: {data.get('last_asr_total_duration_ms', 0):.2f}ms")
                
                if data.get("cycle_kf_detection_count", 0) > 0:
                    avg_kf_time = data["cycle_kf_detection_time_ms_total"] / data["cycle_kf_detection_count"]
                    summary_parts.append(f"KF检测平均耗时(周期): {avg_kf_time:.2f}ms ({data['cycle_kf_detection_count']}次)")
                elif data.get("last_kf_detection_duration_ms", 0) > 0 :
                     summary_parts.append(f"KF检测耗时(上次): {data['last_kf_detection_duration_ms']:.2f}ms")

                summary_parts.append(f"服务器总处理时间(音频): {data.get('last_total_processing_wall_time_ms', 0):.2f}ms")
            
            if has_data_to_report:
                logger.info(" | ".join(summary_parts))
            
            data["last_report_time"] = time.monotonic()

def performance_reporting_task(socketio: Any):
    global app_instance_for_perf
    if not configs.ENABLE_PERFORMANCE_MONITORING or not app_instance_for_perf or not is_running_from_reloader():
        return
    
    _report_performance_summary(socketio)
    
    if configs.ENABLE_PERFORMANCE_MONITORING and any(state_manager.raw_ws_clients):
         state_manager._performance_reporting_timer = threading.Timer(
             configs.PERFORMANCE_REPORT_INTERVAL_SECONDS, 
             performance_reporting_task,
             args=[socketio] # 传递socketio
         )
         state_manager._performance_reporting_timer.daemon = True
         state_manager._performance_reporting_timer.start()
    else:
        state_manager._performance_reporting_timer = None
        logger.info("性能报告任务：无活动客户端或监控已禁用，暂停调度。")

def start_performance_reporting_if_needed(socketio: Any):
    global app_instance_for_perf
    if configs.ENABLE_PERFORMANCE_MONITORING and \
       state_manager._performance_reporting_timer is None and \
       any(state_manager.raw_ws_clients) and \
       app_instance_for_perf and app_instance_for_perf.is_running_from_reloader():
        logger.info("性能报告任务：启动...")
        performance_reporting_task(socketio)

def stop_performance_reporting():
    if state_manager._performance_reporting_timer:
        logger.info("性能报告任务：正在取消...")
        state_manager._performance_reporting_timer.cancel()
        state_manager._performance_reporting_timer = None