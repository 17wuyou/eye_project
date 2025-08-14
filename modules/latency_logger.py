# modules/latency_logger.py
import logging
import time
import uuid
from typing import Optional, Dict, Any
import configs

# 根据全局配置决定是否启用日志记录
ENABLED = getattr(configs, 'ENABLE_LATENCY_LOGGING', False)
LOG_FILE = getattr(configs, 'LATENCY_LOG_FILE', 'log.txt')

_logger: Optional[logging.Logger] = None

def init_logger():
    """
    初始化延迟日志记录器。
    只有在 ENABLE_LATENCY_LOGGING 为 True 时才执行。
    """
    global _logger
    if not ENABLED or _logger is not None:
        return

    _logger = logging.getLogger('latency_tracker')
    _logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

    # 防止将日志消息传播到根记录器
    _logger.propagate = False

    # 为文件输出创建 handler 和 formatter
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)

    # 将 handler 添加到 logger
    _logger.addHandler(fh)
    _logger.info("延迟日志系统已初始化。")

def log_event(
    trace_id: str,
    client_uuid: str,
    event_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
):
    """
    记录一个延迟事件。

    Args:
        trace_id (str): 用于追踪整个流程的唯一ID。
        client_uuid (str): 客户端的UUID。
        event_name (str): 事件的名称 (例如, 'diarization', 'asr_transcription')。
        status (str): 事件状态 ('START', 'END', 'INFO', 'ERROR')。
        details (dict, optional): 包含额外信息的字典 (例如, duration_ms, reason)。
    """
    if not ENABLED or _logger is None:
        return

    log_message = f"TRACE_ID={trace_id} | CLIENT={client_uuid} | EVENT={event_name} | STATUS={status}"
    
    if details:
        details_str = " | " + " | ".join([f"{k}={v}" for k, v in details.items()])
        log_message += details_str
    
    _logger.debug(log_message)

def generate_trace_id() -> str:
    """生成一个唯一的追踪ID。"""
    return str(uuid.uuid4())