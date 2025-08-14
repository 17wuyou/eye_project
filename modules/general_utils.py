# modules/general_utils.py
import os, datetime, logging
from typing import List, Optional, Union # 确保 Union, Optional 已导入

from . import state_manager
import configs

logger = logging.getLogger(__name__)


def _is_text_no_speech(text: str) -> bool:
    if not text: return True
    text_lower_stripped = text.strip().lower()
    if not text_lower_stripped: return True
    for indicator in configs.ASR_NO_SPEECH_INDICATORS:
        if indicator.lower() == text_lower_stripped:
            return True
    return False

def _is_text_blocked(text: str) -> bool:
    if not text or not configs.SUBTITLE_BLOCKED_PHRASES: return False
    text_lower = text.lower()
    for phrase in configs.SUBTITLE_BLOCKED_PHRASES:
        if phrase.lower() in text_lower:
            return True
    return False

def get_all_targetable_client_ids() -> List[str]:
    # 从 state_manager 访问 raw_ws_clients
    return [f"android_{uuid_str}" for uuid_str in state_manager.raw_ws_clients.keys()]

def _save_app_level_audio(client_uuid: str, audio_bytes: bytes, stage_suffix: str):
    # 从 state_manager 访问和修改 app_audio_save_counter
    if client_uuid not in state_manager.app_audio_save_counter:
        state_manager.app_audio_save_counter[client_uuid] = 0
    state_manager.app_audio_save_counter[client_uuid] += 1

    if configs.APP_SAVE_RAW_AUDIO_CHUNKS and \
       state_manager.app_audio_save_counter[client_uuid] % configs.APP_SAVE_RAW_AUDIO_INTERVAL == 0:
        try:
            d = "debug_audio_app_level"
            os.makedirs(d, exist_ok=True)
            counter_val = state_manager.app_audio_save_counter[client_uuid] // configs.APP_SAVE_RAW_AUDIO_INTERVAL
            fn = f"{configs.APP_AUDIO_SAVE_PATH_PREFIX}_{client_uuid}_{stage_suffix}_{counter_val}.pcm"
            sp = os.path.join(d, fn)
            with open(sp, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            logger.error(f"APP级别音频保存: 保存 {stage_suffix} (客户端 {client_uuid}) 时发生错误: {e}")

def parse_iso_timestamp_to_unix(ts_str: Optional[Union[str, int, float]]) -> Optional[float]:
    if ts_str is None:
        return None
    if not isinstance(ts_str, str):
        try:
            num_ts = float(ts_str)
            if num_ts > 410244480000:  # 可能是毫秒
                return num_ts / 1000.0
            elif num_ts > 946684800: # 可能是秒 (Unix 纪元开始后)
                 return num_ts
            else: # 可能是秒或其他较小的值，视为秒
                logger.warning(f"时间戳解析：数字时间戳 '{ts_str}' 的量级不确定，按原样返回（可能代表秒）。")
                return num_ts
        except ValueError:
            logger.warning(f"时间戳解析：无法将非字符串时间戳 '{ts_str}' (类型: {type(ts_str)}) 转换为数字。")
            return None

    original_ts_str_for_log = ts_str
    try:
        # 尝试 ISO 8601 格式 (常见于 JavaScript Date.toISOString())
        if ts_str.endswith("Z"): # UTC
            # Python 的 fromisoformat 在 3.11 版本前不直接喜欢时区前的 'Z'
            dt_obj = datetime.datetime.fromisoformat(ts_str[:-1] + "+00:00")
        elif 'T' in ts_str and (':' in ts_str or '-' in ts_str): # 更通用的带 T 分隔符的类 ISO 格式
            dt_obj = datetime.datetime.fromisoformat(ts_str)
        else:
            # 如果它看起来不像典型的带 T 的 ISO 字符串，则引发 ValueError 以尝试数字解析
            raise ValueError("根据简单检查，不是清晰的 ISO 8601 格式。")
        return dt_obj.timestamp()
    except ValueError:
        pass # 不是有效的 ISO 字符串，接下来将尝试数字解析
    except Exception as e_iso: # 捕获 ISO 解析期间的任何其他意外错误
        logger.error(f"时间戳解析：尝试ISO解析时间戳 '{original_ts_str_for_log}' 时发生未知错误: {e_iso}")
        pass # 继续进行数字解析

    # 尝试将字符串解析为 Unix 时间戳 (秒或毫秒)
    # 检查字符串是否为数字 (整数或浮点数)
    if ts_str.isdigit() or (ts_str.replace('.', '', 1).isdigit() and ts_str.count('.') <= 1):
        try:
            num_ts = float(ts_str)
            # 启发式：如果数字非常大，则假定为毫秒
            # 2100-01-01 00:00:00 UTC 的 Unix 时间戳是 4102444800
            # 2100-01-01 00:00:00 UTC 的 Unix 时间戳 (毫秒) 是 4102444800000
            if num_ts > 4102444800000: # 任意大数检查，如果需要可以调整
                 logger.warning(f"时间戳解析：数字字符串时间戳 '{original_ts_str_for_log}' 过大，可能不是有效的毫秒或秒时间戳。")
                 return None # 或适当处理为错误
            if num_ts > 4102444800: # 如果大于 2100 年的秒数，则可能是毫秒
                logger.debug(f"时间戳解析：将数字字符串时间戳 '{original_ts_str_for_log}' 解析为Unix毫秒时间戳。")
                return num_ts / 1000.0
            else: # 否则，假定为秒
                logger.debug(f"时间戳解析：将数字字符串时间戳 '{original_ts_str_for_log}' 解析为Unix秒时间戳。")
                return num_ts
        except ValueError:
            # 如果 isdigit/isnumeric 检查通过，理想情况下不应发生这种情况，但作为安全措施
            logger.warning(f"时间戳解析：无法将纯数字时间戳字符串 '{original_ts_str_for_log}' 转换为数字。")
            return None
            
    logger.warning(f"时间戳解析：无法将时间戳字符串 '{original_ts_str_for_log}' 解析为任何已知格式 (ISO, Unix ms, Unix s)。")
    return None