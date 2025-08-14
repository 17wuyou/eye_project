# modules/android_commands.py
import json, logging
from typing import Dict, Any
from flask_sock import ConnectionClosed

from . import state_manager
from encryption_util import encrypt_to_string # 假设此文件在项目根目录

logger = logging.getLogger(__name__)

def _send_encrypted_command_to_android(client_uuid_str: str, command_payload: Dict[str, Any]) -> bool:
    target_android_id_for_log = f"android_{client_uuid_str}"
    if client_uuid_str not in state_manager.raw_ws_clients:
        logger.warning(f"发送命令失败: 目标 {target_android_id_for_log} 未找到。")
        return False

    ws_connection = state_manager.raw_ws_clients[client_uuid_str]
    try:
        cmd_json = json.dumps(command_payload)
        enc_cmd = encrypt_to_string(cmd_json) # AES_KEY_STRING 在此函数内部使用
        if not enc_cmd:
            logger.error(f"命令加密失败 for {target_android_id_for_log}. 命令类型: {command_payload.get('type')}.")
            return False
        ws_connection.send(enc_cmd)
        # logger.debug(f"已发送命令 '{command_payload.get('type')}' 到 {target_android_id_for_log}")
        return True
    except ConnectionClosed:
        logger.warning(f"发送命令失败: {target_android_id_for_log} 连接已关闭。")
        # 此处可以考虑从 raw_ws_clients 移除客户端或发出移除信号
        return False
    except Exception as e:
        logger.error(f"发送命令失败: 发送至 {target_android_id_for_log} 时发生错误: {e}", exc_info=False)
        return False