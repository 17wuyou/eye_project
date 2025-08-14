# modules/gui_callbacks.py
import logging
from typing import Dict, Any
from flask import request
from flask_socketio import emit, join_room, leave_room

# 【关键修改】不再从 app 模块导入任何东西

from . import state_manager
from . import android_commands
from .general_utils import get_all_targetable_client_ids

logger = logging.getLogger(__name__)

# 【关键修改】函数接收 socketio 作为参数
def register_gui_callbacks(socketio: Any):
    @socketio.on('connect')
    def sio_connect():
        client_sid = request.sid
        logger.info(f"GUI客户端已连接: SID={client_sid}")
        join_room(state_manager.GUI_LISTENERS_ROOM, sid=client_sid)
        state_manager.socketio_gui_sids.add(client_sid)
        # 向新连接的 GUI 客户端发送当前客户端列表
        emit('update_client_list', get_all_targetable_client_ids(), room=client_sid)
        emit('server_log', {'message': f"GUI客户端 {client_sid} 已连接."}, room=client_sid)

    @socketio.on('disconnect')
    def sio_disconnect():
        client_sid = request.sid
        logger.info(f"GUI客户端已断开连接: SID={client_sid}")
        if client_sid in state_manager.socketio_gui_sids:
            state_manager.socketio_gui_sids.remove(client_sid)
        # Flask-SocketIO 通常会在断开连接时自动处理 leave_room

    @socketio.on('send_command_to_client')
    def sio_send_command_to_android(command_data_from_gui: Dict[str, Any]):
        gui_sid = request.sid
        target_id_str = command_data_from_gui.pop('target_sid', None)

        if not target_id_str:
            emit('server_log', {'message': "错误: 未指定目标客户端ID。"}, room=gui_sid)
            return

        command_type = command_data_from_gui.get('type', '未知命令')

        if target_id_str.startswith("android_"):
            uuid_part = target_id_str.split("android_", 1)[1]
            if android_commands._send_encrypted_command_to_android(uuid_part, command_data_from_gui):
                emit('server_log', {'message': f"命令 '{command_type}' 已发送至 {target_id_str}."}, room=gui_sid)
            else:
                error_msg = f"错误: 发送命令 '{command_type}' 至 {target_id_str} 失败。"
                if uuid_part not in state_manager.raw_ws_clients:
                    error_msg += " 客户端未找到或已断开。"
                emit('server_log', {'message': error_msg}, room=gui_sid)
        else:
            emit('server_log', {'message': f"错误: 无效的目标ID '{target_id_str}' (必须以 'android_' 开头)。"}, room=gui_sid)

# 当 Android 客户端连接/断开时，从 websocket_callbacks 调用此函数
def update_gui_client_dropdowns(socketio: Any):
    socketio.emit('update_client_list', get_all_targetable_client_ids(), room=state_manager.GUI_LISTENERS_ROOM)