# modules/management_callbacks.py
import logging
import os
import re
from flask import request
from flask_socketio import emit

from . import service_management
import configs

logger = logging.getLogger(__name__)

# --- 文件名解析工具函数 ---

def parse_speaker_filename(filename_without_ext: str):
    """
    解析说话人音频文件名。
    格式: personXX-CustomName 或 personXX
    返回: (原始ID, 显示名称)
    """
    match = re.fullmatch(r"(person\d{2,})(-(.+))?", filename_without_ext)
    if match:
        original_id = match.group(1)
        custom_name = match.group(3)
        is_default = not custom_name
        display_name = custom_name if custom_name else original_id
        return original_id, display_name, is_default
    return filename_without_ext, filename_without_ext, False

def parse_face_filename(filename_without_ext: str):
    """
    解析人脸图像文件名。
    格式: person_{id_num:03d}_{name}
    返回: (原始ID, 显示名称, 是否为默认名)
    """
    parts = filename_without_ext.split('_')
    if len(parts) >= 3 and parts[0] == 'person' and parts[1].isdigit():
        original_id = f"{parts[0]}_{parts[1]}"
        custom_name = "_".join(parts[2:])
        is_default = custom_name.startswith("UnKnown") and custom_name[7:].isdigit()
        return original_id, custom_name, is_default
    return filename_without_ext, filename_without_ext, True

def sanitize_filename_part(name: str) -> str:
    """清理文件名中的非法字符，但不包括路径分隔符。"""
    return re.sub(r'[\\/*?"<>|]', "", name).strip()

# --- Socket.IO 事件回调注册 ---

def register_management_callbacks(socketio):
    """注册管理面板的Socket.IO事件处理程序。"""

    @socketio.on('get_media_files')
    def on_get_media_files():
        """处理来自客户端的获取媒体文件列表的请求。"""
        client_sid = request.sid
        logger.info(f"管理客户端 {client_sid} 请求媒体文件列表。")
        
        media_data = {
            'speakers': {'named': [], 'unnamed': []},
            'faces': {'named': [], 'unnamed': []}
        }

        # 1. 处理说话人音频 (MP3)
        speaker_dir = configs.SPEAKER_AUDIO_SAMPLES_PATH
        if os.path.exists(speaker_dir):
            for filename in os.listdir(speaker_dir):
                if filename.lower().endswith(".mp3"):
                    base_name = os.path.splitext(filename)[0]
                    original_id, display_name, is_default = parse_speaker_filename(base_name)
                    category = 'unnamed' if is_default else 'named'
                    media_data['speakers'][category].append({
                        'id': original_id,
                        'display_name': display_name,
                        'filename': filename,
                        'url_path': f'db_files/speakers/{filename}'
                    })

        # 2. 处理人脸图像 (JPG)
        face_image_dir = os.path.join(configs.FACE_DB_PATH, "images")
        if os.path.exists(face_image_dir):
            for filename in os.listdir(face_image_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    base_name = os.path.splitext(filename)[0]
                    original_id, display_name, is_default = parse_face_filename(base_name)
                    category = 'unnamed' if is_default else 'named'
                    media_data['faces'][category].append({
                        'id': original_id,
                        'display_name': display_name,
                        'filename': filename,
                        'url_path': f'db_files/faces/{filename}'
                    })
        
        # 按显示名称排序
        for cat in ['speakers', 'faces']:
            media_data[cat]['named'].sort(key=lambda x: x['display_name'])
            media_data[cat]['unnamed'].sort(key=lambda x: x['display_name'])

        emit('update_media_files', media_data, room=client_sid)

    @socketio.on('rename_media_file')
    def on_rename_media_file(data):
        """处理来自客户端的文件重命名请求。"""
        client_sid = request.sid
        try:
            file_type = data['file_type']
            original_filename = data['original_filename']
            new_display_name = sanitize_filename_part(data['new_name'])

            if not new_display_name:
                raise ValueError("新名称不能为空。")

            base_dir, id_part, old_base_name, extension = (None, None, None, None)

            if file_type == 'speaker':
                base_dir = configs.SPEAKER_AUDIO_SAMPLES_PATH
                old_base_name, extension = os.path.splitext(original_filename)
                id_part, _, _ = parse_speaker_filename(old_base_name)
                new_filename = f"{id_part}-{new_display_name}{extension}"
            elif file_type == 'face':
                base_dir = os.path.join(configs.FACE_DB_PATH, "images")
                old_base_name, extension = os.path.splitext(original_filename)
                id_part, _, _ = parse_face_filename(old_base_name)
                new_filename = f"{id_part}_{new_display_name}{extension}"
            else:
                raise ValueError(f"未知的文件类型: {file_type}")

            old_path = os.path.join(base_dir, original_filename)
            new_path = os.path.join(base_dir, new_filename)

            if not os.path.exists(old_path):
                raise FileNotFoundError(f"原始文件不存在: {old_path}")
            
            if old_path == new_path:
                 emit('rename_status', {'success': True, 'message': '新旧名称相同，无需更改。'}, room=client_sid)
                 return

            logger.info(f"请求重命名文件: 从 '{old_path}' 到 '{new_path}'")
            os.rename(old_path, new_path)
            
            # 关键步骤：触发服务重新加载数据库
            service_management.reload_services_databases()
            
            emit('rename_status', {'success': True, 'message': f"成功将 {original_filename} 重命名为 {new_filename}"}, room=client_sid)
            # 重新发送更新后的文件列表
            on_get_media_files()

        except Exception as e:
            logger.error(f"重命名文件时出错: {e}", exc_info=True)
            emit('rename_status', {'success': False, 'message': f"错误: {e}"}, room=client_sid)