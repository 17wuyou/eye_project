# app.py (修正版)
# 这个版本将Python应用作为C++核心引擎的宿主，通过pybind11桥接直接调用Python函数，
# 而不是通过WebSocket进行通信。这符合.pyd模块的设计意图。

import logging
import os
import sys
import atexit
import threading
import asyncio
from typing import Optional, List, Dict, Any

from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO
from werkzeug.serving import is_running_from_reloader

# --- 路径和配置初始化 ---
# 确保当前目录在Python路径中，以便可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 关键：导入C++桥接模块 ---
# 这会加载 my_project_bridge.cp312-win_amd64.pyd 文件
try:
    import my_project_bridge
except ImportError as e:
    print("="*80)
    print("FATAL ERROR: 无法导入 C++ 桥接模块 'my_project_bridge'。")
    print(f"错误信息: {e}")
    print("请确保:")
    print("1. 'my_project_bridge.cp312-win_amd64.pyd' 文件与 app.py 在同一目录下。")
    print("2. 您正在使用 Python 3.12 (64-bit) 解释器。")
    print("3. 所有 C++ 依赖项 (如 uWebSockets, OpenSSL, OpenCV 的 DLL) 都在系统 PATH 中或与 .pyd 文件在同一目录。")
    print("="*80)
    sys.exit(1)


import configs
from modules import (
    latency_logger,
    service_management,
    state_manager,
    llm_service,
    gui_callbacks,
    management_callbacks
)
from encryption_util import decrypt_from_string

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 目录和环境变量设置 ---
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)
os.makedirs(configs.MODEL_CACHE_DIR, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = configs.MODEL_CACHE_DIR
os.environ['HF_HOME'] = configs.MODEL_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = configs.MODEL_CACHE_DIR

# --- Flask 应用和扩展初始化 ---
app = Flask(__name__, static_folder=static_dir)
app.config['SECRET_KEY'] = configs.AES_KEY_STRING
# 异步模式很重要，因为它允许后台任务（如LLM/TTS）与SocketIO共存
socketio = SocketIO(app, async_mode='threading')


# ==============================================================================
# --- C++ 到 Python 的桥接函数 ---
# 这些是暴露给C++ CoreEngine调用的Python入口点。
# C++中的 AIServiceManager 将会调用这些函数。
# ==============================================================================

def bridge_transcribe(audio_bytes: bytes, sample_rate: int) -> str:
    """供C++调用的ASR转录函数"""
    if service_management.asr_service_instance:
        try:
            # 注意：C++传递过来的bytes可能需要确认格式
            transcript = service_management.asr_service_instance.transcribe_bytes(audio_bytes, sample_rate)
            logger.info(f"[Bridge] ASR Result: {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"[Bridge] ASR转录时出错: {e}", exc_info=True)
            return ""
    logger.warning("[Bridge] ASR服务未初始化，无法转录。")
    return ""

def bridge_recognize_faces(image_bytes: bytes) -> List[Dict[str, Any]]:
    """供C++调用的面部识别函数"""
    if service_management.face_service_instance:
        try:
            # C++传递的是原始图像字节
            results = service_management.face_service_instance.process_frame(image_bytes)
            logger.info(f"[Bridge] Face Recognition Result: {results}")
            return results
        except Exception as e:
            logger.error(f"[Bridge] 人脸识别时出错: {e}", exc_info=True)
            return []
    logger.warning("[Bridge] 人脸服务未初始化，无法识别人脸。")
    return []

def bridge_identify_speaker(audio_bytes: bytes, sample_rate: int) -> List[Dict[str, Any]]:
    """供C++调用的说话人识别函数"""
    if service_management.diarization_service_instance:
        try:
            results = service_management.diarization_service_instance.identify_speaker(audio_bytes, sample_rate)
            logger.info(f"[Bridge] Speaker Identification Result: {results}")
            return results
        except Exception as e:
            logger.error(f"[Bridge] 说话人识别时出错: {e}", exc_info=True)
            return []
    logger.warning("[Bridge] Diarization服务未初始化，无法识别说话人。")
    return []

def bridge_trigger_llm_tts(client_uuid: str, asr_text: str, image_b64: str, trace_id: str):
    """供C++调用的触发LLM和TTS任务的函数"""
    logger.info(f"[Bridge] 收到来自C++的LLM/TTS触发请求 for client {client_uuid}")
    try:
        # LLM/TTS任务是异步的，我们需要在Python的事件循环中安全地运行它
        # 使用socketio.start_background_task来确保它在正确的上下文中运行
        socketio.start_background_task(
            llm_service.handle_llm_and_tts_task,
            client_uuid=client_uuid,
            asr_text=asr_text,
            image_b64=image_b64,
            socketio=socketio,
            trace_id=trace_id
        )
    except Exception as e:
        logger.error(f"[Bridge] 触发LLM/TTS任务时出错: {e}", exc_info=True)


# ==============================================================================
# --- Flask & Socket.IO 路由 (用于Web管理界面) ---
# ==============================================================================

@app.route('/')
def index():
    """服务于主管理页面"""
    return render_template('index.html')

@app.route('/management')
def management():
    """服务于媒体文件管理页面"""
    return render_template('management.html')

@app.route('/db_files/<path:subpath>')
def serve_db_files(subpath):
    """为管理界面提供对数据库中媒体文件（人脸图片、声音样本）的访问"""
    # 安全地构建路径，防止路径遍历攻击
    # 'db' 是一个逻辑路径，实际文件在configs中定义
    if subpath.startswith('faces/'):
        base_dir = os.path.join(configs.FACE_DB_PATH, "images")
        filename = subpath[len('faces/'):]
    elif subpath.startswith('speakers/'):
        base_dir = configs.SPEAKER_AUDIO_SAMPLES_PATH
        filename = subpath[len('speakers/'):]
    else:
        return "Not Found", 404
    
    return send_from_directory(os.path.abspath(base_dir), filename)


# 注册来自其他模块的Socket.IO事件回调
gui_callbacks.register_gui_callbacks(socketio)
management_callbacks.register_management_callbacks(socketio)


# ==============================================================================
# --- 应用生命周期管理 ---
# ==============================================================================

_startup_has_run = False
def on_startup():
    """初始化所有Python服务和资源"""
    global _startup_has_run
    if _startup_has_run:
        return
    
    logger.info("Python AI Service Host: Initializing services...")
    
    if configs.ENABLE_LATENCY_LOGGING:
        latency_logger.init_logger()

    # 初始化所有AI服务模型
    service_management.initialize_services()
    # 确保LLM回退音频存在
    llm_service.ensure_fallback_audio_exists()

    _startup_has_run = True
    logger.info("Python AI Service Host: Services initialized successfully.")

def on_shutdown():
    """关闭并清理所有Python服务和资源"""
    logger.info("Python AI Service Host: Shutting down...")
    service_management.shutdown_services()
    logger.info("Python AI Service Host: Cleanup complete.")


def run_cpp_engine():
    """在一个独立的线程中运行C++核心引擎"""
    logger.info("Starting C++ CoreEngine in a new thread...")
    try:
        # 创建一个字典，将Python函数映射到C++期望的名称
        # C++侧通过这个字典来回调Python函数
        python_callbacks = {
            "transcribe": bridge_transcribe,
            "recognize_faces": bridge_recognize_faces,
            "identify_speaker": bridge_identify_speaker,
            "trigger_llm_tts": bridge_trigger_llm_tts,
        }
        
        # 实例化C++引擎，并将回调函数字典传递给它
        # 注意：这要求C++的CoreEngine或其pybind包装器能够接收这个字典
        cpp_engine = my_project_bridge.CoreEngine(python_callbacks)
        cpp_engine.run()
        logger.info("C++ CoreEngine thread has finished.")
    except Exception as e:
        logger.critical(f"C++ CoreEngine thread encountered a fatal error: {e}", exc_info=True)
        # 可以在这里触发应用的优雅关闭
        os._exit(1) # 强制退出，因为核心组件已失败


# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================

if __name__ == '__main__':
    # Werkzeug重载器检测，确保启动/关闭钩子只在主进程运行一次
    is_flask_main_process = not is_running_from_reloader() or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    
    if is_flask_main_process:
        on_startup()
        atexit.register(on_shutdown)

        # 启动C++核心引擎线程
        engine_thread = threading.Thread(target=run_cpp_engine, daemon=True)
        engine_thread.start()

    # 端口应与C++ CoreEngine监听的WebSocket端口(例如5002)分开
    # 这个端口是给Web管理界面用的
    PORT = int(os.environ.get("PORT_WEB", 5003))
    
    logger.info(f"Starting Flask+SocketIO server for web GUI on http://0.0.0.0:{PORT}")
    logger.info("C++ CoreEngine is running in a background thread and will listen on its configured port (e.g., 5002).")
    
    # 运行Flask应用。`allow_unsafe_werkzeug=True` 是为了在较新版本的 werkzeug 中支持 reloader。
    # 在生产环境中，应使用Gunicorn或类似的WSGI服务器。
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, allow_unsafe_werkzeug=True)

