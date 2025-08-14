import logging, os, sys 
from typing import Optional, Any
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO 
from flask_sock import Sock 
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
import torch 
import cv2 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir) 

import configs
from modules import latency_logger # Ensure this is imported after sys.path update

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('audio_utils').setLevel(logging.INFO)
logging.getLogger('asr_service').setLevel(logging.INFO)
logging.getLogger('diarization_service').setLevel(logging.INFO)
logging.getLogger('keyframe_detector').setLevel(logging.INFO)
logging.getLogger('face_service').setLevel(logging.INFO)
logging.getLogger('pydub.utils').setLevel(logging.WARNING)
logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
logging.getLogger('deepface').setLevel(logging.WARNING) # Though DeepFace might not be used now
logging.getLogger('insightface').setLevel(logging.INFO) # InsightFace logs can be useful
logging.getLogger('modelscope').setLevel(logging.WARNING) # Reduce ModelScope verbosity

logging.getLogger('modules.audio_processing_logic').setLevel(logging.INFO)
logging.getLogger('modules.video_processing_logic').setLevel(logging.INFO)
logging.getLogger('modules.websocket_callbacks').setLevel(logging.INFO)
logging.getLogger('kws_service').setLevel(logging.INFO)
logging.getLogger('modules.llm_service').setLevel(logging.INFO) # Corrected llm_service logger name

# --- Flask 应用和扩展初始化 ---
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)
# Ensure model cache directory exists
os.makedirs(configs.MODEL_CACHE_DIR, exist_ok=True)
# Set environment variables for model caches if not already set (optional, libraries might handle this)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(configs.MODEL_CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
if "MODELSCOPE_CACHE" not in os.environ:
    os.environ["MODELSCOPE_CACHE"] = os.path.join(configs.MODEL_CACHE_DIR, "modelscope")
    os.makedirs(os.environ["MODELSCOPE_CACHE"], exist_ok=True)


app = Flask(__name__, static_folder=static_dir)

app.config['SECRET_KEY'] = configs.AES_KEY_STRING
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)
sock = Sock(app)

from modules import service_management
from modules import performance_utils
from modules import websocket_callbacks
from modules import gui_callbacks
from modules import llm_service # Ensure this import is correct
from modules import management_callbacks

# --- 全局线程池 ---
keyframe_executor: Optional[ThreadPoolExecutor] = None
if configs.KEYFRAME_PROCESSING_MAX_WORKERS > 0:
    keyframe_executor = ThreadPoolExecutor(max_workers=configs.KEYFRAME_PROCESSING_MAX_WORKERS, thread_name_prefix='KFDetectThread')
    logger.info(f"关键帧检测线程池已初始化，最大工作线程数: {configs.KEYFRAME_PROCESSING_MAX_WORKERS}")
else:
    logger.info("关键帧异步处理已禁用 (KEYFRAME_PROCESSING_MAX_WORKERS <= 0)。")

face_executor: Optional[ThreadPoolExecutor] = None
if configs.ENABLE_FACE_SERVICE:
    # Use FACE_PROCESSING_MAX_WORKERS from configs
    face_max_workers = getattr(configs, 'FACE_PROCESSING_MAX_WORKERS', 1) 
    if face_max_workers > 0 :
        face_executor = ThreadPoolExecutor(max_workers=face_max_workers, thread_name_prefix='FaceProcThread')
        logger.info(f"人脸处理线程池已初始化 (最大工作线程数: {face_max_workers})")
    else:
        logger.info("人脸处理异步已禁用 (FACE_PROCESSING_MAX_WORKERS <= 0)。")

llm_executor: Optional[ThreadPoolExecutor] = None
if configs.LLM_PROCESSING_MAX_WORKERS > 0:
    llm_executor = ThreadPoolExecutor(max_workers=configs.LLM_PROCESSING_MAX_WORKERS, thread_name_prefix='LLM_Task_Thread')
    logger.info(f"LLM任务线程池已初始化，最大工作线程数: {configs.LLM_PROCESSING_MAX_WORKERS}")
else:
    logger.info("LLM 异步处理已禁用。")

# --- 音频处理线程池 (新增) ---
# This is for the main audio processing logic that calls diarization and ASR.
# If diarization/ASR services themselves are blocking, this pool helps offload that blocking call.
AUDIO_PROCESSING_MAX_WORKERS = getattr(configs, 'AUDIO_PROCESSING_MAX_WORKERS', 2) # Default to 2 if not in configs
audio_executor: Optional[ThreadPoolExecutor] = None
if AUDIO_PROCESSING_MAX_WORKERS > 0:
    audio_executor = ThreadPoolExecutor(max_workers=AUDIO_PROCESSING_MAX_WORKERS, thread_name_prefix='AudioProcThread')
    logger.info(f"音频处理任务线程池已初始化，最大工作线程数: {AUDIO_PROCESSING_MAX_WORKERS}")
else:
    logger.info("音频处理异步任务已禁用 (AUDIO_PROCESSING_MAX_WORKERS <= 0)。")


if configs.DATASET_SAVE_EVENTS:
    os.makedirs(configs.DATASET_ROOT_DIR, exist_ok=True)
    logger.info(f"事件数据集保存功能已启用。数据将保存到目录: '{configs.DATASET_ROOT_DIR}'")
    if getattr(configs, 'SAVE_SUBTITLES_TO_FILE', False):
        logger.info("旧的字幕文件保存功能 (SAVE_SUBTITLES_TO_FILE) 已被新的事件数据集保存功能覆盖并禁用 (字幕通过事件保存)。")
else:
    logger.info("事件数据集保存功能已禁用。")
    if getattr(configs, 'SAVE_SUBTITLES_TO_FILE', False):
        subtitle_save_dir = getattr(configs, 'SUBTITLES_SAVE_DIR', "saved_subtitles")
        os.makedirs(subtitle_save_dir, exist_ok=True)
        logger.info(f"字幕本地保存功能已启用 (旧方法)。字幕将保存到目录: '{subtitle_save_dir}'")
    else:
        logger.info("字幕本地保存功能已禁用 (旧方法)。")

if configs.ENABLE_PERFORMANCE_MONITORING:
    performance_utils.set_app_instance(app) # Pass app instance for reloader check
    logger.info(f"性能监控已启用。报告周期: {configs.PERFORMANCE_REPORT_INTERVAL_SECONDS}秒。详细组件耗时日志: {configs.LOG_DETAILED_COMPONENT_TIMES}")

@app.route('/')
def control_panel_route():
    return render_template('control_panel.html')

@app.route('/management')
def management_panel_route():
    return render_template('management_panel.html')

@app.route('/db_files/<folder>/<path:filename>')
def serve_db_files(folder, filename):
    safe_base_dir = os.path.abspath(os.path.join(current_dir, 'db')) # Project's db folder
    
    target_dir_name = ""
    if folder == 'speakers':
        target_dir_name = configs.SPEAKER_AUDIO_SAMPLES_PATH
    elif folder == 'faces':
        target_dir_name = os.path.join(configs.FACE_DB_PATH, "images")
    else:
        return "Not Found", 404
    
    # Ensure target_dir_name is absolute for comparison
    directory = os.path.abspath(target_dir_name)
        
    # Security check: Make sure the resolved directory is within the project's 'db' directory structure
    # or the configured paths if they are outside 'db' but explicitly allowed.
    # For simplicity, let's assume configured paths are subdirectories of the project's 'db' or project root.
    # A more robust check might involve a whitelist of allowed base directories.
    
    # Check 1: Is it within the main 'db' folder of the project?
    is_within_project_db = directory.startswith(safe_base_dir)
    
    # Check 2: If not, is it exactly one of the configured paths (already made absolute)?
    # This allows configs to point outside the 'db' folder if necessary, but still specific.
    allowed_abs_paths = [
        os.path.abspath(configs.SPEAKER_AUDIO_SAMPLES_PATH),
        os.path.abspath(os.path.join(configs.FACE_DB_PATH, "images"))
    ]
    is_explicitly_configured_path = directory in allowed_abs_paths

    if not (is_within_project_db or is_explicitly_configured_path):
        logger.warning(f"Forbidden access attempt to: {directory} for file {filename}")
        return "Forbidden", 403
        
    return send_from_directory(directory, filename)


# Pass audio_executor to websocket_callbacks
websocket_callbacks.register_websocket_handlers(sock, socketio, keyframe_executor, face_executor, llm_executor, audio_executor)
gui_callbacks.register_gui_callbacks(socketio)
management_callbacks.register_management_callbacks(socketio)


def on_startup():
    logger.info("应用程序启动中...")
    
    if configs.ENABLE_LATENCY_LOGGING:
        latency_logger.init_logger() # Initialize latency logger
        logger.info(f"详细延迟日志已启用，将记录到文件: {configs.LATENCY_LOG_FILE}")
    else:
        logger.info("详细延迟日志已禁用。")

    try:
        logger.info("正在检查核心依赖...")
        assert np is not None, "NumPy 未导入"
        assert torch is not None, "PyTorch 未导入"
        assert cv2 is not None, "OpenCV (cv2) 未导入"
        logger.info("核心依赖检查通过。")
    except (ImportError, AssertionError) as e: 
        logger.critical(f"严重错误: 缺少核心依赖: {e}。可能需要安装: pip install numpy torch opencv-python", exc_info=True)
        # sys.exit(1) # Consider exiting if core dependencies are missing
    
    ffmpeg_found = any(os.access(os.path.join(p, f), os.X_OK) 
                       for p in os.environ.get("PATH","").split(os.pathsep) 
                       for f in ["ffmpeg", "ffmpeg.exe"])
    if not ffmpeg_found: 
        logger.warning("FFmpeg未在PATH中找到。Diarization或某些音频/视频处理功能可能受限。")
    else: 
        logger.info("FFmpeg已在PATH中找到。")

    service_management.initialize_services()
    llm_service.ensure_fallback_audio_exists()

    logger.info("应用程序启动完成。")

def on_shutdown():
    logger.info("应用程序正在关闭...")
    
    executors_to_shutdown = {
        "关键帧检测": keyframe_executor,
        "人脸处理": face_executor,
        "LLM任务": llm_executor,
        "音频处理": audio_executor # Add audio_executor
    }
    for name, executor in executors_to_shutdown.items():
        if executor:
            logger.info(f"正在关闭 {name} 线程池...")
            executor.shutdown(wait=True)
            logger.info(f"{name} 线程池已关闭。")

    if configs.ENABLE_PERFORMANCE_MONITORING:
        performance_utils.stop_performance_reporting()

    service_management.shutdown_services()
    logger.info("清理完成。应用程序退出。")

if __name__ == '__main__':
    def is_flask_main_process() -> bool:
        # Check if Werkzeug reloader is active and this is the main process
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true": return True
        # If not in debug mode (reloader likely off), or reloader is off, this is the main process
        if not app.debug and not os.environ.get("WERKZEUG_RUN_MAIN"): return True
        return False
    
    # Make it accessible for performance_utils reloader check
    setattr(app, 'is_running_from_reloader', is_flask_main_process)


    if app.is_running_from_reloader(): # True if main process of reloader, or no reloader
        logger.info("Flask应用在主进程中运行 (或非重载模式)。执行启动任务。")
        on_startup()
    else:
        logger.info("Flask应用在Werkzeug重载器的子进程中运行。主启动任务将由父进程处理。")

    PORT = int(os.environ.get("PORT", 5002))
    HOST = '0.0.0.0' # Listen on all available network interfaces
    logger.info(f"准备启动Flask应用 (带SocketIO) 于 {HOST}:{PORT}。调试模式={app.debug} (影响重载器)。")
    logger.info(f"Werkzeug主进程状态 (决定是否执行on_startup/on_shutdown): {app.is_running_from_reloader()}")


    try:
        # use_reloader=False because we handle startup/shutdown manually based on main process check
        socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True, use_reloader=False) 
    except Exception as e_server_start:
        logger.critical(f"启动Flask SocketIO服务器失败: {e_server_start}", exc_info=True)
    finally:
        # This will only be reached if socketio.run returns (e.g., server stopped)
        # Ensure shutdown runs only in the main process context
        if app.is_running_from_reloader():
             on_shutdown()