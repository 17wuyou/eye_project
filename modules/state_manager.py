# modules/state_manager.py
import threading
from collections import deque
from typing import Dict, Any, Deque, Set, Optional # 确保 Optional 已导入

# --- 原始 WebSocket 客户端管理 ---
raw_ws_clients: Dict[str, Any] = {} # 保存实际的 WebSocket 连接对象

# --- GUI 客户端管理 ---
socketio_gui_sids: Set[str] = set()
GUI_LISTENERS_ROOM = 'gui_listeners_room' # SocketIO 房间常量

# --- 关键帧检测状态 ---
keyframe_detectors: Dict[str, Any] = {} # 按客户端存储 KeyframeDetector 实例
last_keyframe_check_times: Dict[str, float] = {} # 跟踪上次运行关键帧检测的时间

# --- 人脸处理状态 ---
last_face_check_times: Dict[str, float] = {} # 跟踪上次运行人脸处理的时间

# --- 新增：KWS 与 LLM 状态 ---
client_kws_detectors: Dict[str, Any] = {} # 按客户端存储 KwsService 实例
client_kws_audio_buffer: Dict[str, bytearray] = {} # 按客户端存储用于KWS检测的初始音频数据
llm_query_active: Dict[str, bool] = {} # 标记客户端是否正在进行LLM查询，防止并发
last_video_frames: Dict[str, str] = {} # 按客户端缓存最新的 base64 视频帧

# --- 音频处理状态 ---
client_audio_accumulators: Dict[str, Dict[str, Any]] = {} # 为 VAD、ASR 缓冲音频块
app_audio_save_counter: Dict[str, int] = {} # 用于保存原始音频块的计数器

# --- 事件数据状态 (用于数据集保存) ---
client_event_data: Dict[str, Dict[str, Any]] = {} # 按客户端保存正在进行的事件数据

# --- 性能监控状态 ---
client_performance_data: Dict[str, Dict[str, Any]] = {} # 按客户端存储性能指标
_performance_reporting_timer: Optional[threading.Timer] = None # 定期性能报告的计时器

# --- 客户端数据锁 ---
# 每个客户端性能数据和事件数据的集中锁
client_data_locks: Dict[str, Dict[str, threading.Lock]] = {}