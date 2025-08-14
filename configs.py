# configs.py



# ==============================================================================
# --- AI模型缓存路径 ---
# ==============================================================================
# 定义一个文件夹，用来存放从网上下载的AI模型（如Whisper, Pyannote），避免每次启动都重新下载
MODEL_CACHE_DIR = "model_cache"#Truman加

# ==============================================================================
# --- 队列与实时性配置 ---
# ==============================================================================
# 定义输入数据包在队列中可以等待的最大时间（毫秒）。超过此时间的数据包将被丢弃。
# 设置为 0 表示禁用此检查。
MAX_INPUT_QUEUE_TIME_MS = 2000 # 2秒#truman加入


# Maximum number of video frames to hold in the backlog for processing.
# This prevents the queue from growing indefinitely if processing is slow.
MAX_VIDEO_PROCESSING_BACKLOG_FRAMES = 20#Truamn

# ==============================================================================
# --- 积压队列配置 ---
# ==============================================================================
# 积压在队列中等待处理的音频数据的最大时长（毫秒）
MAX_AUDIO_PROCESSING_BACKLOG_MS = 3000 # 3秒#Truman






#-----------有关于加密数据------------
AES_KEY_STRING = "lFuvFVCBAVTV3c5G+TBGXBoYeWxYp2+vujq2WS41ygk=" # AES加密密钥

#-------------有关音频流参数------------
AUDIO_SAMPLE_RATE = 16000  # 应用全局音频采样率 (Hz)
AUDIO_CHANNELS = 1         # 音频通道数
AUDIO_SAMPLE_WIDTH = 2     # 音频样本宽度 (字节)
AUDIO_CHUNK_DURATION_MS = 200 # 音频块时长 (毫秒)

DIALOGUE_SILENCE_TIMEOUT_MS = 1500 # 对话静音超时时间 (毫秒) - 稍稍延长以适应KWS后对话
MAX_ACCUMULATED_CHUNKS_FOR_DIALOGUE = int((30 * 1000) / AUDIO_CHUNK_DURATION_MS) # 约30秒音频
SHORT_SILENCE_PADDING_CHUNKS = 2 # 短静音填充块数 (VAD相关)

RMS_VAD_THRESHOLD = 10  # VAD (语音活动检测) 的RMS能量阈值, audio_utils.py 使用

#------------关键帧检测 (增强版)------------
KEYFRAME_DETECTOR_THRESHOLD = 0.65 
KEYFRAME_HISTORY_SIZE = 10       
KEYFRAME_RESIZE_DIM = (160, 90)  
KEYFRAME_MOTION_THRESHOLD_PERCENT = 0.5 
KEYFRAME_MOTION_DIFF_THRESHOLD_VALUE = 25 
KEYFRAME_COOLDOWN_SECONDS = 2.0  

KEYFRAME_USE_H_CHANNEL_FOR_HIST = True 
KEYFRAME_HIST_CHANNELS_H = [0]   
KEYFRAME_HIST_SIZE_H = [180]     
KEYFRAME_HIST_RANGES_H = [0, 180] 
KEYFRAME_HIST_CHANNELS_GRAY = [0] 
KEYFRAME_HIST_SIZE_GRAY = [256]   
KEYFRAME_HIST_RANGES_GRAY = [0, 256]

#------------有关于说话人分割模型 (Diarization)------------
USER_VOICE_SAMPLE_PATH = "user_sample.mp3" 
USER_SPEAKER_SIMILARITY_THRESHOLD = 0.2 
SPEAKER_DATABASE_PATH = "db/historical_speakers.pt" 
SPEAKER_AUDIO_SAMPLES_PATH = "db/speaker_audio_samples" 
NEW_SPEAKER_SIMILARITY_THRESHOLD = 0.2 
DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1" 
EMBEDDING_MODEL_NAME = "pyannote/embedding" 
MIN_SEGMENT_DURATION_FOR_EMBEDDING = 0.5 

#-----------有关于ASR模型------------
ASR_ENGINE_CHOICE = "whisper" 
WHISPER_MODEL_SIZE = "large-v3" 

#======= 新增：KWS, LLM, TTS 配置 =======
# --- KWS (Porcupine) 配置 ---
# 请从 https://console.picovoice.ai/ 获取你的 AccessKey
PICOVOICE_ACCESS_KEY = "7M0CcTw9MNobXpzy+zGgFIEqAaTXF+XmkvTTyi/zH04VWmcdp+csNg==" 
# Porcupine 自带的通用模型文件，通常无需更改
KWS_MODEL_PATH = "kws_models/porcupine_params_zh.pv" # 设置为 None 或省略，SDK会自动使用默认模型
# 在 PicoVoice Console 创建的中文唤醒词模型文件 (.ppn) 的路径列表
KWS_KEYWORD_PATHS = ["kws_models/xiaotong_zh_linux_v3_0_0.ppn"] 
# 对应每个唤醒词的灵敏度 (0.0 - 1.0)，越高越容易触发
KWS_KEYWORD_SENSITIVITIES = [0.5] 
# VAD 激活后，用于检测 KWS 的音频时长（秒）
KWS_AUDIO_BUFFER_SECONDS = 2.0 

# --- LLM (Gemini) 配置 ---
# 请从 Google AI Studio 获取你的 API Key
GEMINI_API_KEY = "AIzaSyBjwevT2P4ICTNPcFPuqyfWbQn9_2HeORk"
# LLM 的提示词模板，{user_question} 将被替换为用户的 ASR 识别结果
LLM_PROMPT_TEMPLATE = """
你是一个智能助手。请根据用户的问题，并参考提供的实时图像，给出一个简洁、有帮助的回答。
图像捕捉了用户提问时所见的场景，它可以为你提供额外的上下文信息。
你的回答应直接针对用户的问题，不要重复问题内容。给我自然的，精炼的简短回答，完全回复我中文。

用户的问题是: "{user_question}"
"""

# --- TTS 与回退配置 ---
FALLBACK_TTS_TEXT = "抱歉，我暂时无法回答"
# 回退提示音的保存路径 (相对于项目根目录的 static 文件夹)
FALLBACK_AUDIO_PATH = "audio/fallback.mp3"
#=======================================

#-------字幕与事件数据集保存-------
SAVE_SUBTITLES_TO_FILE = False
SUBTITLES_SAVE_DIR = "saved_subtitles"

DATASET_SAVE_EVENTS = True
DATASET_ROOT_DIR = "event_dataset"

#-------安卓客户端字幕绘制参数-------
ANDROID_SUBTITLE_BASE_Y_NORMALIZED = 0.95      
ANDROID_SUBTITLE_LINE_HEIGHT_NORMALIZED = 0.07 
ANDROID_SUBTITLE_TEXT_SIZE_SP = 16             
ANDROID_SUBTITLE_DURATION_MS = 10000           
ANDROID_SUBTITLE_MAX_LINES = 4                 

#-------其他-------
SUBTITLE_BLOCKED_PHRASES = [
    "请不吝点赞", "订阅", "转发", "打赏",
    "支持明镜", "点点栏目", "谢谢观看", "欢迎关注"
]
ASR_NO_SPEECH_INDICATORS = ["[未检测到语音]", "[no speech]", "[no voice]", ""]

# --- 性能监控配置 ---
ENABLE_PERFORMANCE_MONITORING = False
PERFORMANCE_REPORT_INTERVAL_SECONDS = 10
LOG_DETAILED_COMPONENT_TIMES = False

# =======================================
# --- 新增: 详细延迟日志配置 ---
ENABLE_LATENCY_LOGGING = True  # 总开关，True为开启，False为关闭
LATENCY_LOG_FILE = "log.txt"   # 日志文件名
# =======================================


# --- 调试用音频保存配置 ---
APP_SAVE_RAW_AUDIO_CHUNKS = False
APP_SAVE_RAW_AUDIO_INTERVAL = 10
APP_AUDIO_SAVE_PATH_PREFIX = "app_raw_audio_chunk"

# --- 线程池配置 ---
KEYFRAME_PROCESSING_MAX_WORKERS = 1
# 新增：LLM 和 TTS 任务是 IO 密集型，可以设置多个工作线程
LLM_PROCESSING_MAX_WORKERS = 4 

# ==============================================================================
# --- 人脸检测与识别服务配置 (InsightFace) ---
# ==============================================================================
ENABLE_FACE_SERVICE = True
# --- 已移除 FACE_SERVICE_DEVICE，服务内部将实现智能GPU选择逻辑 ---
FACE_PROCESSING_INTERVAL_SECONDS = 1.0
FACE_DB_PATH = "db/face_database"
# --- 已移除 DeepFace 的 FACE_MODEL_NAME 和 FACE_DETECTOR_BACKEND ---
# --- 新增 InsightFace 的配置参数 ---
FACE_RECOGNITION_THRESHOLD = 0.5      # 识别相似度阈值 (余弦相似度，越高越相似)
FACE_DETECTION_CONFIDENCE = 0.5       # 人脸检测置信度阈值

# --- 安卓端人脸信息绘制参数 (保持不变) ---
ANDROID_FACE_INFO_BASE_Y_NORMALIZED = 0.05       
ANDROID_FACE_INFO_LINE_HEIGHT_NORMALIZED = 0.05  
ANDROID_FACE_INFO_TEXT_SIZE_SP = 14              
ANDROID_FACE_INFO_DURATION_MS = int(FACE_PROCESSING_INTERVAL_SECONDS * 1000) + 500 
ANDROID_FACE_INFO_MAX_FACES = 5