# modules/service_management.py
import os, logging
from typing import Optional, Union, Type
import torch
import configs

# 导入服务类
from asr_service import IASRService, FunASRService, WhisperASRService
from diarization_service import DiarizationService
from face_service import FaceService

logger = logging.getLogger(__name__)

# 全局服务实例，初始化为 None
asr_service_instance: Optional[IASRService] = None
diarization_service_instance: Optional[DiarizationService] = None
face_service_instance: Optional[FaceService] = None


def initialize_services():
    global asr_service_instance, diarization_service_instance, face_service_instance

    try:
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA 可用。CUDA版本: {torch.version.cuda}, 设备: {torch.cuda.device_count()} [{torch.cuda.get_device_name(0)}]")
        else:
            logger.warning("PyTorch CUDA不可用。将使用CPU。")
    except Exception as e_torch_check:
        logger.error(f"PyTorch/CUDA检查时发生错误: {e_torch_check}", exc_info=True)

    device_pref = "cuda:0" if torch.cuda.is_available() else "cpu" # Default to cuda:0 if available
    logger.info(f"ASR/Diarization/Face 服务将尝试使用设备决策逻辑 (首选: {device_pref} if not overridden by service).")

    # --- ASR 服务初始化 ---
    logger.info(f"选择的ASR引擎: {configs.ASR_ENGINE_CHOICE}")
    if configs.ASR_ENGINE_CHOICE == "funasr":
        logger.info("ASR引擎: FunASR (Online Paraformer).")
        asr_service_instance = FunASRService(device_preference=device_pref) # FunASR handles device internally to some extent
    elif configs.ASR_ENGINE_CHOICE == "whisper":
        logger.info(f"ASR引擎: Whisper (模型: {configs.WHISPER_MODEL_SIZE}).")
        asr_service_instance = WhisperASRService(model_size=configs.WHISPER_MODEL_SIZE, language="zh", device_preference=device_pref)
    
    if asr_service_instance:
        try:
            asr_service_instance.load_model(device_preference=device_pref) # Pass preference again
            logger.info(f"ASR服务 ({configs.ASR_ENGINE_CHOICE}) 已成功加载模型。")
        except Exception as e_asr_load:
            logger.error(f"加载ASR模型 ({configs.ASR_ENGINE_CHOICE}) 失败: {e_asr_load}", exc_info=True)
            asr_service_instance = None
    else:
        logger.warning("没有选择有效的ASR引擎，ASR服务将不可用。")

    # --- Diarization 服务初始化 ---
    try:
        HF_AUTH_TOKEN = os.environ.get("HF_TOKEN") # No need for 'or None', get() returns None if not found
        diarization_service_instance = DiarizationService(
            auth_token=HF_AUTH_TOKEN,
            device_preference=device_pref, # Pyannote uses this preference
            diarization_model_name=configs.DIARIZATION_MODEL_NAME,
            embedding_model_name=configs.EMBEDDING_MODEL_NAME,
            user_similarity_threshold=configs.USER_SPEAKER_SIMILARITY_THRESHOLD,
            speaker_database_path=configs.SPEAKER_DATABASE_PATH,
            speaker_audio_samples_path=configs.SPEAKER_AUDIO_SAMPLES_PATH,
            new_speaker_similarity_threshold=configs.NEW_SPEAKER_SIMILARITY_THRESHOLD,
            min_segment_duration_for_embedding=configs.MIN_SEGMENT_DURATION_FOR_EMBEDDING
        )
        if diarization_service_instance:
            diarization_service_instance.load_model() # Loads to the device preferred in init
            logger.info("Diarization服务已成功加载模型。")
            if diarization_service_instance.pipeline_instance and diarization_service_instance.embedding_model_instance:
                if configs.USER_VOICE_SAMPLE_PATH and os.path.exists(configs.USER_VOICE_SAMPLE_PATH):
                    try:
                        diarization_service_instance.register_user_from_file(configs.USER_VOICE_SAMPLE_PATH)
                        logger.info(f"已成功注册用户声音: {configs.USER_VOICE_SAMPLE_PATH}")
                    except Exception as e_reg_user:
                        logger.error(f"注册用户声音失败 ({configs.USER_VOICE_SAMPLE_PATH}): {e_reg_user}", exc_info=True)
                elif configs.USER_VOICE_SAMPLE_PATH: # Path configured but file not found
                    logger.warning(f"用户声音样本文件未找到: '{configs.USER_VOICE_SAMPLE_PATH}'")
                # else: No path configured, no message needed
            # else: Models didn't load, already logged by DiarizationService
    except Exception as e_diar_init_load:
        logger.error(f"初始化或加载DiarizationService失败: {e_diar_init_load}", exc_info=True)
        diarization_service_instance = None


    # --- 人脸服务初始化 (InsightFace) ---
    if configs.ENABLE_FACE_SERVICE:
        try:
            logger.info("正在初始化人脸服务 (InsightFace)...")
            face_service_instance = FaceService( # InsightFace handles its own device selection internally
                db_path=configs.FACE_DB_PATH,
                recognition_threshold=configs.FACE_RECOGNITION_THRESHOLD,
                detection_confidence=configs.FACE_DETECTION_CONFIDENCE
            )
            # No explicit load_model for FaceService, it's done in __init__
            logger.info("人脸服务 (InsightFace) 已初始化。")
        except Exception as e_face_init:
            logger.critical(f"初始化FaceService (InsightFace) 失败: {e_face_init}", exc_info=True)
            face_service_instance = None
    else:
        logger.info("人脸检测服务已在配置中禁用。")


def shutdown_services():
    global asr_service_instance, diarization_service_instance, face_service_instance
    
    if diarization_service_instance:
        logger.info("正在卸载Diarization模型...")
        try:
            diarization_service_instance.unload_model()
        except Exception as e:
            logger.error(f"卸载Diarization模型时出错: {e}")
        diarization_service_instance = None # Ensure it's None after attempt
    
    if asr_service_instance:
        logger.info(f"正在卸载ASR模型 ({configs.ASR_ENGINE_CHOICE})...")
        try:
            asr_service_instance.unload_model()
        except Exception as e:
            logger.error(f"卸载ASR模型时出错: {e}")
        asr_service_instance = None

    if face_service_instance:
        logger.info("正在卸载FaceService模型(InsightFace)...")
        try:
            # FaceService might have an explicit unload or cleanup
            if hasattr(face_service_instance, 'unload_model'):
                face_service_instance.unload_model() # Call if exists
            # Otherwise, just deleting the instance
        except Exception as e:
            logger.error(f"卸载FaceService模型时出错: {e}")
        face_service_instance = None 
    
    logger.info("所有服务已尝试卸载。")


def reload_services_databases():
    global diarization_service_instance, face_service_instance
    
    logger.info("收到数据库重载请求...")
    
    if diarization_service_instance:
        try:
            logger.info("正在重载说话人数据库...")
            diarization_service_instance._load_speaker_database() # This also updates matrix
            logger.info("说话人数据库重载完毕。")
        except Exception as e:
            logger.error(f"重载说话人数据库时出错: {e}", exc_info=True)
            
    if face_service_instance:
        try:
            logger.info("正在重载人脸数据库 (InsightFace)...")
            face_service_instance._load_database() # This also updates matrix
            logger.info("人脸数据库重载完毕。")
        except Exception as e:
            logger.error(f"重载人脸数据库时出错: {e}", exc_info=True)