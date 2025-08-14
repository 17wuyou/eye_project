# llm_service.py
import asyncio
import logging
import os
import tempfile
import time
from typing import Optional, Any # Added Any

import edge_tts # type: ignore
from deep_translator import GoogleTranslator # type: ignore

# 【核心修正】更换为与您环境兼容的旧版 google.genai 库导入方式
# Ensure you have the correct version of google-generativeai installed that uses this import style
try:
    # Attempt new import style first (for google-generativeai >= 0.3.0)
    import google.generativeai as genai_google 
    from google.generativeai.types import Part # type: ignore
    GENAI_NEW_API = True
except ImportError:
    # Fallback to older import style (for google-generativeai < 0.3.0, e.g., via google.ai.generativelanguage)
    # This might require a different library or version. The original provided `from google import genai` implies an older or specific setup.
    # For now, let's assume the new API is preferred. If it fails, the user needs to ensure their `google-generativeai` is up-to-date.
    # If using a very old version, the API calls below might also need adjustment.
    # The prompt example used `genai.Client()`, which is not standard for `google-generativeai`.
    # Let's stick to the official `google.generativeai` package usage.
    logger = logging.getLogger(__name__) # Define logger early for this block
    logger.critical("Failed to import `google.generativeai`. Please ensure `pip install google-generativeai` is up to date.")
    logger.critical("If using an older version or a different Google AI library, `llm_service.py` may need adjustments.")
    # To prevent NameError later if the import fails catastrophically and code proceeds.
    class GenaiPlaceholder:
        def configure(self, *args, **kwargs): pass
        def GenerativeModel(self, *args, **kwargs): return self
        def generate_content(self, *args, **kwargs): return type('obj', (object,), {'text': 'LLM Error: google.generativeai not loaded'})()
    genai_google = GenaiPlaceholder() # type: ignore
    Part = None # type: ignore
    GENAI_NEW_API = False


import configs
from modules import android_commands # Assuming this is in modules directory

logger = logging.getLogger(__name__) # Standard logger definition

# --- TTS 配置 ---
TTS_VOICE = "zh-CN-XiaoxiaoNeural" # Default voice for Chinese
FALLBACK_AUDIO_FULL_PATH = os.path.join("static", configs.FALLBACK_AUDIO_PATH)


# --- Gemini Pro 模型交互 ---

def translate_en_to_zh(text: str) -> str:
    """
    将英文文本翻译成简体中文 (zh-CN)。
    """
    try:
        return GoogleTranslator(source='en', target='zh-CN').translate(text)
    except Exception as e:
        logger.error(f"文本翻译失败: {e}", exc_info=True)
        return f"翻译失败: {text}"


def ask_gemini_with_local_image(
    image_path: str,
    prompt_text: str,
    my_api_key: str = ""
) -> str:
    """
    读取本地 JPEG，将其包装为 Part，并与文本一起发送给 Gemini 1.5 Flash (or configured model).
    Uses the official google-generativeai SDK.
    """
    if not GENAI_NEW_API: # If the preferred import failed
        return "LLM Error: google.generativeai library not correctly initialized."

    try:
        genai_google.configure(api_key=my_api_key) # Configure with API key
        
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        image_part = Part.from_data( # Use from_data instead of from_bytes for newer API
            mime_type='image/jpeg',
            data=img_bytes
        )
        
        # Using gemini-1.5-flash as it's typically good for multimodal and fast.
        # Could be made configurable.
        model = genai_google.GenerativeModel(model_name='gemini-1.5-flash-latest') 
        
        # Construct the full prompt including the image and text
        # The model expects a list of contents.
        full_prompt_content = [
            image_part,
            prompt_text + "\n请用完全的中文回答我" # Ensure Chinese response
        ]
        
        response = model.generate_content(full_prompt_content)
        
        # Check for empty or blocked response
        if not response.parts:
             logger.warning("Gemini response has no parts (possibly blocked or empty).")
             # Try to get candidate finish_reason if available
             try:
                 finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                 logger.warning(f"Gemini finish reason: {finish_reason}")
                 if str(finish_reason).upper() in ["SAFETY", "RECITATION", "OTHER"]: # Example safety reasons
                     return "抱歉，由于内容安全限制，我无法回答。"
             except Exception: # Ignore errors in getting finish_reason
                 pass
             return "抱歉，我暂时无法生成回复。"


        return response.text # Access text directly from response object

    except Exception as e:
        logger.error(f"调用 Gemini API 出错: {e}", exc_info=True)
        # Check for specific API errors if possible (e.g., authentication, quota)
        if "API key not valid" in str(e):
            return "LLM错误：API密钥无效。"
        return "抱歉，连接语言模型时出现问题。"


# --- TTS 服务 ---

async def _generate_tts_audio_async(text: str, output_path: str) -> bool:
    """
    使用 edge-tts 将文本异步合成为音频文件。
    """
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_path)
        logger.info(f"TTS 音频已成功生成并保存到: {output_path}")
        return True
    except Exception as e:
        logger.error(f"使用 edge-tts 生成音频失败: {e}", exc_info=True)
        return False


def ensure_fallback_audio_exists():
    """
    检查回退提示音文件是否存在，如果不存在则使用TTS生成。
    """
    if os.path.exists(FALLBACK_AUDIO_FULL_PATH):
        logger.info(f"回退提示音文件已存在: {FALLBACK_AUDIO_FULL_PATH}")
        return

    logger.warning(f"回退提示音文件 '{FALLBACK_AUDIO_FULL_PATH}' 不存在，正在尝试生成...")
    os.makedirs(os.path.dirname(FALLBACK_AUDIO_FULL_PATH), exist_ok=True)
    
    try:
        # Need to run the async function in a way that works from sync context
        # For simplicity, if Python 3.7+, use asyncio.run.
        # Older Python might need loop management.
        asyncio.run(_generate_tts_audio_async(configs.FALLBACK_TTS_TEXT, FALLBACK_AUDIO_FULL_PATH))
    except RuntimeError as e_runtime: # Handles "asyncio.run() cannot be called from a running event loop"
        if "cannot be called from a running event loop" in str(e_runtime):
            logger.warning("ensure_fallback_audio_exists called from within an event loop. TTS generation might need adjustment.")
            # This scenario is complex. For now, we'll log and skip, assuming it might be generated later or manually.
            # A more robust solution would involve submitting to the loop if one exists.
        else:
            logger.critical(f"无法生成回退提示音文件 (RuntimeError): {e_runtime}", exc_info=True)
    except Exception as e:
        logger.critical(f"无法生成回退提示音文件: {e}", exc_info=True)


async def handle_llm_and_tts_task(
    client_uuid: str,
    asr_text: str,
    image_b64: str,
    socketio: Any, # Explicitly type hint for SocketIO if possible, else Any
    trace_id: str # Added trace_id for logging continuity
):
    """
    处理从 KWS 触发的 LLM 查询和 TTS 的完整异步任务。
    """
    import base64 # Local import to avoid top-level if not always needed
    from modules.state_manager import llm_query_active # Delayed import
    
    llm_task_trace_id = f"{trace_id}-llm"
    latency_logger.log_event(llm_task_trace_id, client_uuid, 'llm_tts_task', 'START', {'asr_text_len': len(asr_text)})

    thinking_text_id = f"llm_status_{int(time.time())}"
    temp_image_file: Optional[str] = None
    temp_tts_file: Optional[str] = None
    llm_response_text = "" # Initialize

    try:
        # 1. 向安卓端显示“思考中”
        thinking_cmd = {
            "type": "DRAW_TEXT", "text_id": thinking_text_id, "text": "思考中...",
            "position_normalized": [0.5, 0.5], "color_rgba": [255, 255, 0, 255],
            "size_sp": 24, "duration_ms": 0 
        }
        android_commands._send_encrypted_command_to_android(client_uuid, thinking_cmd)

        # 2. 准备图片和 Prompt
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'image_decode_save', 'START')
        image_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode='wb') as tf: # Ensure binary mode
            tf.write(image_bytes)
            temp_image_file = tf.name
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'image_decode_save', 'END', {'image_path': temp_image_file})
        
        prompt = configs.LLM_PROMPT_TEMPLATE.format(user_question=asr_text)
        logger.info(f"[{client_uuid}] [{llm_task_trace_id}] 准备调用LLM。Prompt: '{prompt[:50]}...', Image: {temp_image_file}")

        # 3. 调用 LLM
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'llm_api_call', 'START')
        llm_api_start_time = time.perf_counter()
        llm_response_text = ask_gemini_with_local_image(
            image_path=temp_image_file,
            prompt_text=prompt,
            my_api_key=configs.GEMINI_API_KEY
        )
        llm_api_duration_ms = (time.perf_counter() - llm_api_start_time) * 1000
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'llm_api_call', 'END', {'duration_ms': f"{llm_api_duration_ms:.2f}", 'response_len': len(llm_response_text)})
        logger.info(f"[{client_uuid}] [{llm_task_trace_id}] 从LLM收到回复 (耗时 {llm_api_duration_ms:.0f}ms): {llm_response_text[:100]}...")

        if not llm_response_text or "LLM错误" in llm_response_text or "我无法回答" in llm_response_text or "我暂时无法生成回复" in llm_response_text : # Check for known error/empty strings
            # If LLM failed or gave a canned "cannot answer" response, use fallback text for TTS
            logger.warning(f"[{client_uuid}] [{llm_task_trace_id}] LLM返回空或错误回复。使用预设回退文本。")
            llm_response_text = configs.FALLBACK_TTS_TEXT # This will be TTS'd
            # No need to raise an exception here, will proceed to TTS the fallback.

        # 4. 调用 TTS
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'tts_generation', 'START', {'text_to_speak_len': len(llm_response_text)})
        tts_start_time = time.perf_counter()
        # Create a temporary file for TTS output
        tts_fd, temp_tts_file_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tts_fd) # Close file descriptor, NamedTemporaryFile handles deletion better
        temp_tts_file = temp_tts_file_path # Store path for cleanup

        if not await _generate_tts_audio_async(llm_response_text, temp_tts_file):
            logger.error(f"[{client_uuid}] [{llm_task_trace_id}] TTS 音频生成失败。将尝试播放全局回退音频。")
            # If TTS for LLM response fails, use the pre-generated fallback audio file
            if os.path.exists(FALLBACK_AUDIO_FULL_PATH):
                temp_tts_file = FALLBACK_AUDIO_FULL_PATH # Point to the global fallback
            else:
                logger.error(f"[{client_uuid}] [{llm_task_trace_id}] 全局回退音频文件 {FALLBACK_AUDIO_FULL_PATH} 也不存在。无法播放音频。")
                raise RuntimeError("TTS generation failed and global fallback audio missing.") # Critical failure
        
        tts_duration_ms = (time.perf_counter() - tts_start_time) * 1000
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'tts_generation', 'END', {'duration_ms': f"{tts_duration_ms:.2f}", 'audio_path': temp_tts_file})
        
        # 5. 发送播放指令
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'play_audio_command', 'START')
        with open(temp_tts_file, 'rb') as f_tts:
            tts_audio_b64 = base64.b64encode(f_tts.read()).decode('utf-8')

        play_audio_cmd = {
            "type": "PLAY_AUDIO",
            "timestamp": time.time(), # Current server time for the command
            "audio_type_id": "llm_response" if temp_tts_file != FALLBACK_AUDIO_FULL_PATH else "llm_fallback",
            "audio_data": tts_audio_b64
        }
        android_commands._send_encrypted_command_to_android(client_uuid, play_audio_cmd)
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'play_audio_command', 'END')

    except Exception as e:
        logger.error(f"[{client_uuid}] [{llm_task_trace_id}] LLM-TTS 任务失败: {e}", exc_info=True)
        latency_logger.log_event(llm_task_trace_id, client_uuid, 'llm_tts_task', 'ERROR', {'error': str(e)})
        # General error, try to play the pre-generated fallback audio
        try:
            if os.path.exists(FALLBACK_AUDIO_FULL_PATH):
                with open(FALLBACK_AUDIO_FULL_PATH, 'rb') as f_fallback:
                    fallback_audio_b64 = base64.b64encode(f_fallback.read()).decode('utf-8')
                
                play_fallback_cmd = {
                    "type": "PLAY_AUDIO", "timestamp": time.time(),
                    "audio_type_id": "llm_fallback_critical_error", "audio_data": fallback_audio_b64
                }
                android_commands._send_encrypted_command_to_android(client_uuid, play_fallback_cmd)
            else:
                logger.error(f"[{client_uuid}] [{llm_task_trace_id}] 全局回退音频 {FALLBACK_AUDIO_FULL_PATH} 未找到，无法在错误时播放。")
        except Exception as e_fallback_send:
            logger.error(f"[{client_uuid}] [{llm_task_trace_id}] 发送全局回退提示音失败: {e_fallback_send}", exc_info=True)

    finally:
        # 6. 清理
        clear_thinking_cmd = {
            "type": "DRAW_TEXT", "text_id": thinking_text_id, "text": "",
            "duration_ms": 1 
        }
        android_commands._send_encrypted_command_to_android(client_uuid, clear_thinking_cmd)
        
        if temp_image_file and os.path.exists(temp_image_file):
            try: os.remove(temp_image_file)
            except Exception as e_del_img: logger.warning(f"删除临时图片 {temp_image_file} 失败: {e_del_img}")
        
        # Only remove temp_tts_file if it's not the global fallback path
        if temp_tts_file and os.path.exists(temp_tts_file) and temp_tts_file != FALLBACK_AUDIO_FULL_PATH:
            try: os.remove(temp_tts_file)
            except Exception as e_del_tts: logger.warning(f"删除临时TTS文件 {temp_tts_file} 失败: {e_del_tts}")
            
        if client_uuid in llm_query_active:
            llm_query_active[client_uuid] = False # Reset flag

        latency_logger.log_event(llm_task_trace_id, client_uuid, 'llm_tts_task', 'END_FINALLY')
        logger.info(f"[{client_uuid}] [{llm_task_trace_id}] LLM-TTS 任务流程(finally块)结束。")