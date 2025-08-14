# asr_service.py
import logging
from abc import ABC, abstractmethod
import numpy as np
import torch
import torchaudio
# import torchaudio.functional as F # 未直接使用 F，可以移除
import tempfile
import soundfile as sf
# import io # 未直接使用，可以移除
import os

logger = logging.getLogger(__name__)

class IASRService(ABC):
    @abstractmethod
    def load_model(self, device_preference: str):
        pass

    @abstractmethod
    def unload_model(self):
        pass

    @abstractmethod
    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int) -> str:
        pass

class FunASRService(IASRService):
    def __init__(self, device_preference: str = "cuda:0"):
        self.model = None
        self.opencc_converter = None
        self.device_preference = device_preference
        self.device = None
        try:
            from opencc import OpenCC
            self.opencc_converter = OpenCC('t2s')
            logger.info("FunASRService: OpenCC loaded for Traditional to Simplified Chinese conversion.")
        except ImportError:
            logger.warning("FunASRService: OpenCC not installed. Conversion will not be available.")
            self.opencc_converter = None
        except Exception as e:
            logger.warning(f"FunASRService: Failed to initialize OpenCC with config 't2s': {e}. Conversion will not be available.", exc_info=True)
            self.opencc_converter = None

    def load_model(self, device_preference: str = "cuda:0"):
        self.device_preference = device_preference
        if self.model is not None:
            logger.info("FunASRService: Model already loaded.")
            return
        try:
            from modelscope.pipelines import pipeline as modelscope_pipeline_func
            from modelscope.utils.constant import Tasks as ModelscopeTasks

            if "cuda" in self.device_preference and torch.cuda.is_available():
                self.device = self.device_preference
                logger.info(f"FunASRService: Attempting to use CUDA device: {self.device}")
            else:
                self.device = "cpu"
                logger.info("FunASRService: CUDA not available or not preferred, using CPU.")

            model_id = 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
            logger.info(f"FunASRService: Loading model '{model_id}' to device '{self.device}'...")
            
            self.model = modelscope_pipeline_func(
                task=ModelscopeTasks.auto_speech_recognition,
                model=model_id,
                device=self.device
            )
            logger.info(f"FunASRService: Model '{model_id}' loaded successfully on {self.device}.")
        except ImportError:
            logger.error("FunASRService: Failed to import 'modelscope'. Ensure it is installed (pip install modelscope).", exc_info=True)
            self.model = None
        except Exception as e:
            logger.error(f"FunASRService: Failed to load model '{model_id if 'model_id' in locals() else 'unknown'}': {e}", exc_info=True)
            self.model = None # (错误处理保持不变)


    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device and "cuda" in str(self.device): # 确保 self.device 是字符串或有 .type 属性
                 torch.cuda.empty_cache()
            logger.info("FunASRService: Model unloaded and CUDA cache cleared (if applicable).")

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int) -> str:
        if not self.model:
            logger.warning("FunASRService: Model not loaded. Cannot transcribe.")
            return ""
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            current_sample_rate = sample_rate
            
            if sample_rate != 16000:
                logger.debug(f"FunASRService: Input SR {sample_rate}Hz, model expects 16000Hz. Resampling.")
                audio_tensor_float = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                resampled_tensor_float = resampler(audio_tensor_float)
                audio_np = (resampled_tensor_float.squeeze(0).numpy() * 32768.0).astype(np.int16)
                current_sample_rate = 16000
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
                sf.write(tmp_wav_file.name, audio_np, current_sample_rate, format='WAV', subtype='PCM_16')
                logger.debug(f"FunASRService: Transcribing temp WAV: {tmp_wav_file.name} (SR: {current_sample_rate}Hz, Samples: {len(audio_np)})")
                rec_result = self.model(tmp_wav_file.name)

            text = ""
            if rec_result and isinstance(rec_result, dict) and "text" in rec_result:
                text = rec_result["text"]
            elif rec_result and isinstance(rec_result, list) and len(rec_result) > 0 and isinstance(rec_result[0], dict) and "text" in rec_result[0]:
                 text = rec_result[0]["text"]
            else:
                logger.warning(f"FunASRService: Unexpected result format: {type(rec_result)}")

            if self.opencc_converter and text:
                converted_text = self.opencc_converter.convert(text)
                logger.debug(f"FunASRService transcription (raw): '{text[:30]}...', (converted): '{converted_text[:30]}...'")
                return converted_text.strip()
            else:
                logger.debug(f"FunASRService transcription (raw, no conversion): '{text[:30]}...'")
                return text.strip()
        except Exception as e:
            logger.error(f"FunASRService: Error during transcribe_bytes: {e}", exc_info=True)
            return ""


class WhisperASRService(IASRService):
    # --- 强制语言为中文 ---
    def __init__(self, model_size="small", language="zh", device_preference: str = "cuda:0"):
        self.model_size = model_size
        # 确保 language 参数在此处被设定为 'zh' 或其他中文代码
        self.language = language.lower() if language else "zh" # 默认为中文，并转为小写
        logger.info(f"WhisperASRService: Initializing with language forced to '{self.language}'.")
        self.model = None
        self.opencc_converter = None
        self.device_preference = device_preference
        self.device = None
        try:
            from opencc import OpenCC
            self.opencc_converter = OpenCC('t2s')
            logger.info("WhisperASRService: OpenCC loaded for Traditional to Simplified Chinese conversion.")
        except ImportError:
            logger.warning("WhisperASRService: OpenCC not installed. Conversion will not be available.")
            self.opencc_converter = None
        except Exception as e:
            logger.warning(f"WhisperASRService: Failed to initialize OpenCC with config 't2s': {e}. Conversion will not be available.", exc_info=True)
            self.opencc_converter = None

    def load_model(self, device_preference: str="cuda:0"):
        self.device_preference = device_preference
        if self.model is not None:
            logger.info(f"WhisperASRService: Model '{self.model_size}' already loaded.")
            return
        try:
            import whisper as openai_whisper

            if "cuda" in self.device_preference and torch.cuda.is_available():
                self.device = torch.device(self.device_preference)
                logger.info(f"WhisperASRService: Attempting to use CUDA device: {self.device}")
            else:
                self.device = torch.device("cpu")
                logger.info("WhisperASRService: CUDA not available or not preferred, using CPU.")
            
            logger.info(f"WhisperASRService: Loading model '{self.model_size}' to device '{self.device}' (Language: {self.language})...")
            self.model = openai_whisper.load_model(self.model_size, device=self.device)
            # Whisper 的 load_model 不直接接受 language 参数，是在 transcribe 时指定
            logger.info(f"WhisperASRService: Model '{self.model_size}' loaded successfully on {self.device}.")

        except ImportError: # (错误处理保持不变)
            logger.error("WhisperASRService: Failed to import 'whisper' (openai-whisper). Please ensure it is installed.", exc_info=True)
            self.model = None
        except Exception as e:
            logger.error(f"WhisperASRService: Failed to load model '{self.model_size}': {e}", exc_info=True)
            self.model = None


    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info("WhisperASRService: Model unloaded and CUDA cache cleared (if applicable).")

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int) -> str:
        if not self.model:
            logger.warning("WhisperASRService: Model not loaded. Cannot transcribe.")
            return ""
        try:
            audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_np_float32 = audio_np_int16.astype(np.float32) / 32768.0
            
            transcribe_options = {
                "fp16": (self.device.type == 'cuda'),
                "language": self.language, # 强制指定语言
                # --- 添加中文提示 ---
                "initial_prompt": "以下是普通话的对话。" # 或者更具体的内容，如果知道上下文
            }
            # 对于某些模型和情况，task='transcribe' 可能有助于避免翻译
            # transcribe_options["task"] = "transcribe"


            logger.debug(f"WhisperASRService: Transcribing audio (Samples: {len(audio_np_float32)}, SR: {sample_rate}Hz, Lang: {self.language}, Device: {self.device}, Prompt: '{transcribe_options['initial_prompt']}')")
            
            result = self.model.transcribe(audio_np_float32, **transcribe_options)
            text = result["text"]
            detected_language = result.get('language', 'unknown')
            logger.debug(f"WhisperASRService raw output: '{text[:50]}...', Detected language: {detected_language}")


            # 仅当检测到的语言是中文（或未指定语言但OpenCC可用）时，才进行转换
            # 并且 self.language 也应为中文相关
            should_convert = False
            if self.opencc_converter and text:
                if self.language == 'zh': # 如果我们强制了中文
                    if detected_language == 'zh' or detected_language == 'chinese': # 并且whisper也认为是中文
                        should_convert = True
                    elif detected_language != 'en': # 如果不是英文，也尝试转换（可能是其他中文方言被错误标记）
                        logger.info(f"Whisper detected language '{detected_language}', but forced language is 'zh'. Attempting OpenCC conversion.")
                        should_convert = True
                # else: 如果 self.language 不是 'zh', 则不进行中文特有的转换

            if should_convert:
                converted_text = self.opencc_converter.convert(text)
                logger.debug(f"WhisperASRService transcription (converted): '{converted_text[:30]}...'")
                return converted_text.strip()
            else:
                # 如果不需要转换（例如，语言不是中文，或者已经是简体），直接返回
                return text.strip()

        except Exception as e:
            logger.error(f"WhisperASRService: Error during transcribe_bytes: {e}", exc_info=True)
            return ""

class ASRSession: # (保持不变)
    def __init__(self, client_uuid: str, asr_service_instance: IASRService, on_result_callback, language: str = None):
        self.client_uuid = client_uuid
        self.asr_service = asr_service_instance
        self.on_result_callback = on_result_callback
        self.language = language.lower() if language else "zh" # 默认为中文
        logger.info(f"ASRSession for {client_uuid} initialized (language hint: {self.language}).")

    def process_accumulated_audio(self, accumulated_audio_bytes: bytes, sample_rate: int):
        if not accumulated_audio_bytes:
            return
        logger.info(f"ASRSession [{self.client_uuid}]: processing {len(accumulated_audio_bytes)} bytes via transcribe_bytes (final segment).")
        text = self.asr_service.transcribe_bytes(accumulated_audio_bytes, sample_rate)
        if text:
            self.on_result_callback(self.client_uuid, text, True)

    def close(self):
        logger.info(f"ASRSession for {self.client_uuid} closed.")