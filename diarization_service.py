# diarization_service.py
import torch
from pyannote.audio import Pipeline
from pyannote.audio import Model as PyannoteModel
import torchaudio
import torchaudio.functional as F
import logging
import os
import numpy as np
from typing import List, Tuple, Dict, Optional # 添加 Optional
import re # 用于解析文件名

# 尝试导入 pydub，如果失败则记录错误，因为保存MP3功能将不可用
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Pydub库未找到。将无法保存MP3格式的说话人音频样本。请运行 'pip install pydub'。"
        "另外，请确保ffmpeg已安装并添加到系统PATH。"
    )

logger = logging.getLogger(__name__)

class DiarizationService:
    def __init__(self,
                 auth_token: Optional[str] = None,
                 device_preference: str = "cuda:0",
                 diarization_model_name: str = "pyannote/speaker-diarization-3.1",
                 embedding_model_name: str = "pyannote/embedding",
                 user_similarity_threshold: float = 0.7,
                 speaker_database_path: str = "db/historical_speakers.pt",
                 speaker_audio_samples_path: str = "db/speaker_audio_samples", # 新增：MP3样本路径
                 new_speaker_similarity_threshold: float = 0.65,
                 min_segment_duration_for_embedding: float = 0.5): # 新增：最短片段时长
        self.pipeline_instance = None
        self.embedding_model_instance = None
        self.user_embedding: Optional[torch.Tensor] = None
        self.auth_token = auth_token
        self.device: Optional[torch.device] = None
        self.device_preference = device_preference
        self.diarization_model_name = diarization_model_name
        self.embedding_model_name = embedding_model_name
        
        self.user_similarity_threshold = user_similarity_threshold
        self.speaker_database_path = speaker_database_path
        self.speaker_audio_samples_path = speaker_audio_samples_path # 新增
        self.new_speaker_similarity_threshold = new_speaker_similarity_threshold
        self.min_segment_duration_for_embedding = min_segment_duration_for_embedding # 新增
        
        self.speaker_database: Dict[str, torch.Tensor] = {}
        self.next_speaker_id_counter = 1

        logger.info(f"DiarizationService: 初始化。Diarization模型: {self.diarization_model_name}, Embedding模型: {self.embedding_model_name}")
        logger.info(f"DiarizationService: 用户识别阈值: {self.user_similarity_threshold}")
        logger.info(f"DiarizationService: 历史说话人数据库路径: {self.speaker_database_path}")
        logger.info(f"DiarizationService: 说话人MP3样本路径: {self.speaker_audio_samples_path}")
        logger.info(f"DiarizationService: 新说话人/历史库匹配阈值: {self.new_speaker_similarity_threshold}")
        logger.info(f"DiarizationService: 提取声纹最短片段时长: {self.min_segment_duration_for_embedding}s")

        # 创建必要的目录
        for path in [os.path.dirname(self.speaker_database_path), self.speaker_audio_samples_path]:
            if path and not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"为说话人数据创建了目录: {path}")
                except Exception as e:
                    logger.error(f"创建目录 {path} 失败: {e}")

    def _save_segment_as_mp3(self, audio_bytes: bytes, sample_rate: int, filepath: str):
        """将PCM音频字节保存为MP3文件"""
        if not PYDUB_AVAILABLE:
            logger.warning(f"Pydub不可用，无法将音频保存为MP3: {filepath}")
            return False
        try:
            # pydub期望音频数据是原始字节，参数包括采样率、样本宽度（字节）、通道数
            # 假设我们的audio_bytes是16-bit PCM, 单通道
            audio_segment = AudioSegment(
                data=audio_bytes,
                sample_width=2, # 16-bit = 2 bytes
                frame_rate=sample_rate,
                channels=1
            )
            # 确保输出目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            audio_segment.export(filepath, format="mp3")
            logger.info(f"新说话人音频样本已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存MP3文件 {filepath} 失败: {e}", exc_info=True)
            return False

    def _parse_speaker_filename(self, filename_without_ext: str) -> Tuple[Optional[str], str]:
        """
        从文件名（不含扩展名）解析原始personID和最终标签名。
        支持 "personXX-CustomName" 或 "personXX"。
        如果完全自定义如 "Alice"，则 original_person_id 为 None。
        返回 (original_person_id, final_label_from_filename)
        """
        match = re.fullmatch(r"(person\d{2,})(-(.+))?", filename_without_ext)
        if match:
            original_person_id = match.group(1)  # e.g., "person01"
            custom_part = match.group(3)        # e.g., "Alice" or None
            if custom_part:
                return original_person_id, custom_part
            return original_person_id, original_person_id # No custom part, final is "personXX"
        return None, filename_without_ext # Not "personXX" or "personXX-Custom" format

    def _load_speaker_database(self):
        """
        加载历史说话人数据库 (.pt) 并与MP3样本文件名同步标签。
        """
        if not self.speaker_database_path:
            logger.warning("历史说话人数据库路径未配置，使用空数据库。")
            self.speaker_database = {}
            self.next_speaker_id_counter = 1
            return

        old_db_embeddings: Dict[str, torch.Tensor] = {}
        if os.path.exists(self.speaker_database_path):
            try:
                old_db_embeddings = torch.load(self.speaker_database_path, map_location=self.device)
                # 确保所有加载的嵌入都在正确的设备上
                for spk_id in list(old_db_embeddings.keys()):
                    old_db_embeddings[spk_id] = old_db_embeddings[spk_id].to(self.device)
                logger.info(f"从 {self.speaker_database_path} 初始加载了 {len(old_db_embeddings)} 个声纹。")
            except Exception as e:
                logger.error(f"从 {self.speaker_database_path} 加载历史说话人数据库失败: {e}", exc_info=True)
        else:
            logger.info(f"在 {self.speaker_database_path} 未找到历史说话人数据库文件。")

        # 与MP3文件名同步，构建新的数据库
        updated_speaker_database: Dict[str, torch.Tensor] = {}
        processed_original_ids_from_pt: set[str] = set()

        if os.path.exists(self.speaker_audio_samples_path):
            for filename in os.listdir(self.speaker_audio_samples_path):
                if filename.lower().endswith(".mp3"):
                    filename_no_ext = os.path.splitext(filename)[0]
                    original_person_id, final_label = self._parse_speaker_filename(filename_no_ext)

                    if original_person_id and original_person_id in old_db_embeddings:
                        # MP3文件名指示了一个原始personID，并且该ID在.pt文件中有对应声纹
                        updated_speaker_database[final_label] = old_db_embeddings[original_person_id]
                        processed_original_ids_from_pt.add(original_person_id)
                        logger.debug(f"标签同步：MP3 '{filename}' -> 原始ID '{original_person_id}' 映射到最终标签 '{final_label}'.")
                    elif final_label in old_db_embeddings:
                        # MP3文件名是自定义名，且此自定义名恰好在.pt文件中有对应声纹 (可能是之前重命名的结果)
                        updated_speaker_database[final_label] = old_db_embeddings[final_label]
                        processed_original_ids_from_pt.add(final_label) # 假设这个final_label如果是personXX格式，也标记处理
                        logger.debug(f"标签同步：MP3 '{filename}' (自定义名 '{final_label}') 在 .pt 文件中找到对应声纹.")
                    else:
                        logger.warning(f"标签同步：MP3文件 '{filename}' (解析为标签 '{final_label}', 原始ID '{original_person_id}') 在.pt数据库中找不到对应声纹。可能需要手动清理或这是一个全新的、未记录的样本。")
        else:
            logger.info(f"说话人MP3样本目录 '{self.speaker_audio_samples_path}' 不存在，跳过MP3文件名同步。")
        
        # 添加.pt中存在但没有对应MP3文件（或MP3文件名不规范未被处理）的条目
        for old_id, embedding in old_db_embeddings.items():
            if old_id not in processed_original_ids_from_pt and old_id not in updated_speaker_database:
                # 确保不覆盖已通过MP3文件名正确映射的条目
                # 如果一个old_id (如person01) 的mp3被重命名为Alice.mp3, 那么old_id (person01) 不在processed_original_ids_from_pt
                # 且 old_id (person01) 也不在 updated_speaker_database的键中 (键现在是Alice)
                # 这种情况，如果Alice.mp3的original_id是person01，我们已经处理了。
                # 这里是处理那些在.pt里，但完全没有对应MP3，或者MP3名完全不相关的。
                # 比如.pt里有 person05, 但没有person05.mp3或person05-xxx.mp3。
                is_potentially_renamed_and_processed = False
                for processed_orig_id in processed_original_ids_from_pt:
                    if old_db_embeddings.get(processed_orig_id) is embedding: # 检查是否是同一个声纹对象
                        is_potentially_renamed_and_processed = True
                        break
                if not is_potentially_renamed_and_processed:
                    updated_speaker_database[old_id] = embedding
                    logger.debug(f"标签同步：保留来自 .pt 的条目 '{old_id}' (无匹配MP3或MP3命名不规范)。")


        self.speaker_database = updated_speaker_database
        
        # 更新 next_speaker_id_counter
        max_id_num = 0
        if self.speaker_database:
            for spk_id in self.speaker_database.keys():
                # 不仅从键更新，还要考虑文件名解析出的原始ID
                match_id = re.match(r"person(\d{2,})", spk_id) # 检查最终标签是否是personXX格式
                if match_id:
                    try:
                        max_id_num = max(max_id_num, int(match_id.group(1)))
                    except ValueError:
                        pass # 忽略无法解析的
            # 也检查一下MP3文件名中的personID，以防.pt文件损坏或滞后
            if os.path.exists(self.speaker_audio_samples_path):
                 for filename in os.listdir(self.speaker_audio_samples_path):
                    if filename.lower().endswith(".mp3"):
                        filename_no_ext = os.path.splitext(filename)[0]
                        parsed_orig_id, _ = self._parse_speaker_filename(filename_no_ext)
                        if parsed_orig_id: # personXX or personXX-custom
                            match_id_from_file = re.match(r"person(\d{2,})", parsed_orig_id)
                            if match_id_from_file:
                                try:
                                    max_id_num = max(max_id_num, int(match_id_from_file.group(1)))
                                except ValueError:
                                    pass
        self.next_speaker_id_counter = max_id_num + 1
        
        logger.info(f"说话人数据库同步完成。最终数据库含 {len(self.speaker_database)} 个说话人。下一个说话人ID计数器: {self.next_speaker_id_counter}")

    def _save_speaker_database(self):
        if not self.speaker_database_path:
            logger.warning("历史说话人数据库路径未配置，无法保存。")
            return
        if not self.speaker_database:
             logger.info("历史说话人数据库为空，不执行保存操作。")
             # 为了确保下次加载时能正确处理空的 .pt 文件
             try:
                 torch.save({}, self.speaker_database_path)
             except Exception as e:
                 logger.error(f"保存空的说话人数据库至 {self.speaker_database_path} 失败: {e}", exc_info=True)
             return

        try:
            cpu_speaker_database = {k: v.cpu() for k, v in self.speaker_database.items()}
            torch.save(cpu_speaker_database, self.speaker_database_path)
            logger.info(f"历史说话人数据库已保存至 {self.speaker_database_path}，包含 {len(self.speaker_database)} 个说话人。")
        except Exception as e:
            logger.error(f"保存历史说话人数据库至 {self.speaker_database_path} 失败: {e}", exc_info=True)

    def load_model(self):
        try:
            if "cuda" in self.device_preference and torch.cuda.is_available():
                self.device = torch.device(self.device_preference)
                logger.info(f"DiarizationService: 尝试使用 CUDA 设备: {self.device}")
            else:
                self.device = torch.device("cpu")
                logger.info("DiarizationService: CUDA 不可用或未选择，使用 CPU。")

            effective_auth_token = self.auth_token if self.auth_token else (os.environ.get("HF_TOKEN") or True)
            
            logger.info(f"DiarizationService: 尝试加载 diarization pipeline '{self.diarization_model_name}'...")
            temp_pipeline = Pipeline.from_pretrained(self.diarization_model_name, use_auth_token=effective_auth_token)
            if temp_pipeline is None:
                logger.error(f"DiarizationService: Diarization pipeline '{self.diarization_model_name}' loading failed.")
                self.pipeline_instance = None; return
            self.pipeline_instance = temp_pipeline.to(self.device)
            logger.info(f"DiarizationService: Diarization pipeline '{self.diarization_model_name}' loaded successfully to {self.device}.")

            logger.info(f"DiarizationService: 尝试加载 embedding model '{self.embedding_model_name}'...")
            try:
                temp_embedding_model = PyannoteModel.from_pretrained(self.embedding_model_name, use_auth_token=effective_auth_token)
                if temp_embedding_model is None:
                    logger.error(f"DiarizationService: Embedding model '{self.embedding_model_name}' loading failed. User ID disabled.")
                    self.embedding_model_instance = None
                else:
                    self.embedding_model_instance = temp_embedding_model.to(self.device)
                    self.embedding_model_instance.eval()
                    logger.info(f"DiarizationService: Embedding model '{self.embedding_model_name}' loaded to {self.device}.")
            except Exception as e_emb_load:
                logger.error(f"DiarizationService: Failed to load embedding model '{self.embedding_model_name}': {e_emb_load}", exc_info=True)
                self.embedding_model_instance = None
                logger.warning("User identification will be disabled due to embedding model loading failure.")

            self._load_speaker_database() # 在模型加载后同步数据库

        except Exception as e:
            logger.error(f"DiarizationService: An unexpected error occurred while loading models: {e}", exc_info=True)
            self.pipeline_instance = None
            self.embedding_model_instance = None

    def unload_model(self):
        self._save_speaker_database() # 保存更新后的数据库
        
        if self.pipeline_instance is not None: del self.pipeline_instance; self.pipeline_instance = None
        if self.embedding_model_instance is not None: del self.embedding_model_instance; self.embedding_model_instance = None
        self.user_embedding = None
        if self.device and hasattr(self.device, 'type') and self.device.type == 'cuda': torch.cuda.empty_cache()
        logger.info("DiarizationService: 模型及相关资源已卸载。历史说话人数据库已尝试保存。")

    def register_user_from_file(self, audio_file_path: str):
        if not self.embedding_model_instance:
            logger.error("DiarizationService: Embedding model not available. Cannot register user voice.")
            return
        if not os.path.exists(audio_file_path):
            logger.error(f"DiarizationService: User audio sample file not found: {audio_file_path}")
            return
        try:
            target_sr = 16000 # diarization pipeline 和 embedding model 通常期望16kHz
            waveform, sr = torchaudio.load(audio_file_path)
            duration_sec = waveform.shape[-1] / sr
            logger.info(f"DiarizationService: Registering user voice from {audio_file_path} (Duration: {duration_sec:.2f}s, Original SR: {sr}Hz, Target SR: {target_sr}Hz for embedding)")

            waveform = waveform.to(self.device) # 移到设备
            if sr != target_sr: waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0) # (samples) -> (1, samples)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) # 多通道转单通道
            
            # Embedding模型通常期望 (batch, channels, samples) 或 (batch, samples)
            # pyannote.audio.Model期望 (batch_size, num_channels, num_samples)
            # 如果当前是 (1, samples), 需要变为 (1, 1, samples)
            if waveform.ndim == 2 and waveform.shape[0] == 1: # (1, samples)
                waveform_3d = waveform.unsqueeze(1) # (1, 1, samples)
            elif waveform.ndim == 1: # 理论上不应该到这里，但做个防护
                 waveform_3d = waveform.unsqueeze(0).unsqueeze(0) # (samples) -> (1,1,samples)
            else: # 如果已经是3D (batch, channels, samples)，直接用
                waveform_3d = waveform

            with torch.no_grad():
                embedding_batch = self.embedding_model_instance(waveform_3d) # (batch, embed_dim)
            
            # embedding_batch 可能是 (1, D) 或 (N, D) 如果模型内部处理了帧
            # 我们需要一个单一的嵌入向量 (D,)
            if embedding_batch.ndim > 1 and embedding_batch.shape[0] == 1:
                embedding = embedding_batch.squeeze(0) # (1, D) -> (D,)
            else: # 如果是 (N, D) for N > 1, 取均值
                embedding = torch.mean(embedding_batch, dim=0) if embedding_batch.ndim > 1 else embedding_batch

            self.user_embedding = torch.nn.functional.normalize(embedding, p=2, dim=0) # L2归一化
            logger.info(f"DiarizationService: User voice registered successfully. Embedding shape: {self.user_embedding.shape}")
        except Exception as e:
            logger.error(f"DiarizationService: Failed to register user voice from {audio_file_path}: {e}", exc_info=True)
            self.user_embedding = None

    def process_audio(self, audio_bytes: bytes, input_sample_rate: int) -> List[Tuple[str, str, bytes, int]]:
        if not self.pipeline_instance or not self.embedding_model_instance:
            logger.warning("DiarizationService: Diarization pipeline 或 embedding model 未加载。无法处理音频。")
            return []
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            input_waveform_full = torch.from_numpy(audio_np).unsqueeze(0) # (1, samples)
            target_sr = 16000
            processed_waveform = input_waveform_full
            if input_sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=target_sr)
                processed_waveform = resampler(input_waveform_full)
            
            processed_waveform = processed_waveform.to(self.device)
            audio_input_for_pipeline = {"waveform": processed_waveform, "sample_rate": target_sr}
            
            diarization_result = self.pipeline_instance(audio_input_for_pipeline)
            identified_segments_data: List[Tuple[str, str, bytes, int]] = []
            if diarization_result is None:
                logger.warning("DiarizationService: Diarization pipeline returned None. No segments found.")
                return []

            # 新增：用于缓存已处理的原始pyannote标签及其最终识别结果
            pyannote_label_to_final_label_map: Dict[str, str] = {}

            for turn_segment, track_idx, speaker_label_orig_pyannote in diarization_result.itertracks(yield_label=True):
                final_speaker_label_for_segment = speaker_label_orig_pyannote # 默认
                
                # --- 优化：检查此原始pyannote标签是否已处理 ---
                if speaker_label_orig_pyannote in pyannote_label_to_final_label_map:
                    final_speaker_label_for_segment = pyannote_label_to_final_label_map[speaker_label_orig_pyannote]
                    logger.debug(f"片段 '{speaker_label_orig_pyannote}' 已被处理，复用最终标签 '{final_speaker_label_for_segment}'.")
                else:
                    # --- 此原始标签首次出现，执行完整识别流程 ---
                    start_sample = int(turn_segment.start * target_sr)
                    end_sample = int(turn_segment.end * target_sr)
                    segment_duration_sec = turn_segment.duration

                    # (1, samples) -> (samples) -> (1, samples) for embedding model if needed
                    segment_waveform_for_embedding_extraction = processed_waveform[0, start_sample:end_sample]
                    if segment_waveform_for_embedding_extraction.ndim == 1:
                        segment_waveform_for_embedding_extraction = segment_waveform_for_embedding_extraction.unsqueeze(0) # (1, samples)

                    current_segment_embedding_normalized: Optional[torch.Tensor] = None
                    if segment_waveform_for_embedding_extraction.numel() > 0 and \
                       segment_duration_sec >= self.min_segment_duration_for_embedding:
                        
                        # (1, samples) -> (1, 1, samples) for pyannote.audio.Model
                        segment_for_embedding_input_3d = segment_waveform_for_embedding_extraction.unsqueeze(1)
                        try:
                            with torch.no_grad():
                                current_segment_embedding_batch = self.embedding_model_instance(segment_for_embedding_input_3d)
                            
                            # 平均嵌入 (batch, D) -> (D)
                            current_segment_embedding_vec = torch.mean(current_segment_embedding_batch, dim=0) \
                                if current_segment_embedding_batch.ndim > 1 and current_segment_embedding_batch.shape[0] > 1 \
                                else current_segment_embedding_batch.squeeze(0)
                            
                            current_segment_embedding_normalized = torch.nn.functional.normalize(current_segment_embedding_vec, p=2, dim=0)
                        except Exception as e_emb_extract:
                            logger.warning(f"为片段 {speaker_label_orig_pyannote} 提取声纹嵌入失败: {e_emb_extract}", exc_info=False)
                    
                    if current_segment_embedding_normalized is not None:
                        # 1. 检查是否为已注册的 "user"
                        if self.user_embedding is not None:
                            user_similarity = torch.dot(self.user_embedding, current_segment_embedding_normalized)
                            if user_similarity.item() > self.user_similarity_threshold:
                                final_speaker_label_for_segment = "user"
                        
                        # 2. 如果不是 "user"，则检查历史数据库
                        if final_speaker_label_for_segment == speaker_label_orig_pyannote: # 未被识别为 "user"
                            best_historical_match_id: Optional[str] = None
                            highest_historical_similarity = -1.0
                            for hist_id, hist_embedding in self.speaker_database.items():
                                hist_similarity = torch.dot(hist_embedding, current_segment_embedding_normalized)
                                if hist_similarity.item() > highest_historical_similarity:
                                    highest_historical_similarity = hist_similarity.item()
                                    best_historical_match_id = hist_id
                            
                            if best_historical_match_id and highest_historical_similarity > self.new_speaker_similarity_threshold:
                                final_speaker_label_for_segment = best_historical_match_id
                            else: # 3. 新说话人
                                new_speaker_id = f"person{self.next_speaker_id_counter:02d}"
                                self.speaker_database[new_speaker_id] = current_segment_embedding_normalized.clone()
                                final_speaker_label_for_segment = new_speaker_id
                                self.next_speaker_id_counter += 1
                                logger.info(f"新说话人注册: Pyannote标签 '{speaker_label_orig_pyannote}' 被注册为 '{new_speaker_id}'.")
                                
                                # 保存该新说话人的第一个片段音频为MP3
                                # segment_waveform_float_tensor_for_return is already (num_samples) on device
                                temp_segment_bytes = (segment_waveform_for_embedding_extraction.squeeze(0).cpu().numpy() * 32767).astype(np.int16).tobytes()
                                mp3_filepath = os.path.join(self.speaker_audio_samples_path, f"{new_speaker_id}.mp3")
                                self._save_segment_as_mp3(temp_segment_bytes, target_sr, mp3_filepath)
                    else: # 无法提取声纹，保留原始pyannote标签
                        final_speaker_label_for_segment = speaker_label_orig_pyannote
                        logger.warning(f"无法为片段 '{speaker_label_orig_pyannote}' (时长 {segment_duration_sec:.2f}s) 提取有效声纹，使用原始标签。")

                    # 缓存此原始pyannote标签的最终识别结果
                    pyannote_label_to_final_label_map[speaker_label_orig_pyannote] = final_speaker_label_for_segment
                # --- 完整识别流程结束 ---

                # 提取当前片段的音频数据用于返回 (总是执行，即使标签是复用的)
                seg_start_sample = int(turn_segment.start * target_sr)
                seg_end_sample = int(turn_segment.end * target_sr)
                segment_waveform_for_return = processed_waveform[0, seg_start_sample:seg_end_sample] 
                segment_bytes_for_return = (segment_waveform_for_return.cpu().numpy() * 32767).astype(np.int16).tobytes()
                
                identified_segments_data.append((final_speaker_label_for_segment, speaker_label_orig_pyannote, segment_bytes_for_return, target_sr))
            
            return identified_segments_data
        except Exception as e:
            logger.error(f"DiarizationService: 处理音频时发生错误: {e}", exc_info=True)
            return []