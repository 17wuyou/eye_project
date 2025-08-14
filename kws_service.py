# kws_service.py
import logging
import struct
from typing import List, Optional

import pvporcupine

logger = logging.getLogger(__name__)

class KwsService:
    """
    一个简单的 pvporcupine 服务封装类，用于关键词唤醒检测。
    """
    def __init__(self, access_key: str, model_path: str, keyword_paths: List[str], sensitivities: List[float]):
        """
        初始化 Porcupine KWS 引擎。

        :param access_key: PicoVoice AccessKey.
        :param model_path: Porcupine 模型文件路径 (.pv).
        :param keyword_paths: 关键词模型文件路径列表 (.ppn).
        :param sensitivities: 对应关键词的灵敏度列表 (0.0 to 1.0).
        """
        self._porcupine = None
        self._access_key = access_key
        self._model_path = model_path
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities
        self.sample_rate = None
        self.frame_length = None

        try:
            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                model_path=self._model_path,
                keyword_paths=self._keyword_paths,
                sensitivities=self._sensitivities
            )
            self.sample_rate = self._porcupine.sample_rate
            self.frame_length = self._porcupine.frame_length
            logger.info(f"Porcupine KWS 引擎已成功初始化。采样率: {self.sample_rate}, 帧长度: {self.frame_length}")
        except pvporcupine.PorcupineError as e:
            logger.error(f"初始化 Porcupine 失败: {e}", exc_info=True)
            self._porcupine = None
            raise

    def process(self, pcm_chunk: bytes) -> bool:
        """
        处理一个音频块以检测关键词。

        :param pcm_chunk: 16-bit little-endian PCM 音频块。
        :return: 如果检测到关键词则返回 True，否则返回 False。
        """
        if not self._porcupine or not pcm_chunk:
            return False

        try:
            # 将字节数据解包为 int16 样本
            pcm = struct.unpack_from("h" * (len(pcm_chunk) // 2), pcm_chunk)
            
            # Porcupine 每次处理一个固定长度的帧
            for i in range(0, len(pcm), self.frame_length):
                frame = pcm[i:i + self.frame_length]
                if len(frame) == self.frame_length:
                    result = self._porcupine.process(frame)
                    if result >= 0:
                        logger.info(f"检测到关键词 (索引: {result})!")
                        return True
            return False
        except Exception as e:
            logger.error(f"处理 KWS 音频帧时出错: {e}", exc_info=True)
            return False

    def delete(self):
        """
        释放 Porcupine 引擎资源。
        """
        if self._porcupine:
            self._porcupine.delete()
            logger.info("Porcupine KWS 引擎资源已释放。")