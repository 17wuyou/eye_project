# face_service.py
import os
import logging
import json
import re
import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self,
                 db_path: str,
                 recognition_threshold: float = 0.5,
                 detection_confidence: float = 0.5):
        
        logger.info("正在初始化 FaceService (使用 InsightFace)...")

        self.db_path = db_path
        self.features_path = os.path.join(db_path, "features")
        self.images_path = os.path.join(db_path, "images")
        self.metadata_path = os.path.join(db_path, "metadata.jsonl")

        self.recognition_threshold = recognition_threshold
        self.detection_confidence = detection_confidence

        # --- 智能GPU选择逻辑 ---
        gpu_id = -1 # 默认为 CPU
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            if num_devices > 1:
                logger.info(f"检测到 {num_devices} 个CUDA设备。将优先使用 cuda:1。")
                gpu_id = 1
            elif num_devices == 1:
                logger.info("检测到 1 个CUDA设备。将使用 cuda:0。")
                gpu_id = 0
            else: # 不太可能，但作为健壮性检查
                logger.warning("torch.cuda.is_available() is True, 但未找到设备。回退至CPU。")
        else:
            logger.info("未检测到可用的CUDA设备。将使用CPU。")

        # --- 初始化 InsightFace 分析器 ---
        try:
            # 使用 buffalo_l 模型包，它会自动下载并包含检测和识别模型
            self.face_analyzer: FaceAnalysis = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                provider_options=[{'device_id': gpu_id}, {}]
            )
            # 准备模型，设置检测阈值
            self.face_analyzer.prepare(ctx_id=gpu_id, det_thresh=self.detection_confidence)
            logger.info(f"InsightFace 'buffalo_l' 模型包已在设备 (ctx_id={gpu_id}) 上准备就绪。")
        except Exception as e:
            logger.critical(f"初始化 InsightFace 分析器失败: {e}", exc_info=True)
            raise

        self.known_faces: List[Dict] = []
        self.next_unknown_id: int = 1

        self._initialize_database()
        logger.info(f"FaceService 初始化完成。已加载 {len(self.known_faces)} 个已知人脸。下一个未知ID: {self.next_unknown_id}")
    
    def _initialize_database(self):
        """加载或创建人脸数据库目录"""
        os.makedirs(self.features_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        self._load_database()

    def _load_database(self):
        """从元数据文件加载已知人脸信息，并根据图片文件名同步人名。"""
        max_id = 0
        known_faces_temp = []
        if not os.path.exists(self.metadata_path):
            logger.info(f"在 {self.metadata_path} 未找到元数据文件。从空数据库开始。")
            self.known_faces = []
            self.next_unknown_id = 1
            return

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    feature_path = os.path.join(self.db_path, data["feature_path"])
                    image_path_from_meta = os.path.join(self.db_path, data["image_path"])
                    
                    if not os.path.exists(feature_path) or not os.path.exists(image_path_from_meta):
                        logger.warning(f"跳过条目 {data['id']}，因为特征或图像文件缺失。")
                        continue

                    # 从文件名解析名字，确保与文件系统同步
                    image_filename = os.path.basename(image_path_from_meta)
                    base_name, _ = os.path.splitext(image_filename)
                    parts = base_name.split('_')
                    name = "_".join(parts[2:]) if len(parts) > 2 else data.get("name", f"UnKnown{int(data['id'].split('_')[-1]):02d}")

                    embedding = np.load(feature_path)
                    known_faces_temp.append({
                        "id": data["id"], "name": name, "embedding": embedding
                    })

                    match_id = re.search(r"person_(\d+)", data["id"])
                    if match_id:
                        max_id = max(max_id, int(match_id.group(1)))

                except (json.JSONDecodeError, KeyError, IndexError, FileNotFoundError) as e:
                    logger.error(f"解析元数据/文件名或加载文件时出错: '{line.strip()}'. 错误: {e}")
        
        self.known_faces = known_faces_temp
        self.next_unknown_id = max_id + 1

    def _expand_bbox(self, bbox: np.ndarray, scale_factor: float, img_shape: Tuple[int, int]) -> np.ndarray:
        """按比例扩大边界框，并确保其不越界。"""
        img_h, img_w = img_shape
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        
        center_x, center_y = x1 + w // 2, y1 + h // 2
        
        new_w = int(w * (1 + scale_factor))
        new_h = int(h * (1 + scale_factor))
        
        new_x1 = max(0, center_x - new_w // 2)
        new_y1 = max(0, center_y - new_h // 2)
        new_x2 = min(img_w, center_x + new_w // 2)
        new_y2 = min(img_h, center_y + new_h // 2)
        
        return np.array([new_x1, new_y1, new_x2, new_y2]).astype(int)

    def _save_new_face(self, embedding: np.ndarray, original_bgr_image: np.ndarray, face: 'Face') -> str:
        """保存一个新的人脸到数据库，使用扩大后的边界框裁剪图像。"""
        new_id_num = self.next_unknown_id
        new_id_str = f"person_{new_id_num:03d}"
        new_name = f"UnKnown{new_id_num:02d}"

        # 保存特征向量
        feature_filename = f"{new_id_str}.npy"
        feature_rel_path = os.path.join("features", feature_filename)
        np.save(os.path.join(self.db_path, feature_rel_path), embedding)
        
        # 步骤C & D: 扩大边界框，裁剪图像并保存
        expanded_bbox = self._expand_bbox(face.bbox, 0.15, original_bgr_image.shape[:2])
        ex_x1, ex_y1, ex_x2, ex_y2 = expanded_bbox
        expanded_face_image = original_bgr_image[ex_y1:ex_y2, ex_x1:ex_x2]
        
        image_filename = f"{new_id_str}_{new_name}.jpg"
        image_rel_path = os.path.join("images", image_filename)
        cv2.imwrite(os.path.join(self.db_path, image_rel_path), expanded_face_image)

        # 保存元数据
        new_metadata = {
            "id": new_id_str, "name": new_name,
            "feature_path": feature_rel_path.replace("\\", "/"),
            "image_path": image_rel_path.replace("\\", "/")
        }
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_metadata, ensure_ascii=False) + "\n")
        
        self.known_faces.append({
            "id": new_id_str, "name": new_name, "embedding": embedding
        })
        
        self.next_unknown_id += 1
        logger.info(f"新的人脸已注册: {new_name} (ID: {new_id_str})")
        return new_name

    def _find_match(self, face_embedding: np.ndarray, original_bgr_image: np.ndarray, face: 'Face') -> str:
        """在已知人脸中寻找匹配项，或创建新条目。"""
        if not self.known_faces:
            return self._save_new_face(face_embedding, original_bgr_image, face)

        max_sim = -1.0
        best_match_name = None
        
        # InsightFace 的 normed_embedding 已经 L2 标准化，余弦相似度等价于点积
        for known_face in self.known_faces:
            similarity = np.dot(face_embedding, known_face["embedding"])
            if similarity > max_sim:
                max_sim = similarity
                best_match_name = known_face["name"]

        if max_sim > self.recognition_threshold:
            logger.debug(f"匹配到已知人脸: {best_match_name} (相似度: {max_sim:.4f})")
            return best_match_name
        else:
            logger.debug(f"未找到足够相似的人脸 (最高相似度: {max_sim:.4f})。注册为新面孔。")
            return self._save_new_face(face_embedding, original_bgr_image, face)

    def process_frame(self, frame_bytes: bytes) -> List[Dict[str, Any]]:
        """处理单帧图像，返回检测到的人脸信息列表。"""
        try:
            # 1. 解码图像
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("从字节码解码图像失败。")
                return []
            
            # InsightFace 的 'get' 方法期望 BGR 格式图像
            
            # 2. 使用 InsightFace 进行检测和识别
            # .get() 一次性完成检测、对齐、特征提取
            faces = self.face_analyzer.get(img_bgr)
            
            if not faces:
                return []

            results = []
            # 3. 遍历检测到的每个人脸
            for face in faces:
                # 步骤B (由SDK完成): 'face.normed_embedding' 即为提取的特征向量
                embedding = face.normed_embedding
                
                # 4. 匹配或注册新面孔
                name = self._find_match(embedding, img_bgr, face)
                
                # 5. 准备返回结果
                results.append({
                    "name": name,
                    # 返回边界框可用于未来在客户端精确绘图
                    "bbox": face.bbox.astype(int).tolist() 
                })
            
            return results
        except Exception as e:
            logger.error(f"人脸处理主循环中发生意外错误: {e}", exc_info=True)
            return []