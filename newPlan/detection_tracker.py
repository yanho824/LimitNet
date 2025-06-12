# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
try:
    from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
    print("✅ ByteTrack 可用")
except ImportError:
    print("⚠️ Warning: ByteTrack not available, using simple tracking fallback")
    BYTETRACK_AVAILABLE = False
    BYTETracker = None

import supervision as sv
from typing import List, Tuple, Optional
import torchvision.transforms as transforms


class SimpleTracker:
    """简单的追踪器回退方案"""
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        
    def update(self, detections, frame):
        """更新追踪"""
        if len(detections) == 0:
            return np.empty((0, 8))
        
        tracks = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # 简单分配ID（实际应用中应该用更复杂的匹配算法）
            track_id = self.next_id
            self.next_id += 1
            
            # 格式: [x1, y1, x2, y2, track_id, conf, cls, frame_id]
            track = [x1, y1, x2, y2, track_id, conf, cls, 0]
            tracks.append(track)
        
        return np.array(tracks)
    
    def reset(self):
        """重置追踪器"""
        self.next_id = 0
        self.tracks = {}


class DetectionTracker:
    """
    YOLOv11n + ByteTrack 检测追踪模块
    用于替代原来的显著性检测，生成高优先级传输区域
    """
    
    def __init__(self, 
                 model_path: str = 'yolo11n.pt',
                 tracker_method: str = 'bytetrack',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda',
                 frame_rate: int = 30):
        """
        初始化检测追踪器
        
        Args:
            model_path: YOLOv11模型路径
            tracker_method: 追踪方法 (目前主要使用'bytetrack')
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IOU阈值
            device: 计算设备
            frame_rate: 视频帧率，用于ByteTrack初始化
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_rate = frame_rate
        
        # 初始化YOLOv11检测器
        self.detector = YOLO(model_path)
        print(f"✅ 加载YOLO模型: {model_path}")
        
        # 初始化ByteTrack追踪器
        if BYTETRACK_AVAILABLE and BYTETracker is not None:
            try:
                # 配置ByteTrack参数
                self.tracker = BYTETracker(
                    track_thresh=0.5,        # 追踪阈值
                    track_buffer=30,         # 追踪缓冲帧数
                    match_thresh=0.8,        # 匹配阈值
                    frame_rate=frame_rate    # 帧率
                )
                print("✅ 使用ByteTrack追踪器")
            except Exception as e:
                print(f"❌ ByteTrack初始化失败: {e}, 使用简单追踪器")
                self.tracker = SimpleTracker()
        else:
            print("❌ ByteTrack不可用，使用简单追踪器")
            self.tracker = SimpleTracker()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.reset_tracker()
    
    def reset_tracker(self):
        """重置追踪器状态"""
        if isinstance(self.tracker, SimpleTracker):
            self.tracker.reset()
        # ByteTrack不需要reset方法
    
    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        对单帧进行检测和追踪
        
        Args:
            frame: 输入图像 (H, W, 3) BGR格式
            
        Returns:
            priority_mask: 优先级掩码 (H, W) 
            track_info: 追踪信息列表
        """
        # YOLOv11检测
        results = self.detector(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 提取检测结果
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # ByteTrack需要的格式: [x1, y1, x2, y2, score] (不包括class)
            detections = np.column_stack([boxes, scores])
        
        # 追踪更新
        tracks = []
        if len(detections) > 0:
            try:
                if isinstance(self.tracker, SimpleTracker):
                    # SimpleTracker的格式
                    det_with_class = np.column_stack([boxes, scores, classes]) if len(boxes) > 0 else []
                    tracks = self.tracker.update(det_with_class, frame)
                else:
                    # ByteTrack的update方法
                    online_targets = self.tracker.update(
                        detections,
                        [frame.shape[0], frame.shape[1]],
                        [frame.shape[0], frame.shape[1]]
                    )
                    # 转换ByteTrack输出格式
                    for t in online_targets:
                        tlwh = t.tlwh
                        track_id = t.track_id
                        score = t.score
                        # 转换为 [x1, y1, x2, y2, track_id, conf, class, frame_id]
                        track = [
                            tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3],
                            track_id, score, 0, 0
                        ]
                        tracks.append(track)
            except Exception as e:
                print(f"⚠️ 追踪更新失败: {e}, 使用检测结果")
                # 如果追踪失败，直接使用检测结果
                tracks = []
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, score = det
                    # 创建伪追踪结果: [x1, y1, x2, y2, track_id, conf, class, frame_id]
                    track = [x1, y1, x2, y2, i, score, 0, 0]  # 临时ID
                    tracks.append(track)
                tracks = np.array(tracks) if tracks else np.empty((0, 8))
        else:
            tracks = np.empty((0, 8))
        
        # 确保tracks是numpy数组
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks) if len(tracks) > 0 else np.empty((0, 8))
        
        # 生成优先级掩码
        priority_mask = self._generate_priority_mask(frame.shape[:2], tracks)
        
        # 准备追踪信息
        track_info = []
        for track in tracks:
            if len(track) >= 5:  # 至少需要bbox和ID
                track_info.append({
                    'id': int(track[4]) if len(track) > 4 else 0,
                    'bbox': track[:4],  # x1, y1, x2, y2
                    'conf': track[5] if len(track) > 5 else 0.5,
                    'class': int(track[6]) if len(track) > 6 else 0
                })
        
        return priority_mask, track_info
    
    def _generate_priority_mask(self, frame_shape: Tuple[int, int], tracks: np.ndarray) -> np.ndarray:
        """
        根据检测追踪结果生成优先级掩码
        
        Args:
            frame_shape: 图像尺寸 (H, W)
            tracks: 追踪结果
            
        Returns:
            priority_mask: 优先级掩码 (H, W)
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        if len(tracks) == 0:
            return mask
        
        # 为每个追踪目标分配优先级
        for i, track in enumerate(tracks):
            if len(track) >= 6:
                x1, y1, x2, y2 = track[:4].astype(int)
                conf = track[5]
                
                # 确保边界框在图像范围内
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                
                if x2 > x1 and y2 > y1:
                    # 基于置信度和目标大小设置优先级
                    area_ratio = ((x2-x1) * (y2-y1)) / (w * h)
                    priority = conf * (0.8 + 0.2 * min(area_ratio * 10, 1.0))
                    
                    # 在掩码中设置优先级区域
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], priority)
        
        return mask
    
    def process_tensor_batch(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        """
        处理PyTorch张量批次
        
        Args:
            tensor_batch: 输入张量批次 (B, C, H, W)
            
        Returns:
            priority_masks: 优先级掩码张量 (B, 1, H, W)
        """
        batch_size = tensor_batch.shape[0]
        h, w = tensor_batch.shape[2], tensor_batch.shape[3]
        
        priority_masks = torch.zeros((batch_size, 1, h, w), device=tensor_batch.device)
        
        for i in range(batch_size):
            # 转换为numpy格式进行处理
            img_tensor = tensor_batch[i]  # (C, H, W)
            
            # 反归一化并转换为BGR格式
            img_np = self._tensor_to_numpy(img_tensor)
            
            # 检测和追踪
            priority_mask, _ = self.detect_and_track(img_np)
            
            # 转换回张量格式
            mask_tensor = torch.from_numpy(priority_mask).unsqueeze(0)  # (1, H, W)
            mask_tensor = mask_tensor.to(tensor_batch.device)
            
            priority_masks[i] = mask_tensor
        
        return priority_masks
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将PyTorch张量转换为numpy数组用于检测
        
        Args:
            tensor: 输入张量 (C, H, W), 已归一化
            
        Returns:
            numpy数组 (H, W, C), BGR格式, 0-255范围
        """
        # 反归一化 (假设使用ImageNet标准化)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor.device.type == 'cuda':
            mean = mean.cuda()
            std = std.cuda()
        
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        # 转换为numpy并调整通道顺序
        img_np = denormalized.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        img_np = (img_np * 255).astype(np.uint8)
        
        # RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_bgr


class LimitNetDetectionTracker:
    """
    集成ByteTrack检测追踪的LimitNet包装器
    使用YOLOv11n + ByteTrack替代原始的显著性检测
    """
    
    def __init__(self, 
                 detector_config: dict = None,
                 priority_boost: float = 1.5,
                 background_priority: float = 0.1):
        """
        初始化ByteTrack追踪器
        
        Args:
            detector_config: 检测器配置字典，包含:
                - model_path: YOLO模型路径
                - conf_threshold: 检测置信度阈值
                - iou_threshold: NMS IoU阈值
                - device: 计算设备
                - frame_rate: 视频帧率
            priority_boost: 检测目标的优先级增强系数
            background_priority: 背景区域的基础优先级
        """
        if detector_config is None:
            detector_config = {
                'model_path': 'yolo11n.pt',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'device': 'cuda',
                'frame_rate': 30
            }
        
        self.detector = DetectionTracker(**detector_config)
        self.priority_boost = priority_boost
        self.background_priority = background_priority
        print(f"✅ 初始化LimitNet ByteTrack追踪器，优先级增强: {priority_boost}")
    
    def generate_priority_map(self, encoded_features: torch.Tensor, 
                            original_images: torch.Tensor) -> torch.Tensor:
        """
        为LimitNet生成优先级图，替代原来的显著性检测
        
        Args:
            encoded_features: 编码特征 (B, C, H, W)
            original_images: 原始图像 (B, C, H, W)
            
        Returns:
            priority_map: 优先级图 (B, 1, H, W) 匹配encoded_features的空间尺寸
        """
        batch_size = encoded_features.shape[0]
        feature_h, feature_w = encoded_features.shape[2], encoded_features.shape[3]
        
        # 使用检测追踪获取优先级掩码
        priority_masks = self.detector.process_tensor_batch(original_images)
        
        # 调整尺寸以匹配编码特征
        if priority_masks.shape[2:] != (feature_h, feature_w):
            priority_masks = transforms.Resize(
                (feature_h, feature_w),
                interpolation=transforms.InterpolationMode.BILINEAR
            )(priority_masks)
        
        # 应用优先级增强
        enhanced_masks = priority_masks * self.priority_boost + self.background_priority
        
        # 确保值在合理范围内
        enhanced_masks = torch.clamp(enhanced_masks, 0, 2.0)
        
        return enhanced_masks
    
    def reset_tracker(self):
        """重置追踪器"""
        self.detector.reset_tracker() 