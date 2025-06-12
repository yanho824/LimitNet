# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from limitnet_with_detection import LimitNetWithDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Process Video with LimitNet Detection')
    parser.add_argument('--model', type=str, choices=['cifar', 'imagenet'], default='cifar',
                       help='Model type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--input_video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder for processed frames')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--percentage', type=float, default=0.3,
                       help='Percentage of latent variables to keep')
    parser.add_argument('--frame_rate', type=int, default=30,
                       help='Output video frame rate')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (None for all)')
    
    # 检测器配置
    parser.add_argument('--yolo_model', type=str, default='yolo11n.pt',
                       help='YOLO model path')
    parser.add_argument('--detection_conf', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--detection_iou', type=float, default=0.45,
                       help='Detection IoU threshold')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_comparison', action='store_true',
                       help='Save comparison frames (original vs reconstructed)')
    parser.add_argument('--save_priority_maps', action='store_true',
                       help='Save priority maps for each frame')
    
    return parser.parse_args()


class VideoProcessor:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 用于显示的变换
        self.display_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def denormalize_tensor(self, tensor):
        """反归一化张量"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor.device.type == 'cuda':
            mean = mean.cuda()
            std = std.cuda()
        
        denormalized = tensor * std + mean
        return torch.clamp(denormalized, 0, 1)
    
    def tensor_to_frame(self, tensor):
        """将张量转换为OpenCV格式的帧"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # 转换为numpy并调整维度顺序
        frame_np = tensor.cpu().detach().numpy().transpose(1, 2, 0)
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def save_priority_map_frame(self, priority_map, frame_idx, output_dir):
        """保存优先级图帧"""
        if priority_map.dim() == 4:
            priority_map = priority_map[0, 0]
        elif priority_map.dim() == 3:
            priority_map = priority_map[0]
        
        priority_np = priority_map.cpu().detach().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(priority_np, cmap='jet')
        plt.colorbar()
        plt.title(f'Frame {frame_idx} - Detection Priority Map')
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f'priority_map_{frame_idx:06d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    
    def create_comparison_frame(self, original, reconstructed, priority_map, frame_idx):
        """创建对比帧"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始帧
        if original.dim() == 4:
            original = original[0]
        original_np = original.cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(original_np)
        axes[0].set_title(f'Frame {frame_idx} - Original')
        axes[0].axis('off')
        
        # 重建帧
        if reconstructed.dim() == 4:
            reconstructed = reconstructed[0]
        reconstructed_np = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)
        axes[1].imshow(reconstructed_np)
        axes[1].set_title(f'Frame {frame_idx} - Reconstructed (p={self.args.percentage})')
        axes[1].axis('off')
        
        # 优先级图
        if priority_map.dim() == 4:
            priority_map = priority_map[0, 0]
        elif priority_map.dim() == 3:
            priority_map = priority_map[0]
        priority_np = priority_map.cpu().detach().numpy()
        im = axes[2].imshow(priority_np, cmap='jet')
        axes[2].set_title(f'Frame {frame_idx} - Priority Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        
        # 转换为OpenCV格式
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        # RGB to BGR
        comparison_frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        return comparison_frame
    
    def process_frame(self, frame, frame_idx):
        """处理单帧"""
        # 预处理帧
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        display_tensor = self.display_transform(frame)
        
        # 模型推理
        with torch.no_grad():
            reconstructed, priority_map = self.model(input_tensor)
            reconstructed_display = self.denormalize_tensor(reconstructed)
        
        # 转换为OpenCV格式
        reconstructed_frame = self.tensor_to_frame(reconstructed_display)
        
        results = {
            'original_tensor': display_tensor,
            'reconstructed_tensor': reconstructed_display,
            'priority_map': priority_map,
            'reconstructed_frame': reconstructed_frame
        }
        
        return results
    
    def process_video(self):
        """处理整个视频"""
        # 打开输入视频
        cap = cv2.VideoCapture(self.args.input_video)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.args.input_video}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {total_frames} 帧, {fps} FPS, {width}x{height}")
        
        # 确定处理帧数
        frames_to_process = min(total_frames, self.args.max_frames) if self.args.max_frames else total_frames
        print(f"将处理 {frames_to_process} 帧")
        
        # 创建输出目录
        os.makedirs(self.args.output_folder, exist_ok=True)
        if self.args.save_comparison:
            os.makedirs(os.path.join(self.args.output_folder, 'comparisons'), exist_ok=True)
        if self.args.save_priority_maps:
            os.makedirs(os.path.join(self.args.output_folder, 'priority_maps'), exist_ok=True)
        
        # 准备输出视频编写器
        output_writers = {}
        if self.args.output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writers['reconstructed'] = cv2.VideoWriter(
                self.args.output_video, fourcc, self.args.frame_rate, (224, 224)
            )
            
            if self.args.save_comparison:
                comparison_path = self.args.output_video.replace('.mp4', '_comparison.mp4')
                # 比较视频的尺寸需要根据实际创建的比较帧确定
                output_writers['comparison'] = None  # 稍后初始化
        
        # 重置追踪器
        self.model.reset_detection_tracker()
        
        frame_idx = 0
        processed_frames = 0
        
        with tqdm(total=frames_to_process, desc="处理视频帧") as pbar:
            while cap.isOpened() and processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # 处理帧
                    results = self.process_frame(frame, frame_idx)
                    
                    # 保存重建帧
                    reconstructed_path = os.path.join(
                        self.args.output_folder, f'reconstructed_{frame_idx:06d}.png'
                    )
                    cv2.imwrite(reconstructed_path, results['reconstructed_frame'])
                    
                    # 写入输出视频
                    if 'reconstructed' in output_writers and output_writers['reconstructed']:
                        output_writers['reconstructed'].write(results['reconstructed_frame'])
                    
                    # 保存优先级图
                    if self.args.save_priority_maps:
                        self.save_priority_map_frame(
                            results['priority_map'], 
                            frame_idx, 
                            os.path.join(self.args.output_folder, 'priority_maps')
                        )
                    
                    # 创建和保存对比帧
                    if self.args.save_comparison:
                        comparison_frame = self.create_comparison_frame(
                            results['original_tensor'],
                            results['reconstructed_tensor'],
                            results['priority_map'],
                            frame_idx
                        )
                        
                        comparison_path = os.path.join(
                            self.args.output_folder, 'comparisons', f'comparison_{frame_idx:06d}.png'
                        )
                        cv2.imwrite(comparison_path, comparison_frame)
                        
                        # 初始化比较视频编写器
                        if 'comparison' in output_writers and output_writers['comparison'] is None:
                            comparison_video_path = self.args.output_video.replace('.mp4', '_comparison.mp4')
                            h, w = comparison_frame.shape[:2]
                            output_writers['comparison'] = cv2.VideoWriter(
                                comparison_video_path, fourcc, self.args.frame_rate, (w, h)
                            )
                        
                        if 'comparison' in output_writers and output_writers['comparison']:
                            output_writers['comparison'].write(comparison_frame)
                    
                    processed_frames += 1
                    frame_idx += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"处理第 {frame_idx} 帧时出错: {e}")
                    frame_idx += 1
                    continue
        
        # 释放资源
        cap.release()
        for writer in output_writers.values():
            if writer:
                writer.release()
        
        print(f"视频处理完成！处理了 {processed_frames} 帧")
        print(f"输出文件夹: {self.args.output_folder}")
        if self.args.output_video:
            print(f"输出视频: {self.args.output_video}")


def main():
    args = parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    
    # 配置检测器
    detector_config = {
        'model_path': args.yolo_model,
        'conf_threshold': args.detection_conf,
        'iou_threshold': args.detection_iou,
        'device': device
    }
    
    print("加载模型...")
    try:
        model = LimitNetWithDetection.load_from_checkpoint(
            args.model_path,
            model_type=args.model,
            detector_config=detector_config
        )
        model.eval().to(device)
        model.p = args.percentage
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建视频处理器
    processor = VideoProcessor(model, device, args)
    
    # 处理视频
    print("开始处理视频...")
    processor.process_video()
    
    print("视频处理完成！")


if __name__ == "__main__":
    main() 