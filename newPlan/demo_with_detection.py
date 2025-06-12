# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

from limitnet_with_detection import LimitNetWithDetection


def parse_args():
    parser = argparse.ArgumentParser(description='LimitNet Detection Demo')
    parser.add_argument('--model', type=str, choices=['cifar', 'imagenet'], default='cifar',
                       help='Model type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--percentage', type=float, default=0.3,
                       help='Percentage of latent variables to keep (0.0-1.0)')
    parser.add_argument('--output_dir', type=str, default='./demo_outputs',
                       help='Output directory for results')
    
    # 检测器配置
    parser.add_argument('--yolo_model', type=str, default='yolo11n.pt',
                       help='YOLO model path')
    parser.add_argument('--detection_conf', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--detection_iou', type=float, default=0.45,
                       help='Detection IoU threshold')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_and_preprocess_image(image_path):
    """加载并预处理图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 用于显示的变换（不归一化）
    transform_for_display = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    display_tensor = transform_for_display(image)
    
    return input_tensor, display_tensor, original_size


def denormalize_tensor(tensor):
    """反归一化张量用于显示"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.device.type == 'cuda':
        mean = mean.cuda()
        std = std.cuda()
    
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


def save_tensor_as_image(tensor, filepath, title=None):
    """保存张量为图像"""
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]  # 取第一个样本
    
    # 转换为numpy并调整维度顺序
    img_np = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img_np)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def save_priority_map(priority_map, filepath, title=None):
    """保存优先级图"""
    if priority_map.dim() == 4:  # (B, 1, H, W)
        priority_map = priority_map[0, 0]  # 取第一个样本的第一个通道
    elif priority_map.dim() == 3:  # (1, H, W)
        priority_map = priority_map[0]
    
    priority_np = priority_map.cpu().detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(priority_np, cmap='jet', alpha=0.8)
    plt.colorbar(label='Priority')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def create_comparison_plot(original, reconstructed, priority_map, save_path):
    """创建对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    if original.dim() == 4:
        original = original[0]
    original_np = original.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 重建图像
    if reconstructed.dim() == 4:
        reconstructed = reconstructed[0]
    reconstructed_np = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('Reconstructed Image', fontsize=12)
    axes[1].axis('off')
    
    # 优先级图
    if priority_map.dim() == 4:
        priority_map = priority_map[0, 0]
    elif priority_map.dim() == 3:
        priority_map = priority_map[0]
    priority_np = priority_map.cpu().detach().numpy()
    im = axes[2].imshow(priority_np, cmap='jet')
    axes[2].set_title('Detection Priority Map', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_detections(image_path, detector, save_path):
    """可视化检测结果"""
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 进行检测
    priority_mask, track_info = detector.detect_and_track(image)
    
    # 在图像上绘制检测框
    image_with_boxes = image.copy()
    for track in track_info:
        bbox = track['bbox']
        conf = track['conf']
        track_id = track['id']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制边界框
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签
        label = f"ID:{track_id} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 保存结果
    cv2.imwrite(save_path, image_with_boxes)
    print(f"检测结果已保存至: {save_path}")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        # 加载训练好的模型
        model = LimitNetWithDetection.load_from_checkpoint(
            args.model_path,
            model_type=args.model,
            detector_config=detector_config
        )
        model.eval().to(device)
        model.p = args.percentage  # 设置保留比例
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print("处理图像...")
    # 加载和预处理图像
    input_tensor, display_tensor, original_size = load_and_preprocess_image(args.image_path)
    input_tensor = input_tensor.to(device)
    
    print(f"输入图像尺寸: {original_size}")
    print(f"保留比例: {args.percentage}")
    
    # 进行推理
    with torch.no_grad():
        # 前向传播
        reconstructed, priority_map = model(input_tensor)
        
        # 反归一化重建图像用于显示
        reconstructed_display = denormalize_tensor(reconstructed)
    
    print("保存结果...")
    
    # 保存各种输出
    save_tensor_as_image(
        display_tensor, 
        os.path.join(args.output_dir, 'original.png'),
        'Original Image'
    )
    
    save_tensor_as_image(
        reconstructed_display,
        os.path.join(args.output_dir, 'reconstructed.png'),
        f'Reconstructed (p={args.percentage})'
    )
    
    save_priority_map(
        priority_map,
        os.path.join(args.output_dir, 'priority_map.png'),
        'Detection Priority Map'
    )
    
    # 创建对比图
    create_comparison_plot(
        display_tensor,
        reconstructed_display,
        priority_map,
        os.path.join(args.output_dir, 'comparison.png')
    )
    
    # 可视化检测结果
    visualize_detections(
        args.image_path,
        model.detection_tracker.detector,
        os.path.join(args.output_dir, 'detections.png')
    )
    
    # 保存统计信息
    stats = {
        'percentage_kept': args.percentage,
        'original_size': original_size,
        'model_type': args.model,
        'device': device,
        'priority_map_stats': {
            'min': float(priority_map.min()),
            'max': float(priority_map.max()),
            'mean': float(priority_map.mean()),
            'std': float(priority_map.std())
        }
    }
    
    import json
    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"所有结果已保存至: {args.output_dir}")
    print(f"优先级图统计: min={stats['priority_map_stats']['min']:.3f}, "
          f"max={stats['priority_map_stats']['max']:.3f}, "
          f"mean={stats['priority_map_stats']['mean']:.3f}")


if __name__ == "__main__":
    main() 