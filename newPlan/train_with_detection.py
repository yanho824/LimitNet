# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from limitnet_with_detection import LimitNetWithDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Train LimitNet with Detection Tracking')
    
    # 模型配置
    parser.add_argument('--model', type=str, choices=['cifar', 'imagenet'], default='cifar',
                       help='Model type: cifar or imagenet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs_phase1', type=int, default=50, help='Epochs for phase 1 (reconstruction)')
    parser.add_argument('--epochs_phase2', type=int, default=30, help='Epochs for phase 2 (detection + reconstruction)')
    parser.add_argument('--epochs_phase3', type=int, default=20, help='Epochs for phase 3 (classification)')
    
    # 数据配置
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # 检测器配置
    parser.add_argument('--yolo_model', type=str, default='yolo11n.pt', help='YOLO model path')
    parser.add_argument('--detection_conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--detection_iou', type=float, default=0.45, help='Detection IoU threshold')
    parser.add_argument('--priority_boost', type=float, default=1.5, help='Priority boost for detected objects')
    parser.add_argument('--background_priority', type=float, default=0.1, help='Background priority')
    
    # 训练配置
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_detection', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # 日志配置
    parser.add_argument('--wandb_project', type=str, default='LimitNet_Detection', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log every n steps')
    
    return parser.parse_args()


def get_dataset(args):
    """获取数据集"""
    if args.model == 'cifar':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 检查本地是否已有数据集
        cifar_path = os.path.join(args.data_root, 'cifar-100-python')
        download_needed = not os.path.exists(cifar_path)
        if download_needed:
            print("📥 本地未找到CIFAR-100数据集，将自动下载...")
        else:
            print("✅ 找到本地CIFAR-100数据集，跳过下载")
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_root, train=True, download=download_needed, transform=transform_train
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=args.data_root, train=False, download=download_needed, transform=transform_val
        )
        
    elif args.model == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_root, 'train'), transform=transform_train
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_root, 'val'), transform=transform_val
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader


def train_phase(model, train_loader, val_loader, phase, epochs, args):
    """训练特定阶段"""
    print(f"\n=== 开始训练阶段 {phase} ===")
    
    # 设置模型阶段
    model.PHASE = phase
    
    # 配置检查点和日志
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, f'phase_{phase}'),
        filename=f'limitnet_detection_phase{phase}_' + '{epoch:02d}_{val_loss_avg_phase' + str(phase) + ':.3f}',
        monitor=f'val_loss_avg_phase{phase}',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.wandb_name}_phase{phase}" if args.wandb_name else f"phase{phase}",
        save_dir=args.checkpoint_dir
    )
    
    # 配置训练器
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=1,
        precision='16-mixed' if args.device == 'cuda' else 32,
        gradient_clip_val=1.0,
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 保存阶段模型
    phase_model_path = os.path.join(args.checkpoint_dir, f'limitnet_detection_phase{phase}_final.ckpt')
    trainer.save_checkpoint(phase_model_path)
    print(f"阶段 {phase} 训练完成，模型保存至: {phase_model_path}")
    
    return trainer.callback_metrics


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 配置检测器
    detector_config = {
        'model_path': args.yolo_model,
        'conf_threshold': args.detection_conf,
        'iou_threshold': args.detection_iou,
        'device': args.device
    }
    
    print(f"使用设备: {args.device}")
    print(f"检测器配置: {detector_config}")
    
    # 获取数据
    print("准备数据集...")
    train_loader, val_loader = get_dataset(args)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("初始化模型...")
    model = LimitNetWithDetection(
        model_type=args.model,
        detector_config=detector_config
    )
    
    # 检查是否需要恢复训练或加载预训练权重
    start_phase = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"从检查点恢复训练: {args.resume_from}")
        try:
            # 尝试加载Lightning检查点
            model = LimitNetWithDetection.load_from_checkpoint(
                args.resume_from,
                model_type=args.model,
                detector_config=detector_config
            )
            print("✅ 成功加载Lightning检查点")
        except Exception as e:
            print(f"Lightning检查点加载失败: {e}")
            try:
                # 尝试加载普通PyTorch模型
                checkpoint = torch.load(args.resume_from, map_location=args.device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("✅ 成功加载PyTorch状态字典")
                    if 'transfer_info' in checkpoint:
                        print("🔄 检测到权重迁移信息:")
                        print(f"  原始模型: {checkpoint['transfer_info'].get('original_model_path', '未知')}")
                        print(f"  迁移组件: {checkpoint['transfer_info'].get('transferred_components', [])}")
                else:
                    print("❌ 无法识别的模型格式")
            except Exception as e2:
                print(f"PyTorch模型加载也失败: {e2}")
                print("将使用随机初始化的模型")
    
    # 三阶段训练
    phases_config = [
        (1, args.epochs_phase1, "重建训练"),
        (2, args.epochs_phase2, "检测+重建训练"), 
        (3, args.epochs_phase3, "分类训练")
    ]
    
    for phase, epochs, description in phases_config[start_phase-1:]:
        print(f"\n{'='*50}")
        print(f"阶段 {phase}: {description}")
        print(f"训练轮数: {epochs}")
        print(f"{'='*50}")
        
        try:
            metrics = train_phase(model, train_loader, val_loader, phase, epochs, args)
            print(f"阶段 {phase} 完成，最终指标: {metrics}")
            
        except KeyboardInterrupt:
            print(f"\n训练被用户中断，正在保存当前模型...")
            interrupted_path = os.path.join(args.checkpoint_dir, f'interrupted_phase{phase}.ckpt')
            torch.save(model.state_dict(), interrupted_path)
            print(f"模型已保存至: {interrupted_path}")
            break
        except Exception as e:
            print(f"阶段 {phase} 训练出错: {str(e)}")
            break
    
    print("\n训练完成!")
    final_model_path = os.path.join(args.checkpoint_dir, 'limitnet_detection_final.ckpt')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型保存至: {final_model_path}")


if __name__ == "__main__":
    main() 