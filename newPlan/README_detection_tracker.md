# LimitNet with YOLOv11n + ByteTrack 检测追踪方案

## 项目概述

这是对原始LimitNet项目的改进版本，使用YOLOv11n目标检测和ByteTrack多目标追踪来替代原来的显著性检测，以生成更加准确和实用的高优先级传输区域。

## 主要改进

### 1. 检测追踪替代显著性检测
- **原方案**: 使用BASNet进行显著性检测
- **新方案**: 使用YOLOv11n + ByteTrack进行目标检测和追踪
- **优势**: 
  - 更准确的目标识别
  - 实时的目标追踪能力（ByteTrack专为速度优化）
  - 可配置的检测阈值
  - 支持多种目标类别
  - 更低的计算复杂度

### 2. 智能优先级分配
- 基于目标检测置信度和目标大小动态分配优先级
- 支持多目标同时追踪和优先级管理
- 保持背景区域的基础优先级

### 3. 端到端的训练和推理
- 保持原有的三阶段训练策略
- 检测器独立运行，不参与梯度更新
- 支持视频序列的连续追踪

## 文件结构

```
newPlan/
├── requirements_new.txt           # 新增依赖包
├── detection_tracker.py           # 检测追踪核心模块
├── limitnet_with_detection.py     # 修改后的LimitNet模型
├── train_with_detection.py        # 训练脚本
├── demo_with_detection.py         # 单图像演示脚本
├── process_video_with_detection.py # 视频处理脚本
└── README_detection_tracker.md    # 本说明文档
```

## 安装依赖

```bash
cd newPlan
pip install -r requirements_new.txt
```

## 核心组件说明

### 1. DetectionTracker 类
负责YOLOv11n检测和BoxMOT追踪的核心逻辑：
- 实时目标检测
- 多目标追踪
- 优先级掩码生成

### 2. LimitNetWithDetection 类
集成检测追踪的LimitNet模型：
- 保持原有的编码器-解码器架构
- 用检测结果替代显著性图
- 支持三阶段训练

### 3. 优先级计算机制
```python
priority = confidence * (0.8 + 0.2 * min(area_ratio * 10, 1.0))
```
- `confidence`: 检测置信度
- `area_ratio`: 目标占图像面积比例
- 结合置信度和大小信息计算最终优先级

## 使用方法

### 1. 训练模型

```bash
# CIFAR100训练
python train_with_detection.py \
    --model cifar \
    --data_root /path/to/cifar100 \
    --batch_size 32 \
    --epochs_phase1 50 \
    --epochs_phase2 30 \
    --epochs_phase3 20 \
    --checkpoint_dir ./checkpoints_detection \
    --wandb_project LimitNet_Detection

# ImageNet训练
python train_with_detection.py \
    --model imagenet \
    --data_root /path/to/imagenet \
    --batch_size 16 \
    --epochs_phase1 30 \
    --epochs_phase2 20 \
    --epochs_phase3 10 \
    --checkpoint_dir ./checkpoints_detection
```

### 2. 单图像演示

```bash
python demo_with_detection.py \
    --model cifar \
    --model_path ./checkpoints_detection/limitnet_detection_final.ckpt \
    --image_path /path/to/image.jpg \
    --percentage 0.3 \
    --output_dir ./demo_outputs \
    --yolo_model yolo11n.pt
```

### 3. 视频处理

```bash
python process_video_with_detection.py \
    --model cifar \
    --model_path ./checkpoints_detection/limitnet_detection_final.ckpt \
    --input_video /path/to/video.mp4 \
    --output_folder ./video_outputs \
    --output_video ./reconstructed_video.mp4 \
    --percentage 0.3 \
    --save_comparison \
    --save_priority_maps
```

## 参数说明

### 检测器参数
- `--yolo_model`: YOLOv11模型路径 (默认: yolo11n.pt)
- `--detection_conf`: 检测置信度阈值 (默认: 0.25)
- `--detection_iou`: NMS IoU阈值 (默认: 0.45)
- `--priority_boost`: 检测目标优先级增强系数 (默认: 1.5)
- `--background_priority`: 背景区域基础优先级 (默认: 0.1)

### 训练参数
- `--epochs_phase1`: 重建训练轮数
- `--epochs_phase2`: 检测+重建训练轮数  
- `--epochs_phase3`: 分类训练轮数
- `--percentage`: 数据保留比例 (0.0-1.0)

## 优势对比

| 特性 | 原显著性检测方案 | 新检测追踪方案 |
|------|------------------|----------------|
| 目标识别准确性 | 基于视觉显著性，可能不准确 | 基于训练好的检测器，更准确 |
| 实时性能 | 需要额外的显著性网络 | YOLOv11n速度快 |
| 目标追踪 | 不支持 | 支持多目标追踪 |
| 可解释性 | 显著性图较抽象 | 检测框和ID更直观 |
| 配置灵活性 | 固定的显著性检测 | 可调节检测阈值和优先级 |
| 应用场景 | 通用图像压缩 | 特别适合监控、自动驾驶等 |

## 典型应用场景

1. **智能监控系统**: 优先传输人员和车辆等重要目标
2. **自动驾驶**: 优先传输行人、车辆、交通标志等关键物体
3. **无人机图传**: 根据任务需求优先传输特定目标
4. **移动直播**: 智能识别并优先传输主要内容

## 性能期望

相比原方案，新方案预期能够：
- 提高目标区域的重建质量
- 在相同带宽下传输更多有用信息
- 在视频序列中保持目标的时间一致性
- 提供更好的用户体验和应用价值

## 技术细节

### 检测追踪工作流程
1. YOLOv11n对输入图像进行目标检测
2. BoxMOT对检测结果进行多目标追踪
3. 根据检测框和置信度生成优先级掩码
4. 优先级掩码指导LimitNet的渐进式传输

### 训练策略
- **阶段1**: 仅训练重建能力，检测器提供优先级指导
- **阶段2**: 联合训练重建和检测一致性
- **阶段3**: 端到端分类训练

### 内存和计算优化
- 检测器使用FP16精度减少显存占用
- 支持批量处理提高效率
- 可选的CPU fallback支持

## 注意事项

1. 首次运行会自动下载YOLOv11n权重
2. BoxMOT追踪器需要连续的视频帧才能发挥最佳效果
3. 检测阈值需要根据具体应用场景调整
4. GPU内存至少需要4GB以上

## 未来改进方向

1. 支持自定义目标类别和优先级
2. 集成更多追踪算法选择
3. 添加目标级别的压缩控制
4. 支持分布式训练和推理
5. 集成边缘设备优化版本

## 问题反馈

如有问题或建议，请通过以下方式反馈：
- 创建GitHub Issue
- 邮件联系项目维护者
- 参与技术讨论社区 