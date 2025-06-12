# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import wandb
from efficientnet_pytorch import EfficientNet

from detection_tracker import LimitNetDetectionTracker

device = 'cuda'


class CIFAR100Classifier(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        # 加载 newPlan 目录下的 EfficentNet-CIFAR100 模型
        self.model = torch.load('EfficentNet-CIFAR100')
        self.accuracy = Accuracy(task="multiclass", num_classes=100)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        outputs = self(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss
        }

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'acc': acc
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer


class Encoder(L.LightningModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.run_on_nRF = False
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=4, padding=1)
        self.conv3 = nn.Conv2d(16, 12, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Decoder(L.LightningModule):
    def __init__(self):
        super(Decoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(12, 64, 7, stride=2, padding=3, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 64, 5, stride=4, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x)) 
        x = F.relu(self.t_conv3(x)) 
        x = F.relu(self.t_conv4(x)) 
        x = self.t_conv5(x)
        x = torch.sigmoid(x)
        return x


class LimitNetWithDetection(L.LightningModule):
    """
    修改后的LimitNet模型，使用检测追踪替代显著性检测
    """
    def __init__(self, model_type='cifar', detector_config=None):
        super(LimitNetWithDetection, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # 初始化分类器
        if model_type == 'cifar':
            self.accuracy = Accuracy(task="multiclass", num_classes=100)
            self.cls_model = CIFAR100Classifier(learning_rate=0.001)
        elif model_type == 'imagenet':
            self.accuracy = Accuracy(task="multiclass", num_classes=1000)
            self.cls_model = mobilenet_v2(pretrained=True)
        
        # 初始化检测追踪器
        if detector_config is None:
            detector_config = {
                'model_path': 'yolo11n.pt',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        
        self.detection_tracker = LimitNetDetectionTracker(
            detector_config=detector_config,
            priority_boost=1.5,
            background_priority=0.1
        )
        
        self.transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # 训练阶段控制
        self.PHASE = None
        self.p = None  # 保留比例参数
        self.replace_tensor = torch.cuda.FloatTensor([0.0])[0] if torch.cuda.is_available() else torch.FloatTensor([0.0])[0]
        
        # 训练记录
        self.training_step_loss = []
        self.training_step_loss_detection = []
        self.training_step_loss_ce = []
        self.training_step_loss_rec = []
        self.training_step_acc = []
        
        self.validation_step_loss = []
        self.validation_step_loss_detection = []
        self.validation_step_loss_ce = []
        self.validation_step_loss_rec = []
        self.validation_step_acc = []
        
        self.save_hyperparameters()

    def random_noise(self, x, r1, r2):
        temp_x = x.clone()
        noise = (r1 - r2) * torch.rand(x.shape) + r2
        return torch.clamp(temp_x + noise, min=0.0, max=1.0)

    def forward(self, x):
        """
        前向传播，根据训练阶段返回不同输出
        """
        # 编码
        encoded = self.encoder(x)
        
        # 根据训练阶段决定是否使用检测
        if self.PHASE == 1:  # 阶段1: 只做重建，不用检测
            # 使用原始的rate_less方法进行丢弃
            if self.training or self.p is not None:
                encoded_dropped = self.rate_less(encoded)
            else:
                encoded_dropped = encoded
            
            # 添加噪声
            if self.training:
                noise = torch.rand_like(encoded_dropped, dtype=torch.float) * 0.02 - 0.01
                encoded_dropped = encoded_dropped + noise
            
            # 解码重建
            reconstructed = self.decoder(encoded_dropped)
            # 应用标准化
            reconstructed = self.transforms(reconstructed)
            return reconstructed
            
        else:  # 阶段2和3: 使用检测
            # 生成基于检测的优先级图
            detection_priority_map = self.detection_tracker.generate_priority_map(encoded, x)
            
            # 应用渐进式丢弃
            if self.training or self.p is not None:
                encoded_dropped = self.gradual_dropping_with_detection(encoded, detection_priority_map)
            else:
                encoded_dropped = encoded
            
            # 添加噪声
            if self.training:
                noise = torch.rand_like(encoded_dropped, dtype=torch.float) * 0.02 - 0.01
                encoded_dropped = encoded_dropped + noise
            
            # 解码重建
            reconstructed = self.decoder(encoded_dropped)
            # 应用标准化
            reconstructed = self.transforms(reconstructed)
            
            # 根据训练阶段返回不同结果
            if self.PHASE == 2:  # 检测+重建阶段
                return reconstructed, detection_priority_map
            elif self.PHASE == 3:  # 分类阶段
                # 对重建图像进行分类
                classification_output = self.cls_model(reconstructed)
                return classification_output
            else:  # 推理阶段
                return reconstructed, detection_priority_map

    def gradual_dropping_with_detection(self, x, detection_map):
        """
        基于检测结果的渐进式丢弃机制
        
        Args:
            x: 编码特征 (B, C, H, W)
            detection_map: 检测优先级图 (B, 1, H, W)
        
        Returns:
            处理后的编码特征
        """
        temp_x = x.clone()
        
        for i in range(x.shape[0]):
            # 获取当前样本的优先级图
            priority_map = detection_map[i, 0]  # (H, W)
            
            # 扩展到所有通道
            priority_expanded = priority_map.repeat(12, 1, 1)  # (C, H, W)
            
            # 为不同通道添加不同的优先级偏移
            for j in range(priority_expanded.shape[0]):
                priority_expanded[j, :, :] = priority_expanded[j, :, :] + (12-j) / 5
            
            # 确定保留比例
            if self.p:
                p = self.p
            else:
                p = np.random.uniform(0, 1.0, 1)[0]
            
            if p != 1.0:
                # 计算阈值进行丢弃
                q = torch.quantile(priority_expanded.view(-1), 1-p, dim=0, keepdim=True)
                selection = (priority_expanded < q)  # 选择丢弃的位置
                selection = selection.view(12, 28, 28)
                
                # 应用丢弃
                temp_x[i, :, :, :] = torch.where(
                    selection,
                    self.replace_tensor,
                    x[i, :, :, :],
                )
        
        return temp_x

    def configure_optimizers(self):
        # 只优化编码器和解码器，不优化检测器
        params_to_optimize = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.PHASE == 3:
            params_to_optimize += list(self.cls_model.parameters())
        
        optimizer = torch.optim.AdamW(params_to_optimize, lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        """训练步骤，支持多阶段训练"""
        
        if self.PHASE == 1:  # 重建训练阶段
            images, _ = train_batch
            outputs = self(images)
            loss_rec = nn.MSELoss()(outputs, images)
            
            self.log('train_loss_rec_phase1', loss_rec, on_epoch=True, prog_bar=True, logger=True)
            self.training_step_loss_rec.append(loss_rec)
            self.training_step_loss.append(loss_rec)
            
            return {'loss': loss_rec}
        
        elif self.PHASE == 2:  # 检测+重建训练阶段
            images, _ = train_batch
            outputs, detection_map = self(images)
            
            # 重建损失
            loss_rec = nn.MSELoss()(outputs, images)
            
            # 检测一致性损失（可选）
            # 这里可以添加检测结果的一致性约束
            loss_detection = torch.tensor(0.0, device=images.device)
            
            total_loss = loss_rec + 0.1 * loss_detection
            
            self.log('train_loss_rec_phase2', loss_rec, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss_detection_phase2', loss_detection, on_epoch=True, prog_bar=True, logger=True)
            
            self.training_step_loss_rec.append(loss_rec)
            self.training_step_loss_detection.append(loss_detection)
            self.training_step_loss.append(total_loss)
            
            return {'loss': total_loss}
        
        elif self.PHASE == 3:  # 分类训练阶段
            images, labels = train_batch
            outputs = self(images)
            loss_ce = nn.CrossEntropyLoss()(outputs, labels)
            
            preds = torch.argmax(outputs, dim=1)
            acc = self.accuracy(preds, labels)
            
            self.log('train_loss_ce_phase3', loss_ce, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_acc_phase3', acc, on_epoch=True, prog_bar=True, logger=True)
            
            self.training_step_loss_ce.append(loss_ce)
            self.training_step_acc.append(acc)
            self.training_step_loss.append(loss_ce)
            
            return {'loss': loss_ce}

    def validation_step(self, val_batch, batch_idx):
        """验证步骤"""
        loss_list = []
        acc_list = []
        
        # 在不同的保留比例下测试
        for test_p in [0.2, 0.5, 0.8]:
            self.p = test_p
            
            if self.PHASE == 1:
                images, _ = val_batch
                outputs = self(images)
                loss = nn.MSELoss()(outputs, images)
                loss_list.append(loss.item())
                
            elif self.PHASE == 2:
                images, _ = val_batch
                outputs, detection_map = self(images)
                loss_rec = nn.MSELoss()(outputs, images)
                loss_list.append(loss_rec.item())
                
            elif self.PHASE == 3:
                images, labels = val_batch
                outputs = self(images)
                loss_ce = nn.CrossEntropyLoss()(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                acc = self.accuracy(preds, labels)
                
                loss_list.append(loss_ce.item())
                acc_list.append(acc.item())
        
        self.p = None  # 重置
        
        # 记录平均指标
        avg_loss = torch.tensor(np.mean(loss_list))
        if acc_list:
            avg_acc = torch.tensor(np.mean(acc_list))
            self.validation_step_acc.append(avg_acc)
            
        self.validation_step_loss.append(avg_loss)
        
        return {'val_loss': avg_loss}

    def on_train_epoch_end(self):
        """训练轮次结束处理"""
        if self.training_step_loss:
            avg_loss = torch.stack(self.training_step_loss).mean()
            self.log(f'train_loss_avg_phase{self.PHASE}', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        
        # 清空记录
        self.training_step_loss.clear()
        self.training_step_loss_rec.clear()
        self.training_step_loss_detection.clear()
        self.training_step_loss_ce.clear()
        self.training_step_acc.clear()

    def on_validation_epoch_end(self):
        """验证轮次结束处理"""
        if self.validation_step_loss:
            avg_loss = torch.stack(self.validation_step_loss).mean()
            self.log(f'val_loss_avg_phase{self.PHASE}', avg_loss, on_epoch=True, prog_bar=True, logger=True)
            
        if self.validation_step_acc:
            avg_acc = torch.stack(self.validation_step_acc).mean()
            self.log(f'val_acc_avg_phase{self.PHASE}', avg_acc, on_epoch=True, prog_bar=True, logger=True)
        
        # 清空记录
        self.validation_step_loss.clear()
        self.validation_step_loss_rec.clear()
        self.validation_step_loss_detection.clear()
        self.validation_step_loss_ce.clear()
        self.validation_step_acc.clear()

    def reset_detection_tracker(self):
        """重置检测追踪器（用于新视频序列）"""
        self.detection_tracker.reset_tracker()

    def rate_less(self, x):
        """保持原来的rate_less方法用于兼容性"""
        temp_x = x.clone()
        for i in range(x.shape[0]):
            if self.p:
                p = self.p
            else:
                p = np.random.uniform(0, 1.0, 1)[0]
            if p != 1.0:
                p = int(p * x.shape[1])
                replace_tensor = torch.rand(x.shape[1]-p, x.shape[2], x.shape[3]).fill_(0)
                temp_x[i, -(x.shape[1]-p):, :, :] = replace_tensor
        return temp_x 