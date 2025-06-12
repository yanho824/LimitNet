# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import os
import sys
from collections import OrderedDict

# 添加根目录到Python路径，以便导入models模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # LimitNet根目录
sys.path.insert(0, root_dir)

print(f"🔧 添加模块搜索路径: {root_dir}")

# 现在应该可以成功导入models模块
try:
    from models.model import LimitNet  # 导入原始LimitNet类
    print("✅ 成功导入原始LimitNet类")
except ImportError as e:
    print(f"⚠️  无法导入原始LimitNet类: {e}")

from limitnet_with_detection import LimitNetWithDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer weights from original LimitNet to Detection version')
    parser.add_argument('--original_model_path', type=str, required=True,
                       help='Path to original LimitNet model')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for new model with transferred weights')
    parser.add_argument('--model_type', type=str, choices=['cifar', 'imagenet'], default='cifar',
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    return parser.parse_args()


def load_original_model(model_path, device='cuda'):
    """加载原始LimitNet模型 - 支持多种格式和错误恢复"""
    print(f"加载原始模型: {model_path}")
    
    # 方法1: 尝试weights_only加载（跳过模块依赖）
    try:
        print("🔍 方法1: weights_only模式...")
        original_state = torch.load(model_path, map_location=device, weights_only=True)
        print("✅ 成功加载原始模型 (weights_only)")
        return original_state
    except Exception as e:
        print(f"⚠️  方法1失败: {e}")
    
    # 方法2: 自定义pickle加载器（跳过models模块）
    try:
        print("🔍 方法2: 自定义pickle加载器...")
        import pickle
        
        class ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # 如果遇到models模块相关的类，返回一个占位符
                if 'models' in module:
                    return lambda *args, **kwargs: None
                return super().find_class(module, name)
        
        with open(model_path, 'rb') as f:
            original_state = ModelUnpickler(f).load()
        print("✅ 成功加载原始模型 (自定义unpickler)")
        return original_state
    except Exception as e:
        print(f"⚠️  方法2失败: {e}")
    
    # 方法3: 尝试直接加载并忽略错误
    try:
        print("🔍 方法3: 直接加载...")
        original_state = torch.load(model_path, map_location=device)
        print("✅ 成功加载原始模型 (直接加载)")
        return original_state
    except Exception as e:
        print(f"⚠️  方法3失败: {e}")
    
    print(f"❌ 所有加载方法都失败了")
    return None


def extract_transferable_weights(original_state):
    """提取可迁移的权重"""
    transferable_weights = OrderedDict()
    
    # 需要迁移的组件
    components_to_transfer = [
        'encoder',      # 编码器
        'decoder',      # 解码器
        'cls_model'     # 分类器
    ]
    
    print("🔍 分析原始模型格式...")
    print(f"原始模型类型: {type(original_state)}")
    
    if isinstance(original_state, dict):
        print("📦 字典格式，检查可用键...")
        print(f"可用键: {list(original_state.keys())}")
        
        if 'state_dict' in original_state:
            state_dict = original_state['state_dict']
            print("✅ 找到state_dict键")
        else:
            state_dict = original_state
            print("✅ 直接使用字典作为state_dict")
    elif hasattr(original_state, 'state_dict'):
        print("📦 模型对象格式，提取state_dict...")
        state_dict = original_state.state_dict()
        print("✅ 成功提取state_dict")
    else:
        print("❌ 无法解析模型状态字典")
        print(f"模型对象属性: {dir(original_state)[:10]}...")  # 显示前10个属性
        return None
    
    print(f"\n🔍 分析原始模型权重... (共{len(state_dict)}个参数)")
    
    # 显示前几个键作为示例
    sample_keys = list(state_dict.keys())[:5]
    print(f"示例键: {sample_keys}")
    
    for key, value in state_dict.items():
        if hasattr(value, 'shape'):
            print(f"  - {key}: {value.shape}")
            
            # 检查是否属于可迁移的组件
            for component in components_to_transfer:
                if key.startswith(component):
                    transferable_weights[key] = value
                    break
        else:
            print(f"  - {key}: {type(value)} (非张量)")
    
    print(f"\n✅ 找到 {len(transferable_weights)} 个可迁移的权重")
    if len(transferable_weights) > 0:
        print("迁移的权重包括:")
        for key in transferable_weights.keys():
            print(f"  ✓ {key}")
    
    return transferable_weights


def create_new_model_with_weights(model_type, transferable_weights, device='cuda'):
    """创建带有迁移权重的新模型"""
    print("🚀 创建新的检测追踪模型...")
    
    # 配置检测器（使用简单配置避免依赖问题）
    detector_config = {
        'model_path': 'yolo11n.pt',
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'device': device
    }
    
    # 创建新模型
    new_model = LimitNetWithDetection(
        model_type=model_type,
        detector_config=detector_config
    )
    
    # 获取新模型的状态字典
    new_state_dict = new_model.state_dict()
    
    # 迁移权重
    transferred_count = 0
    skipped_count = 0
    
    print("\n🔄 迁移权重...")
    for key, value in transferable_weights.items():
        if key in new_state_dict:
            if new_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
                transferred_count += 1
                print(f"  ✅ {key}: {value.shape}")
            else:
                print(f"  ⚠️  {key}: 形状不匹配 {new_state_dict[key].shape} vs {value.shape}")
                skipped_count += 1
        else:
            print(f"  ❌ {key}: 在新模型中不存在")
            skipped_count += 1
    
    # 加载新的状态字典
    new_model.load_state_dict(new_state_dict)
    
    print(f"\n📊 迁移统计:")
    print(f"  - 成功迁移: {transferred_count} 个权重")
    print(f"  - 跳过: {skipped_count} 个权重")
    print(f"  - 迁移比例: {transferred_count/(transferred_count+skipped_count)*100:.1f}%")
    
    return new_model


def analyze_model_differences(original_state, new_model):
    """分析模型差异"""
    print("\n🔍 模型差异分析:")
    
    if isinstance(original_state, dict):
        if 'state_dict' in original_state:
            original_keys = set(original_state['state_dict'].keys())
        else:
            original_keys = set(original_state.keys())
    else:
        print("❌ 无法分析原始模型")
        return
    
    new_keys = set(new_model.state_dict().keys())
    
    # 原始模型有但新模型没有的
    only_in_original = original_keys - new_keys
    if only_in_original:
        print("\n❌ 仅在原始模型中存在（将被丢弃）:")
        for key in sorted(only_in_original):
            print(f"  - {key}")
    
    # 新模型有但原始模型没有的
    only_in_new = new_keys - original_keys
    if only_in_new:
        print("\n🆕 仅在新模型中存在（将随机初始化）:")
        for key in sorted(only_in_new):
            print(f"  - {key}")
    
    # 共同存在的
    common_keys = original_keys & new_keys
    print(f"\n✅ 共同存在（可以迁移）: {len(common_keys)} 个权重")


def save_model(model, output_path, additional_info=None):
    """保存模型"""
    print(f"\n💾 保存模型至: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备保存的数据
    save_data = {
        'state_dict': model.state_dict(),
        'model_type': model.hparams.get('model_type', 'unknown') if hasattr(model, 'hparams') else 'unknown',
        'transfer_info': additional_info or {}
    }
    
    torch.save(save_data, output_path)
    print("✅ 模型保存成功!")


def main():
    args = parse_args()
    
    print("🎯 LimitNet权重迁移工具")
    print("从原始LimitNet迁移编码器和解码器权重到检测追踪版本")
    
    # 设置设备
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载原始模型
    original_state = load_original_model(args.original_model_path, device)
    if original_state is None:
        print("❌ 无法加载原始模型，退出...")
        return
    
    # 提取可迁移的权重
    transferable_weights = extract_transferable_weights(original_state)
    if transferable_weights is None or len(transferable_weights) == 0:
        print("❌ 没有找到可迁移的权重，退出...")
        return
    
    # 创建新模型并迁移权重
    try:
        new_model = create_new_model_with_weights(args.model_type, transferable_weights, device)
    except Exception as e:
        print(f"❌ 创建新模型失败: {e}")
        return
    
    # 分析模型差异
    analyze_model_differences(original_state, new_model)
    
    # 保存新模型
    transfer_info = {
        'original_model_path': args.original_model_path,
        'transfer_timestamp': torch.tensor(torch.initial_seed()),  # 简单的时间戳
        'transferred_components': ['encoder', 'decoder', 'cls_model']
    }
    
    save_model(new_model, args.output_path, transfer_info)
    
    print("\n🎉 权重迁移完成!")
    print("\n📋 下一步:")
    print(f"1. 使用迁移后的模型开始训练:")
    print(f"   python train_with_detection.py --resume_from {args.output_path}")
    print("2. 由于检测机制改变，建议:")
    print("   - 阶段1: 使用较少的epochs (10-20)")
    print("   - 阶段2: 重点训练检测-重建配合 (20-30 epochs)")
    print("   - 阶段3: 微调分类性能 (10-20 epochs)")


if __name__ == "__main__":
    main() 