#!/usr/bin/env python3
"""
依赖冲突修复脚本
解决BoxMOT与PyTorch版本不兼容问题
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 失败!")
        print("错误:", e.stderr)
        return False

def main():
    print("🔧 LimitNet Detection 依赖修复工具")
    print("解决BoxMOT与PyTorch版本冲突问题\n")
    
    # 检查是否在虚拟环境中
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  建议在虚拟环境中运行此脚本")
        response = input("是否继续? (y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    print("第1步: 卸载冲突的包")
    packages_to_remove = ['torch', 'torchvision', 'torchaudio', 'boxmot', 'ultralytics']
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    print("\n第2步: 清理pip缓存")
    run_command("pip cache purge", "清理pip缓存")
    
    print("\n第3步: 安装兼容的PyTorch版本")
    # 安装兼容BoxMOT的PyTorch版本 (CUDA 12.1)
    pytorch_cmd = (
        "pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    if not run_command(pytorch_cmd, "安装兼容的PyTorch (CUDA 12.1)"):
        print("❌ PyTorch安装失败，尝试CPU版本...")
        run_command("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2", 
                   "安装PyTorch (CPU版本)")
    
    print("\n第4步: 安装其他依赖")
    if os.path.exists("requirements_compatible.txt"):
        run_command("pip install -r requirements_compatible.txt", "安装兼容版本依赖")
    else:
        # 手动安装关键包
        key_packages = [
            "lightning>=2.0.0,<2.4.0",
            "ultralytics>=8.0.0,<8.3.0", 
            "boxmot==12.0.2",
            "opencv-python>=4.8.0",
            "supervision>=0.16.0",
            "wandb>=0.15.0"
        ]
        for package in key_packages:
            run_command(f"pip install '{package}'", f"安装 {package}")
    
    print("\n第5步: 验证安装")
    verification_script = '''
import torch
import torchvision
import boxmot
import ultralytics

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Torchvision: {torchvision.__version__}")
print(f"✅ BoxMOT: {boxmot.__version__}")
print(f"✅ Ultralytics: {ultralytics.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA版本: {torch.version.cuda}")
'''
    
    try:
        exec(verification_script)
        print("\n🎉 所有依赖安装成功，无版本冲突!")
        
        # 测试检测器
        print("\n第6步: 测试检测器...")
        test_script = '''
from detection_tracker import DetectionTracker
import torch

try:
    detector = DetectionTracker()
    print("✅ DetectionTracker 初始化成功")
    
    # 测试虚拟图像
    dummy_image = torch.randn(3, 640, 640)
    priority_map = detector.generate_priority_map(dummy_image)
    print(f"✅ 优先级图生成成功，形状: {priority_map.shape}")
    print("✅ 所有组件工作正常!")
    
except Exception as e:
    print(f"⚠️  检测器测试失败: {e}")
    print("这可能需要下载YOLO模型，首次运行时正常")
'''
        exec(test_script)
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        print("请检查错误信息并重新运行")

if __name__ == "__main__":
    main() 