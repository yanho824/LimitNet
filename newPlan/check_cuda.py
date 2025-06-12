#!/usr/bin/env python3
"""
CUDA诊断脚本
检查CUDA可用性和相关配置
"""

import sys
import os
import subprocess
import platform


def print_section(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")


def run_command(command):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return str(e), False


def check_system_info():
    """检查系统信息"""
    print_section("系统信息")
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")


def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print_section("NVIDIA驱动检查")
    
    output, success = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA驱动已安装")
        print(output)
        return True
    else:
        print("❌ NVIDIA驱动未安装或不可用")
        print("请安装NVIDIA显卡驱动程序")
        return False


def check_cuda_toolkit():
    """检查CUDA工具包"""
    print_section("CUDA工具包检查")
    
    # 检查nvcc命令
    output, success = run_command("nvcc --version")
    if success:
        print("✅ CUDA工具包已安装")
        print(output)
        return True
    else:
        print("❌ CUDA工具包未安装或不在PATH中")
        
        # 检查常见的CUDA安装路径
        cuda_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "/usr/local/cuda",
            "/opt/cuda"
        ]
        
        for path in cuda_paths:
            if os.path.exists(path):
                print(f"发现CUDA安装路径: {path}")
                break
        else:
            print("未找到CUDA安装路径")
        return False


def check_pytorch():
    """检查PyTorch安装"""
    print_section("PyTorch检查")
    
    try:
        import torch
        print(f"✅ PyTorch已安装，版本: {torch.__version__}")
        
        # 检查CUDA支持
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {'✅ 是' if cuda_available else '❌ 否'}")
        
        if cuda_available:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("\n❌ PyTorch检测不到CUDA支持")
            print("可能的原因:")
            print("1. 安装了CPU版本的PyTorch")
            print("2. CUDA版本不匹配")
            print("3. NVIDIA驱动程序问题")
            
        return cuda_available
        
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_environment():
    """检查环境变量"""
    print_section("环境变量检查")
    
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'PATH', 'LD_LIBRARY_PATH']
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: 未设置")


def provide_solutions():
    """提供解决方案"""
    print_section("解决方案建议")
    
    print("🔧 如果CUDA不可用，请尝试以下解决方案:")
    print()
    
    print("1️⃣ 安装NVIDIA驱动程序:")
    print("   - 访问 https://www.nvidia.com/drivers")
    print("   - 下载并安装适合您显卡的最新驱动")
    print()
    
    print("2️⃣ 安装CUDA工具包:")
    print("   - 访问 https://developer.nvidia.com/cuda-downloads")
    print("   - 下载并安装CUDA工具包（推荐11.8或12.x版本）")
    print()
    
    print("3️⃣ 重新安装PyTorch（CUDA版本）:")
    print("   # 卸载当前PyTorch")
    print("   pip uninstall torch torchvision torchaudio")
    print()
    print("   # 安装CUDA版本的PyTorch")
    print("   # CUDA 11.8:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("   # CUDA 12.1:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    
    print("4️⃣ 验证安装:")
    print("   python -c \"import torch; print(f'CUDA可用: {torch.cuda.is_available()}')\"")
    print()
    
    print("5️⃣ 如果仍然不工作:")
    print("   - 重启计算机")
    print("   - 检查Windows更新")
    print("   - 确保没有冲突的显卡驱动")
    print("   - 在设备管理器中检查显卡状态")


def test_cuda_operations():
    """测试CUDA操作"""
    print_section("CUDA功能测试")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("🧪 测试CUDA操作...")
            
            # 创建张量
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # 矩阵乘法测试
            import time
            start_time = time.time()
            z = torch.mm(x, y)
            end_time = time.time()
            
            print(f"✅ CUDA矩阵乘法测试成功")
            print(f"   计算时间: {end_time - start_time:.4f}秒")
            print(f"   结果形状: {z.shape}")
            print(f"   GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # 清理GPU内存
            del x, y, z
            torch.cuda.empty_cache()
            
        else:
            print("❌ CUDA不可用，无法进行测试")
            
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")


def main():
    print("🚀 CUDA诊断工具")
    print("这个工具将帮助您诊断CUDA配置问题")
    
    # 系统信息
    check_system_info()
    
    # 检查NVIDIA驱动
    driver_ok = check_nvidia_driver()
    
    # 检查CUDA工具包
    cuda_toolkit_ok = check_cuda_toolkit()
    
    # 检查PyTorch
    pytorch_cuda_ok = check_pytorch()
    
    # 检查环境变量
    check_environment()
    
    # 如果CUDA可用，进行功能测试
    if pytorch_cuda_ok:
        test_cuda_operations()
    
    # 提供解决方案
    if not pytorch_cuda_ok:
        provide_solutions()
    
    print_section("诊断完成")
    
    if pytorch_cuda_ok:
        print("🎉 CUDA配置正常！您可以使用GPU加速训练。")
    else:
        print("⚠️  CUDA配置有问题，将使用CPU训练（速度较慢）。")
        print("建议按照上述解决方案修复CUDA配置。")


if __name__ == "__main__":
    main() 