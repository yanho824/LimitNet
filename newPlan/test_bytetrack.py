#!/usr/bin/env python3
"""
测试ByteTrack检测追踪功能
"""

import cv2
import numpy as np
import torch
from detection_tracker import DetectionTracker, LimitNetDetectionTracker

def test_basic_detection():
    """测试基本的检测功能"""
    print("🔧 测试基本检测功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # 初始化检测器
        detector = DetectionTracker(
            model_path='yolo11n.pt',
            conf_threshold=0.5,
            device='cpu'  # 使用CPU避免GPU问题
        )
        
        # 进行检测
        priority_mask, track_info = detector.detect_and_track(test_image)
        
        print(f"✅ 检测成功!")
        print(f"   优先级掩码尺寸: {priority_mask.shape}")
        print(f"   检测到 {len(track_info)} 个目标")
        
        for i, info in enumerate(track_info):
            print(f"   目标 {i}: ID={info['id']}, 置信度={info['conf']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        return False

def test_limitnet_integration():
    """测试与LimitNet的集成"""
    print("\n🔧 测试LimitNet集成...")
    
    try:
        # 配置
        detector_config = {
            'model_path': 'yolo11n.pt',
            'conf_threshold': 0.3,
            'device': 'cpu',
            'frame_rate': 30
        }
        
        # 创建LimitNet检测追踪器
        limitnet_tracker = LimitNetDetectionTracker(
            detector_config=detector_config,
            priority_boost=1.5,
            background_priority=0.1
        )
        
        # 创建模拟的编码特征和原始图像
        batch_size = 2
        encoded_features = torch.randn(batch_size, 12, 28, 28)  # LimitNet编码特征
        original_images = torch.randn(batch_size, 3, 224, 224)  # 原始图像
        
        # 生成优先级图
        priority_map = limitnet_tracker.generate_priority_map(encoded_features, original_images)
        
        print(f"✅ LimitNet集成成功!")
        print(f"   编码特征尺寸: {encoded_features.shape}")
        print(f"   优先级图尺寸: {priority_map.shape}")
        print(f"   优先级值范围: {priority_map.min().item():.3f} - {priority_map.max().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LimitNet集成失败: {e}")
        return False

def test_video_tracking():
    """测试视频追踪功能"""
    print("\n🔧 测试视频追踪功能...")
    
    try:
        detector = DetectionTracker(
            model_path='yolo11n.pt',
            conf_threshold=0.5,
            device='cpu',
            frame_rate=30
        )
        
        # 创建几个连续的测试帧
        num_frames = 5
        for i in range(num_frames):
            # 创建测试帧（模拟移动的目标）
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 添加一个简单的"目标"（白色矩形）
            x = 100 + i * 20  # 模拟移动
            y = 100
            cv2.rectangle(test_frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
            
            # 检测和追踪
            priority_mask, track_info = detector.detect_and_track(test_frame)
            
            print(f"   帧 {i}: 检测到 {len(track_info)} 个目标")
            
        print("✅ 视频追踪测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 视频追踪失败: {e}")
        return False

def test_bytetrack_import():
    """测试ByteTrack导入"""
    print("🔧 测试ByteTrack导入...")
    
    try:
        from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
        print("✅ ByteTrack导入成功!")
        
        # 尝试创建ByteTrack实例
        tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        print("✅ ByteTrack实例化成功!")
        return True
        
    except ImportError as e:
        print(f"⚠️ ByteTrack导入失败: {e}")
        print("   将使用简单追踪器作为后备方案")
        return False
    except Exception as e:
        print(f"⚠️ ByteTrack实例化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始ByteTrack功能测试...")
    print("=" * 50)
    
    # 测试结果
    results = []
    
    # 1. 测试ByteTrack导入
    results.append(("ByteTrack导入", test_bytetrack_import()))
    
    # 2. 测试基本检测
    results.append(("基本检测", test_basic_detection()))
    
    # 3. 测试视频追踪
    results.append(("视频追踪", test_video_tracking()))
    
    # 4. 测试LimitNet集成
    results.append(("LimitNet集成", test_limitnet_integration()))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15s}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！ByteTrack配置成功！")
    else:
        print("⚠️ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main() 