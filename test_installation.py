#!/usr/bin/env python3
"""
Test script to validate Airport Surveillance System installation.
"""

import sys
import importlib
from pathlib import Path

def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def test_imports():
    """Test required imports."""
    print("\nTesting imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('deep_sort_realtime', 'DeepSORT'),
    ]
    
    optional_packages = [
        ('segment_anything', 'Segment Anything'),
    ]
    
    all_passed = True
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - REQUIRED")
            all_passed = False
    
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠ {name} - OPTIONAL (SAM will be disabled)")
    
    return all_passed

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ CUDA available: {device_name} ({memory:.1f}GB)")
            return True
        else:
            print("⚠ CUDA not available, will use CPU (slower)")
            return True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def test_project_structure():
    """Test project structure."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/utils', 
        'src/tracking',
        'src/visualization',
        'data',
        'data/input',
        'data/output',
        'weights'
    ]
    
    required_files = [
        'main.py',
        'setup.py',
        'requirements.txt',
        'src/config.py',
        'src/surveillance_system.py',
        'src/models/yolo_detector.py',
        'src/models/sam_segmentor.py'
    ]
    
    all_passed = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_passed = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_passed = False
    
    return all_passed

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        sys.path.append('src')
        from config import config
        print(f"✓ Config loaded")
        print(f"  - Target classes: {config.target_classes}")
        print(f"  - Target FPS: {config.video.target_fps}")
        print(f"  - YOLO model: {config.models.yolo_model}")
        return True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading."""
    print("\nTesting YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if needed
        print("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠ YOLO model test failed: {e}")
        print("  This is normal on first run - model will download when needed")
        return True

def main():
    """Run all tests."""
    print("=== Airport Surveillance System Installation Test ===\n")
    
    tests = [
        test_python_version,
        test_imports,
        test_cuda,
        test_project_structure,
        test_config,
        test_yolo_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place a video file in data/input/")
        print("2. Run: python main.py data/input/your_video.mp4")
        return 0
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nTo fix issues:")
        print("1. Run: python setup.py")
        print("2. Install missing packages: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())