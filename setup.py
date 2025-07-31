#!/usr/bin/env python3
"""
Setup script for Airport Surveillance System

This script helps set up the environment and download required model weights.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def install_requirements():
    """Install Python requirements."""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def download_sam_weights():
    """Download SAM model weights."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    sam_weights_file = weights_dir / "sam_vit_h_4b8939.pth"
    
    if sam_weights_file.exists():
        print("✓ SAM weights already exist")
        return True
    
    print("Downloading SAM weights (this may take a while)...")
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    try:
        urllib.request.urlretrieve(sam_url, sam_weights_file)
        print(f"✓ SAM weights downloaded to {sam_weights_file}")
        return True
    except Exception as e:
        print(f"✗ Error downloading SAM weights: {e}")
        print("You can manually download from:")
        print(sam_url)
        return False

def setup_directories():
    """Create necessary directories."""
    dirs = [
        "data/input",
        "data/output", 
        "data/logs",
        "weights"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")

def check_system_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, will use CPU (slower)")
    except ImportError:
        print("⚠ PyTorch not installed yet")
    
    return True

def main():
    print("=== Airport Surveillance System Setup ===\n")
    
    if not check_system_requirements():
        sys.exit(1)
    
    setup_directories()
    
    if not install_requirements():
        print("\nSetup failed. Please install requirements manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("SAM Model Weights")
    print("="*50)
    print("The system uses Segment Anything Model (SAM) for precise segmentation.")
    print("This requires downloading ~2.4GB model weights.")
    
    download_sam = input("Download SAM weights now? (y/n): ").lower().strip()
    if download_sam in ['y', 'yes']:
        download_sam_weights()
    else:
        print("⚠ SAM weights not downloaded. You can:")
        print("1. Run this setup script again")
        print("2. Download manually from: https://github.com/facebookresearch/segment-anything")
        print("3. Run with --disable-sam flag to use YOLO bounding boxes only")
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Place your video file in data/input/")
    print("2. Run the system:")
    print("   python main.py data/input/your_video.mp4")
    print("\nFor help:")
    print("   python main.py --help")
    
    print("\nExample command:")
    print("   python main.py data/input/airport_video.mp4 --fps 15 --confidence 0.6")

if __name__ == "__main__":
    main()