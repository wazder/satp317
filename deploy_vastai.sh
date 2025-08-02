#!/bin/bash
# Vast.ai Deployment Script for Airport Surveillance System

echo "🚀 Setting up Airport Surveillance System on Vast.ai..."

# Check if running in correct environment
if [ ! -f "/etc/hostname" ]; then
    echo "⚠️  Warning: This script is designed for Linux environments"
fi

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y > /dev/null 2>&1

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y wget curl git ffmpeg > /dev/null 2>&1

# Setup project
echo "📂 Setting up project..."
if [ ! -d "satp317" ]; then
    git clone https://github.com/wazder/satp317.git
fi
cd satp317

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/{input,output,logs} weights

# Download test video from Google Drive
echo "📹 Downloading test video from Google Drive..."
if [ ! -f "data/input/test_video.mp4" ]; then
    echo "Downloading surveillance test video..."
    # Convert Google Drive share link to direct download link
    FILE_ID="1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN"
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O data/input/test_video.mp4
    
    if [ -f "data/input/test_video.mp4" ]; then
        echo "✅ Test video downloaded successfully"
        # Check file size
        FILE_SIZE=$(stat -f%z "data/input/test_video.mp4" 2>/dev/null || stat -c%s "data/input/test_video.mp4" 2>/dev/null || echo "0")
        echo "   File size: $(($FILE_SIZE / 1024 / 1024)) MB"
    else
        echo "⚠️  Video download failed - you can upload manually to data/input/"
    fi
fi

# GPU Check
echo "🎮 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️  CUDA not available - will use CPU (very slow)')
"

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
sys.path.append('src')
try:
    from src.config import config
    print('✅ Config loaded successfully')
    print(f'   Target classes: {config.target_classes}')
    print(f'   YOLO model: {config.models.yolo_model}')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "🎯 Setup complete! Ready to run Airport Surveillance System!"
echo ""
echo "Quick start commands:"
echo ""
echo "💡 For RTX 4090/3090 (High performance):"
echo "  python main.py data/input/test_video.mp4 --fps 30 --confidence 0.3"
echo ""
echo "⚡ For RTX 3080/lower (Balanced):"
echo "  python main.py data/input/test_video.mp4 --fps 15 --disable-sam --confidence 0.5"
echo ""
echo "🔥 For maximum speed (CPU fallback):"
echo "  python main.py data/input/test_video.mp4 --fps 10 --disable-sam --basic-mode"
echo ""
echo "📊 Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "📁 Results will be in: data/output/"
echo "   - debug_output.mp4 (processed video)"
echo "   - *.csv files (detection logs)"
echo ""
echo "🔄 If video download failed, run:"
echo "  bash download_test_video.sh"