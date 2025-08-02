# Vast.ai Deployment Tips

## 🎯 Instance Selection
- **Minimum**: RTX 3060 (6GB VRAM) + 16GB RAM
- **Recommended**: RTX 4090 (24GB VRAM) + 32GB RAM  
- **Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- **Storage**: 50GB+ (models + videos + outputs)

## ⚡ Performance Optimization

### GPU Memory Management
```bash
# Check GPU memory before running
nvidia-smi

# If OOM errors occur, reduce batch processing:
python main.py video.mp4 --fps 10 --disable-sam
```

### Network Optimization
```bash
# Use local downloads for better speed
wget YOUR_VIDEO_URL -O data/input/video.mp4

# Compress output if uploading results
cd data/output && tar -czf results.tar.gz *.mp4 *.csv
```

## 📊 Monitoring Commands

```bash
# GPU monitoring (run in separate terminal)
watch -n 1 nvidia-smi

# Disk space monitoring  
df -h

# Process monitoring
htop

# Network monitoring (if downloading large files)
nethogs
```

## 🔧 Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```bash
# Solution: Reduce processing load
python main.py video.mp4 --fps 8 --disable-sam --basic-mode
```

**2. Slow Processing**
```bash
# Check if using CPU instead of GPU
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
```

**3. Storage Full**
```bash
# Clean up old outputs
rm -rf data/output/*
rm -rf ~/.cache/torch/hub/*
```

**4. Network Issues**
```bash
# Use wget instead of git clone if github is slow
wget https://github.com/wazder/satp317/archive/main.zip
unzip main.zip && mv satp317-main satp317
```

## 💡 Cost Optimization

### Hourly Cost Reduction:
- **Pause instance** when not processing
- **Use preemptible instances** for long jobs
- **Download results frequently** to avoid loss
- **Use smaller YOLO models** (yolov8n.pt vs yolov8x.pt)

### Efficient Workflow:
```bash
# 1. Setup (5 min)
bash deploy_vastai.sh

# 2. Upload your video
# Via Jupyter: drag & drop to data/input/

# 3. Process (varies by video length)
python main.py data/input/your_video.mp4 --fps 15

# 4. Download results  
# Via Jupyter: download data/output/ folder

# 5. Pause/terminate instance
```

## 🎮 Gaming GPU vs Professional

**Gaming GPUs (RTX series) - Recommended:**
- ✅ Great price/performance 
- ✅ Large VRAM (RTX 4090: 24GB)
- ✅ Good for inference

**Professional GPUs (Tesla/Quadro):**
- ⚠️ More expensive
- ⚠️ Often older architecture
- ✅ Better for training (not needed here)

## 📈 Expected Performance

| GPU | Video Processing | Cost/Hour | Recommended |
|-----|------------------|-----------|-------------|
| RTX 4090 | 10-20 FPS | $0.3-0.6 | ⭐⭐⭐ Best |
| RTX 3090 | 8-15 FPS | $0.2-0.4 | ⭐⭐⭐ Great |
| RTX 3080 | 5-10 FPS | $0.15-0.3 | ⭐⭐ Good |
| RTX 3060 | 3-6 FPS | $0.1-0.2 | ⭐ Budget |

*Processing speed depends on video resolution and settings*