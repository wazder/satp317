# Airport Surveillance Object Tracking and Logging System

A comprehensive offline video analytics system for airport surveillance that detects, tracks, and logs target objects (person, backpack, suitcase, handbag) with advanced computer vision techniques.

## 🎯 Features

### Core Functionality
- **Multi-Object Detection**: YOLO-based detection of 4 target classes
- **Precise Segmentation**: SAM (Segment Anything Model) for instance-level masks
- **Object Tracking**: DeepSORT/ByteTrack for consistent ID assignment
- **ROI Monitoring**: Configurable Region of Interest with entry/exit logging
- **Feature Extraction**: Dominant color and size analysis from segmentation masks
- **Debug Visualization**: Comprehensive visual debugging with overlays

### Pipeline Overview
1. **Frame Input** - Process 1080p video at configurable FPS
2. **SAM Segmentation** - Generate instance masks for all objects
3. **YOLO Detection** - Detect and classify target objects
4. **Cross-Check Matching** - IoU-based matching between YOLO and SAM
5. **Feature Extraction** - Color and size analysis from matched masks
6. **Object Tracking** - Assign consistent IDs across frames
7. **ROI Event Detection** - Log entry/exit events
8. **Debug Visualization** - Output annotated video with full debug info

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (RTX 4090 or similar recommended)
- 8GB+ GPU VRAM for optimal performance

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd satp317
   ```

2. **Run automated setup**
   ```bash
   python setup.py
   ```
   This will:
   - Install all Python dependencies
   - Create necessary directories
   - Optionally download SAM model weights (~2.4GB)

3. **Manual installation (alternative)**
   ```bash
   pip install -r requirements.txt
   mkdir -p data/{input,output,logs} weights
   ```

### Basic Usage

1. **Get a test video**
   ```bash
   # Download the provided test video
   wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN' -O data/input/test_video.mp4
   
   # OR place your own video
   cp your_video.mp4 data/input/
   ```

2. **Run the surveillance system**
   ```bash
   # With test video
   python main.py data/input/test_video.mp4
   
   # With your own video
   python main.py data/input/your_video.mp4
   ```

3. **View results**
   - Debug video: `data/output/debug_output.mp4`
   - Event logs: `data/output/roi_events.csv` and `roi_events.json`
   - Summary: `data/output/event_summary.json`

## 📖 Advanced Usage

### Command Line Options

```bash
python main.py [input_video] [options]

Options:
  --output-dir DIR          Output directory (default: data/output)
  --fps FPS                 Target processing FPS (default: 15)
  --yolo-model MODEL        YOLO model file (default: yolov8n.pt)
  --confidence THRESHOLD    Detection confidence (default: 0.5)
  --disable-sam            Disable SAM segmentation
  --roi-points COORDS      Custom ROI coordinates
```

### Examples

**High-quality processing:**
```bash
python main.py data/input/test_video.mp4 --fps 30 --yolo-model yolov8l.pt --confidence 0.7
```

**Fast processing (CPU-friendly):**
```bash
python main.py data/input/test_video.mp4 --fps 10 --disable-sam --confidence 0.4
```

**Custom ROI:**
```bash
python main.py data/input/test_video.mp4 --roi-points "100,200,800,200,800,600,100,600"
```

### Configuration

Edit `src/config.py` to customize:

- **Target Classes**: Modify `target_classes` and `target_class_ids`
- **ROI Settings**: Default ROI polygon, colors, thickness
- **Tracking Parameters**: Max age, IoU thresholds
- **Visualization**: Enable/disable overlays, colors, transparency

## 📊 Output Files

### Debug Video (`debug_output.mp4`)
Annotated video showing:
- YOLO bounding boxes with confidence scores
- SAM segmentation masks (semi-transparent)
- Object tracking IDs and trajectories
- ROI polygon and crossing events
- Real-time statistics panel
- Feature information (color, size)

### Event Logs
**CSV Format** (`roi_events.csv`):
```csv
timestamp,frame_number,track_id,event_type,object_class,confidence,center_x,center_y,bbox_x1,bbox_y1,bbox_x2,bbox_y2,dominant_color,color_rgb,color_confidence,pixel_area,width,height,aspect_ratio,size_category,perimeter,compactness
```

**JSON Format** (`roi_events.json`):
```json
{
  "metadata": {
    "total_events": 45,
    "generated_at": "2024-01-15T10:30:00",
    "config": {...}
  },
  "events": [...]
}
```

### Summary Report (`event_summary.json`)
Statistical overview including:
- Total events by type (ENTER/EXIT)
- Object class distribution
- Unique object count
- Average confidence scores
- Processing duration

## 🏗️ Architecture

### Project Structure
```
satp317/
├── src/
│   ├── models/
│   │   ├── yolo_detector.py      # YOLO detection
│   │   └── sam_segmentor.py      # SAM segmentation
│   ├── tracking/
│   │   └── object_tracker.py     # DeepSORT/ByteTrack
│   ├── utils/
│   │   ├── roi_manager.py        # ROI logic
│   │   ├── feature_extractor.py  # Color/size analysis
│   │   └── logger.py             # Event logging
│   ├── visualization/
│   │   └── debug_visualizer.py   # Debug overlays
│   ├── config.py                 # Configuration
│   └── surveillance_system.py   # Main system
├── data/
│   ├── input/                    # Input videos
│   ├── output/                   # Results
│   └── logs/                     # Log files
├── weights/                      # Model weights
├── main.py                       # Entry point
├── setup.py                      # Setup script
└── requirements.txt              # Dependencies
```

### Key Components

1. **YOLODetector** (`src/models/yolo_detector.py`)
   - Uses Ultralytics YOLO for object detection
   - Filters for target classes only
   - Configurable confidence thresholds

2. **SAMSegmentor** (`src/models/sam_segmentor.py`)  
   - Facebook's Segment Anything Model
   - Generates precise instance masks
   - Fallback to bounding boxes if unavailable

3. **ObjectTracker** (`src/tracking/object_tracker.py`)
   - DeepSORT for robust multi-object tracking
   - Maintains object identities across frames
   - Handles occlusions and re-identification

4. **ROIManager** (`src/utils/roi_manager.py`)
   - Configurable polygon-based ROI
   - Entry/exit event detection
   - State management for tracked objects

5. **FeatureExtractor** (`src/utils/feature_extractor.py`)
   - Dominant color extraction using K-means
   - Size and shape analysis
   - Color name mapping

## ⚡ Performance

### Expected Processing Speed
- **RTX 4090**: 3-5 FPS (with SAM)
- **RTX 3080**: 2-3 FPS (with SAM) 
- **CPU Only**: 0.5-1 FPS (SAM disabled)

### Memory Requirements
- **GPU VRAM**: 8GB+ recommended
- **System RAM**: 16GB+ recommended
- **Storage**: ~1GB per minute of output video

### Optimization Tips
1. Use `--fps 10` for faster processing
2. Use `--disable-sam` for CPU-only systems
3. Use smaller YOLO models (`yolov8n.pt`) for speed
4. Reduce video resolution before processing

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
python main.py video.mp4 --disable-sam
```

**SAM Weights Not Found**
```bash
# Run setup again or download manually
python setup.py
```

**Tracking Failures**
```bash
# Adjust tracking parameters in config.py
tracking.max_age = 50
tracking.min_hits = 5
```

**Poor Detection Quality**
```bash
# Increase confidence threshold
python main.py video.mp4 --confidence 0.7
```

### Debug Mode Benefits

This implementation focuses on debug mode to provide:
- **Visual Verification**: See exactly what the system detects
- **Parameter Tuning**: Adjust thresholds based on visual feedback
- **Quality Assurance**: Validate before production deployment
- **Feature Development**: Test new capabilities with immediate feedback

## 🔮 Future Enhancements

### Planned Features
- Real-time video stream processing
- Web-based configuration interface
- Advanced behavior analysis
- Multi-camera synchronization
- Cloud deployment support

### Production Mode
After debug validation, the system can be streamlined to:
- Remove visualization overhead
- Optimize for processing speed
- Add real-time streaming capabilities
- Implement alert systems

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 🖥️ Running on Vast.ai (Cloud GPU)

### Quick Cloud Setup:

1. **Launch Vast.ai instance** with RTX 4090/3090 (8GB+ VRAM)
2. **Access via Jupyter Lab** (use the provided HTTP link)
3. **Open terminal in Jupyter** and run:

```bash
# Clone project
git clone https://github.com/wazder/satp317.git
cd satp317

# Install dependencies
pip install -r requirements.txt
python setup.py

# Download test video
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN' -O data/input/test_video.mp4

# Run surveillance system
python main.py data/input/test_video.mp4 --fps 15 --confidence 0.6

# Monitor progress (in another terminal)
watch -n 1 nvidia-smi
```

4. **Download results** via Jupyter file browser:
   - `data/output/debug_output.mp4` (annotated video)
   - `data/output/roi_events.csv` (event logs)

### Performance Tips for Cloud:
- **RTX 4090**: Use `--fps 30 --yolo-model yolov8l.pt`
- **RTX 3080**: Use `--fps 15 --yolo-model yolov8m.pt`  
- **Budget mode**: Use `--fps 10 --disable-sam`

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with:
   - System specifications
   - Error messages
   - Sample video (if possible)
   - Configuration used

---

**Built for airport security and surveillance applications with enterprise-grade accuracy and performance.**
