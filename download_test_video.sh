#!/bin/bash
# Download test video from Google Drive - Standalone script

echo "📹 Downloading test video from Google Drive..."

# Create input directory if it doesn't exist
mkdir -p data/input

# Google Drive file ID from the shared link
FILE_ID="1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN"

# Download the file
echo "Downloading surveillance test video..."
wget --no-check-certificate \
     --progress=bar:force:noscroll \
     "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
     -O data/input/test_video.mp4

# Check if download was successful
if [ -f "data/input/test_video.mp4" ]; then
    echo "✅ Test video downloaded successfully"
    
    # Get file size (cross-platform)
    if command -v stat >/dev/null 2>&1; then
        FILE_SIZE=$(stat -f%z "data/input/test_video.mp4" 2>/dev/null || stat -c%s "data/input/test_video.mp4" 2>/dev/null || echo "0")
        echo "   File size: $(($FILE_SIZE / 1024 / 1024)) MB"
    fi
    
    # Get video info if ffprobe is available
    if command -v ffprobe >/dev/null 2>&1; then
        echo "   Video info:"
        ffprobe -v quiet -print_format json -show_format -show_streams "data/input/test_video.mp4" | \
        python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            width = stream.get('width', 'Unknown')
            height = stream.get('height', 'Unknown')
            fps = eval(stream.get('r_frame_rate', '0/1'))
            duration = float(data.get('format', {}).get('duration', 0))
            print(f'   Resolution: {width}x{height}')
            print(f'   FPS: {fps:.2f}')
            print(f'   Duration: {duration:.1f}s')
            break
except:
    print('   Could not parse video info')
"
    fi
    
    echo ""
    echo "🎯 Ready to process! Run:"
    echo "   python main.py data/input/test_video.mp4 --fps 15"
    
else
    echo "❌ Download failed!"
    echo ""
    echo "💡 Alternative methods:"
    echo "1. Manual download:"
    echo "   Visit: https://drive.google.com/file/d/1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN/view"
    echo "   Download and save as: data/input/test_video.mp4"
    echo ""
    echo "2. Using gdown (if installed):"
    echo "   pip install gdown"
    echo "   gdown 1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN -O data/input/test_video.mp4"
    echo ""
    echo "3. Using curl:"
    echo "   curl -L 'https://drive.google.com/uc?export=download&id=1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN' -o data/input/test_video.mp4"
fi