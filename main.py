#!/usr/bin/env python3
"""
Airport Surveillance Object Tracking and Logging System - Debug Mode

This script runs the complete surveillance pipeline with debug visualization.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.surveillance_system import AirportSurveillanceSystem
from src.config import config

def main():
    parser = argparse.ArgumentParser(
        description="Airport Surveillance Object Tracking System - Debug Mode"
    )
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Output directory for results (default: data/output)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target processing FPS (default: 15)"
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--disable-sam",
        action="store_true",
        help="Disable SAM segmentation (use bounding boxes only)"
    )
    parser.add_argument(
        "--roi-points",
        type=str,
        help="ROI points as comma-separated coordinates: x1,y1,x2,y2,x3,y3,x4,y4"
    )
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    # Update configuration
    config.video.target_fps = args.fps
    config.models.yolo_model = args.yolo_model
    config.models.yolo_confidence = args.confidence
    
    # Parse ROI points if provided
    if args.roi_points:
        try:
            coords = list(map(int, args.roi_points.split(',')))
            if len(coords) % 2 != 0:
                raise ValueError("ROI points must be pairs of coordinates")
            
            roi_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            config.roi.roi_points = roi_points
            print(f"Using custom ROI with {len(roi_points)} points")
        except ValueError as e:
            print(f"Error parsing ROI points: {e}")
            return 1
    
    # Print configuration
    print("=== Airport Surveillance System - Debug Mode ===")
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target FPS: {config.video.target_fps}")
    print(f"YOLO model: {config.models.yolo_model}")
    print(f"Confidence threshold: {config.models.yolo_confidence}")
    print(f"Target classes: {config.target_classes}")
    print(f"ROI points: {config.roi.roi_points}")
    print(f"SAM segmentation: {'Disabled' if args.disable_sam else 'Enabled'}")
    print()
    
    # Initialize and run surveillance system
    try:
        surveillance = AirportSurveillanceSystem(
            input_video_path=args.input_video,
            output_dir=args.output_dir
        )
        
        # Disable SAM if requested
        if args.disable_sam:
            print("Warning: SAM segmentation disabled, using YOLO bounding boxes only")
            surveillance.sam_segmentor = None
        
        # Run the system
        success = surveillance.run()
        
        if success:
            print("\n=== Processing Complete ===")
            print(f"Debug video saved to: {args.output_dir}/debug_output.mp4")
            print(f"Event logs saved to: {args.output_dir}/")
            print("\nFiles generated:")
            output_path = Path(args.output_dir)
            for file in output_path.glob("*"):
                print(f"  - {file.name}")
            return 0
        else:
            print("Error: Processing failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if 'surveillance' in locals():
            surveillance.cleanup()

if __name__ == "__main__":
    sys.exit(main())