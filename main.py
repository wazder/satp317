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
        description="Airport Surveillance Object Tracking System with LineLogic Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adaptive System Anything for 90% accuracy (default)
  python main.py video.mp4
  
  # Custom accuracy target with adaptive system
  python main.py video.mp4 --target-accuracy 0.95
  
  # LineLogic mode only (disable adaptive system)
  python main.py video.mp4 --disable-adaptive --linelogic
  
  # Traditional surveillance mode
  python main.py video.mp4 --disable-adaptive --basic-mode
  
  # Performance optimized adaptive system
  python main.py video.mp4 --disable-sam --target-accuracy 0.90
        """
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
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--linelogic",
        action="store_true",
        default=True,
        help="Use LineLogic advanced tracking (default)"
    )
    mode_group.add_argument(
        "--basic-mode",
        action="store_true",
        help="Use basic surveillance mode"
    )
    
    # YOLO parameters
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25 for LineLogic, 0.5 for basic)"
    )
    
    # LineLogic specific parameters
    parser.add_argument(
        "--line-positions",
        type=str,
        default="880,960,1040,1120",
        help="Line positions as comma-separated x coordinates (default: 880,960,1040,1120)"
    )
    parser.add_argument(
        "--min-safe-time",
        type=float,
        default=0.5,
        help="Min safe time for LineLogic (default: 0.5s)"
    )
    parser.add_argument(
        "--min-uncertain-time",
        type=float,
        default=0.28,
        help="Min uncertain time for LineLogic (default: 0.28s)"
    )
    parser.add_argument(
        "--min-very-brief-time",
        type=float,
        default=0.17,
        help="Min very brief time for LineLogic (default: 0.17s)"
    )
    
    # Adaptive System parameters
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.90,
        help="Target accuracy for adaptive system (default: 0.90)"
    )
    parser.add_argument(
        "--disable-adaptive",
        action="store_true",
        help="Disable adaptive system (use basic LineLogic/surveillance)"
    )
    
    # Other options
    parser.add_argument(
        "--disable-sam",
        action="store_true",
        help="Disable SAM segmentation (use bounding boxes only)"
    )
    parser.add_argument(
        "--roi-points",
        type=str,
        help="ROI points as comma-separated coordinates: x1,y1,x2,y2,x3,y3,x4,y4 (basic mode only)"
    )
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    # Determine mode
    use_linelogic = not args.basic_mode
    
    # Update configuration
    config.video.target_fps = args.fps
    config.models.yolo_model = args.yolo_model
    
    # Set confidence threshold based on mode
    if args.confidence == 0.25 and args.basic_mode:
        # User didn't specify confidence and using basic mode, use basic default
        config.models.yolo_confidence = 0.5
    elif args.confidence == 0.5 and use_linelogic:
        # User didn't specify confidence and using linelogic, use linelogic default
        config.models.yolo_confidence = 0.25
    else:
        # User specified confidence explicitly
        config.models.yolo_confidence = args.confidence
    
    # Parse LineLogic line positions
    line_positions = None
    if use_linelogic:
        try:
            line_positions = list(map(int, args.line_positions.split(',')))
            print(f"Using line positions: {line_positions}")
        except ValueError as e:
            print(f"Error parsing line positions: {e}")
            return 1
    
    # Parse ROI points if provided (basic mode only)
    if args.roi_points and not use_linelogic:
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
    elif args.roi_points and use_linelogic:
        print("Warning: ROI points ignored in LineLogic mode (uses line detection instead)")
    
    # Print configuration
    if not args.disable_adaptive:
        mode_name = f"Adaptive System Anything (Target: {args.target_accuracy:.0%})"
    elif use_linelogic:
        mode_name = "LineLogic Advanced Tracking"
    else:
        mode_name = "Basic Surveillance"
    
    print(f"=== Airport Surveillance System - {mode_name} ===")
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target FPS: {config.video.target_fps}")
    print(f"YOLO model: {config.models.yolo_model}")
    print(f"Confidence threshold: {config.models.yolo_confidence}")
    print(f"Target classes: {config.target_classes}")
    
    if not args.disable_adaptive:
        print(f"🎯 Target accuracy: {args.target_accuracy:.0%}")
        print(f"🤖 Ensemble models: Multiple YOLO variants")
        print(f"🧠 Adaptive thresholds: Dynamic optimization")
        print(f"⏱️ Temporal validation: Multi-frame consistency")
    elif use_linelogic:
        print(f"Line positions: {line_positions}")
        print(f"Min safe time: {args.min_safe_time}s")
        print(f"Min uncertain time: {args.min_uncertain_time}s")
        print(f"Min very brief time: {args.min_very_brief_time}s")
    else:
        print(f"ROI points: {config.roi.roi_points}")
    
    print(f"SAM segmentation: {'Disabled' if args.disable_sam else 'Enabled'}")
    print()
    
    # Initialize and run surveillance system
    try:
        surveillance = AirportSurveillanceSystem(
            input_video_path=args.input_video,
            output_dir=args.output_dir,
            use_linelogic=use_linelogic,
            use_adaptive_system=not args.disable_adaptive,  # Enable by default unless disabled
            target_accuracy=args.target_accuracy
        )
        
        # Configure LineLogic parameters if using LineLogic mode
        if use_linelogic and hasattr(surveillance, 'line_positions'):
            surveillance.line_positions = line_positions
            # Store timing parameters for later use
            surveillance.linelogic_params = {
                'min_safe_time': args.min_safe_time,
                'min_uncertain_time': args.min_uncertain_time,
                'min_very_brief_time': args.min_very_brief_time
            }
        
        # Disable SAM if requested
        if args.disable_sam:
            print("Warning: SAM segmentation disabled, using YOLO bounding boxes only")
            surveillance.sam_segmentor = None
        
        # Run the system
        success = surveillance.run()
        
        if success:
            print("\n=== Processing Complete ===")
            print(f"Debug video saved to: {args.output_dir}/debug_output.mp4")
            
            if use_linelogic:
                print(f"LineLogic logs saved to: {args.output_dir}/")
                print("\nLineLogic files generated:")
            else:
                print(f"Event logs saved to: {args.output_dir}/")
                print("\nFiles generated:")
            
            output_path = Path(args.output_dir)
            for file in sorted(output_path.glob("*")):
                if file.is_file():
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