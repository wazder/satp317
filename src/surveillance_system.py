import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import time
import json
import csv
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Add LineLogic path for integration
linelogic_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LineLogic_Shareable_20250731_210934", "src")
if os.path.exists(linelogic_path):
    sys.path.insert(0, linelogic_path)

from .config import config
from .models.yolo_detector import YOLODetector
from .models.sam_segmentor import SAMSegmentor
from .tracking.object_tracker import ObjectTracker
from .utils.roi_manager import ROIManager
from .utils.feature_extractor import FeatureExtractor
from .utils.logger import EventLogger
from .visualization.debug_visualizer import DebugVisualizer

# Import LineLogic components
try:
    import supervision as sv
    from frame_logic import FrameBasedTracker
    from utils import load_model
    LINELOGIC_AVAILABLE = True
except ImportError:
    print("LineLogic components not available, using fallback mode")
    LINELOGIC_AVAILABLE = False

class AirportSurveillanceSystem:
    def __init__(self, input_video_path: str, output_dir: str = "data/output", use_linelogic: bool = True, use_adaptive_system: bool = False, target_accuracy: float = 0.90):
        self.input_video_path = input_video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_linelogic = use_linelogic and LINELOGIC_AVAILABLE
        
        # Initialize components based on mode
        if self.use_linelogic:
            print("🚀 Initializing with LineLogic advanced tracking...")
            self.init_linelogic_components()
        else:
            print("🔧 Initializing with basic surveillance components...")
            self.init_basic_components()
        
        # Video processing setup
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.processed_frames = 0
        
        # LineLogic specific setup
        if self.use_linelogic:
            self.setup_linelogic()
    
    def init_linelogic_components(self):
        """Initialize LineLogic components for advanced tracking."""
        # Load LineLogic YOLO model
        self.linelogic_model = load_model()
        self.linelogic_model.conf = config.models.yolo_confidence
        self.linelogic_model.iou = 0.45
        self.linelogic_model.imgsz = 1024
        
        # Keep SAM for segmentation if needed
        self.sam_segmentor = SAMSegmentor()
        self.feature_extractor = FeatureExtractor()
        self.event_logger = EventLogger(self.output_dir)
        self.debug_visualizer = DebugVisualizer()
        
        # LineLogic tracker will be initialized after video info is available
        self.frame_tracker = None
        
    def init_basic_components(self):
        """Initialize basic surveillance components."""
        self.yolo_detector = YOLODetector()
        self.sam_segmentor = SAMSegmentor()
        self.object_tracker = ObjectTracker()
        self.roi_manager = ROIManager(config.roi.roi_points)
        self.feature_extractor = FeatureExtractor()
        self.event_logger = EventLogger(self.output_dir)
        self.debug_visualizer = DebugVisualizer()
    
    def setup_linelogic(self):
        """Setup LineLogic specific configurations."""
        # Define virtual lines for crossing detection
        self.line_height = 1080  # Adjustable based on video resolution
        self.line_positions = [880, 960, 1040, 1120]  # X coordinates
        self.line_ids = [1, 2, 3, 4]
        
        # Target classes for LineLogic
        self.target_classes = ["person", "backpack", "handbag", "suitcase"]
        
        # CSV logging paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_csv_path = self.output_dir / f"linelogic_log_{timestamp}.csv"
        self.results_csv_path = self.output_dir / f"linelogic_results_{timestamp}.csv"
        
    def initialize_video(self) -> bool:
        """Initialize video capture and output writer."""
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.input_video_path}")
            return False
        
        # Get video properties
        original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {original_fps} FPS, {total_frames} frames")
        
        # Initialize LineLogic frame tracker if using LineLogic
        if self.use_linelogic:
            self.frame_tracker = FrameBasedTracker(
                fps=original_fps,
                min_safe_time=0.5,
                min_uncertain_time=0.28,
                min_very_brief_time=0.17
            )
            
            # Setup lines for LineLogic
            self.line_height = height
            line_points = [sv.Point(x, 0) for x in self.line_positions]
            self.lines = [sv.LineZone(start=p, end=sv.Point(p.x, self.line_height)) for p in line_points]
            self.line_annotators = [
                sv.LineZoneAnnotator(
                    display_in_count=False,
                    display_out_count=False,
                    text_thickness=2,
                    text_scale=1.0
                )
                for _ in self.line_ids
            ]
            
            print(f"🎯 LineLogic initialized with {len(self.lines)} detection lines")
        
        # Setup output video writer
        output_video_path = self.output_dir / "debug_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            config.video.target_fps,
            (width, height)
        )
        
        # Calculate frame skip for FPS reduction
        self.frame_skip = max(1, original_fps // config.video.target_fps)
        
        return True
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a single frame through the complete pipeline."""
        
        if self.use_linelogic:
            return self.process_frame_linelogic(frame, frame_number)
        else:
            return self.process_frame_basic(frame, frame_number)
    
    def process_frame_linelogic(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process frame using LineLogic advanced tracking."""
        # 1. Run YOLO inference with LineLogic model
        results = self.linelogic_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Filter to target classes
        coco_names = self.linelogic_model.model.names
        selected_class_ids = [k for k, v in coco_names.items() if v in self.target_classes]
        detections = detections[np.isin(detections.class_id, selected_class_ids)]
        
        # 3. Track objects with ByteTrack
        detections = self.frame_tracker.byte_tracker.update_with_detections(detections)
        
        # 4. Update object presence for frame-based validation
        self.frame_tracker.update_object_presence(detections, frame_number, coco_names)
        
        # 5. Process line crossings with frame-based logic
        self.frame_tracker.process_line_crossing(detections, frame_number, self.lines, self.line_ids, coco_names)
        
        # 6. Optional: SAM segmentation for enhanced features
        sam_masks, sam_boxes = [], []
        if hasattr(self, 'sam_segmentor') and self.sam_segmentor:
            try:
                sam_masks, sam_boxes = self.sam_segmentor.segment_frame(frame)
            except Exception as e:
                print(f"SAM segmentation failed: {e}")
        
        # 7. Annotate frame with LineLogic style
        debug_frame = frame.copy()
        debug_frame = sv.BoxAnnotator().annotate(debug_frame, detections)
        debug_frame = sv.LabelAnnotator().annotate(debug_frame, detections)
        
        # 8. Annotate lines
        for line, line_annotator in zip(self.lines, self.line_annotators):
            debug_frame = line_annotator.annotate(debug_frame, line)
        
        # 9. Add frame info overlay
        cv2.putText(debug_frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 10. Display current counts
        results_summary = self.frame_tracker.get_results_summary()
        y_offset = 70
        for class_name, counts in results_summary.items():
            text = f"{class_name}: Safe={counts['safe']}, Uncertain={counts['uncertain']}, Total={counts['total']}"
            cv2.putText(debug_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return debug_frame
    
    def process_frame_basic(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process frame using basic surveillance pipeline."""
        # 1. SAM - Full image segmentation
        if self.sam_segmentor:
            sam_masks, sam_boxes = self.sam_segmentor.segment_frame(frame)
        else:
            sam_masks, sam_boxes = [], []
        
        # 2. YOLO Detection
        yolo_detections = self.yolo_detector.detect(frame)
        
        # 3. Cross-check (IoU Matching)
        matched_objects = self.cross_check_detections(yolo_detections, sam_masks, sam_boxes)
        
        # 4. Extract features from matched masks
        for obj in matched_objects:
            if obj['mask'] is not None:
                features = self.feature_extractor.extract_features(frame, obj['mask'])
                obj.update(features)
        
        # 5. Object tracking
        tracked_objects = self.object_tracker.update(matched_objects, frame_number)
        
        # 6. ROI event detection and logging
        roi_events = self.roi_manager.check_roi_crossings(tracked_objects, frame_number)
        if roi_events:
            self.event_logger.log_events(roi_events, frame_number / config.video.target_fps)
        
        # 7. Debug visualization
        debug_frame = self.debug_visualizer.draw_debug_info(
            frame.copy(),
            yolo_detections,
            sam_masks,
            tracked_objects,
            self.roi_manager.roi_polygon,
            roi_events
        )
        
        return debug_frame
    
    def cross_check_detections(self, yolo_detections: List[Dict], 
                             sam_masks: List[np.ndarray], 
                             sam_boxes: List[List[float]]) -> List[Dict]:
        """Match YOLO detections with SAM masks using IoU threshold."""
        matched_objects = []
        
        for yolo_det in yolo_detections:
            yolo_box = yolo_det['bbox']
            best_iou = 0
            best_mask_idx = -1
            
            # Find best matching SAM mask
            for i, sam_box in enumerate(sam_boxes):
                iou = self.calculate_iou(yolo_box, sam_box)
                if iou > best_iou and iou >= config.models.sam_iou_threshold:
                    best_iou = iou
                    best_mask_idx = i
            
            # Create matched object
            matched_obj = {
                'bbox': yolo_box,
                'class': yolo_det['class'],
                'confidence': yolo_det['confidence'],
                'mask': sam_masks[best_mask_idx] if best_mask_idx >= 0 else None,
                'iou_score': best_iou
            }
            matched_objects.append(matched_obj)
        
        return matched_objects
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def run(self) -> bool:
        """Run the complete surveillance pipeline."""
        if not self.initialize_video():
            return False
        
        print("Starting airport surveillance processing...")
        start_time = time.time()
        
        try:
            frame_number = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Skip frames to achieve target FPS
                if frame_number % self.frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Process frame
                debug_frame = self.process_frame(frame, self.processed_frames)
                
                # Write output frame
                self.out.write(debug_frame)
                
                self.processed_frames += 1
                frame_number += 1
                
                # Progress reporting
                if self.processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.processed_frames / elapsed
                    print(f"Processed {self.processed_frames} frames, {fps:.2f} FPS")
            
            # Finalize logging
            self.event_logger.save_logs()
            
            # Save LineLogic results if applicable
            if self.use_linelogic and self.frame_tracker:
                self.save_linelogic_results()
            
            elapsed_time = time.time() - start_time
            print(f"Processing complete! {self.processed_frames} frames in {elapsed_time:.2f}s")
            print(f"Average processing speed: {self.processed_frames/elapsed_time:.2f} FPS")
            
            if self.use_linelogic:
                self.print_linelogic_summary()
            
            return True
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
        
        finally:
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
    
    def save_linelogic_results(self):
        """Save LineLogic tracking results to CSV files."""
        # Export enhanced CSV log
        with open(self.log_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Object ID", "Class", "Line Number", "Direction", "Frame", 
                "Timestamp (min:sec)", "Confidence", "Tracking Duration"
            ])
            writer.writerows(self.frame_tracker.get_log_rows())
        
        print(f"📁 LineLogic log saved to {self.log_csv_path}")
        
        # Export results summary
        results_summary = self.frame_tracker.get_results_summary()
        with open(self.results_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Safe", "Uncertain", "Very Brief", "Total"])
            for class_name, counts in results_summary.items():
                writer.writerow([
                    class_name,
                    counts["safe"],
                    counts["uncertain"], 
                    counts.get("very_brief", 0),
                    counts["total"]
                ])
        
        print(f"📊 LineLogic results saved to {self.results_csv_path}")
    
    def print_linelogic_summary(self):
        """Print LineLogic tracking results summary."""
        print("\n📊 LineLogic Frame-based tracking results:")
        results_summary = self.frame_tracker.get_results_summary()
        for class_name, counts in results_summary.items():
            safe = counts["safe"]
            uncertain = counts["uncertain"]
            total = counts["total"]
            print(f"{class_name:<10} → Safe: {safe}, Uncertain: {uncertain}, Total: {total}")
        
        # Print discarded crossings summary
        discarded = self.frame_tracker.get_discarded_summary()
        print(f"\n🗑️ Discarded crossings (too brief to count): {len(discarded)}")
        if discarded:
            print("Sample discarded crossings (tid, class, line, dir, frame, duration_frames):")
            for row in discarded[:5]:  # Show first 5
                print(f"  {row}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()