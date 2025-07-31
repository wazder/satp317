import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import time
import json
import csv
from pathlib import Path

from .config import config
from .models.yolo_detector import YOLODetector
from .models.sam_segmentor import SAMSegmentor
from .tracking.object_tracker import ObjectTracker
from .utils.roi_manager import ROIManager
from .utils.feature_extractor import FeatureExtractor
from .utils.logger import EventLogger
from .visualization.debug_visualizer import DebugVisualizer

class AirportSurveillanceSystem:
    def __init__(self, input_video_path: str, output_dir: str = "data/output"):
        self.input_video_path = input_video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.yolo_detector = YOLODetector()
        self.sam_segmentor = SAMSegmentor()
        self.object_tracker = ObjectTracker()
        self.roi_manager = ROIManager(config.roi.roi_points)
        self.feature_extractor = FeatureExtractor()
        self.event_logger = EventLogger(self.output_dir)
        self.debug_visualizer = DebugVisualizer()
        
        # Video processing setup
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.processed_frames = 0
        
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
        
        # 1. SAM - Full image segmentation
        sam_masks, sam_boxes = self.sam_segmentor.segment_frame(frame)
        
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
            
            elapsed_time = time.time() - start_time
            print(f"Processing complete! {self.processed_frames} frames in {elapsed_time:.2f}s")
            print(f"Average processing speed: {self.processed_frames/elapsed_time:.2f} FPS")
            
            return True
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
        
        finally:
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()