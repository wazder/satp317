import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import random

from ..config import config

class DebugVisualizer:
    def __init__(self):
        self.colors = self._generate_colors(50)  # Generate 50 random colors for tracks
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.visualization.font_scale
        self.font_thickness = config.visualization.font_thickness
        self.mask_alpha = config.visualization.mask_alpha
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate random colors for visualization."""
        colors = []
        for _ in range(num_colors):
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            colors.append(color)
        return colors
    
    def draw_debug_info(self, frame: np.ndarray, 
                       yolo_detections: List[Dict],
                       sam_masks: List[np.ndarray],
                       tracked_objects: List[Dict],
                       roi_polygon: List[Tuple[int, int]],
                       roi_events: List[Dict]) -> np.ndarray:
        """
        Draw comprehensive debug information on frame.
        
        Args:
            frame: Input frame
            yolo_detections: YOLO detection results
            sam_masks: SAM segmentation masks
            tracked_objects: Tracked objects with IDs
            roi_polygon: ROI polygon points
            roi_events: ROI crossing events
            
        Returns:
            Frame with debug visualizations
        """
        debug_frame = frame.copy()
        
        # 1. Draw ROI
        if config.visualization.show_roi:
            debug_frame = self._draw_roi(debug_frame, roi_polygon)
        
        # 2. Draw SAM masks
        if config.visualization.show_sam_masks and sam_masks:
            debug_frame = self._draw_sam_masks(debug_frame, sam_masks)
        
        # 3. Draw YOLO detections (if not using tracking)
        if config.visualization.show_yolo_boxes and not tracked_objects:
            debug_frame = self._draw_yolo_detections(debug_frame, yolo_detections)
        
        # 4. Draw tracked objects
        if config.visualization.show_tracking_ids and tracked_objects:
            debug_frame = self._draw_tracked_objects(debug_frame, tracked_objects)
        
        # 5. Draw ROI events
        if roi_events:
            debug_frame = self._draw_roi_events(debug_frame, roi_events)
        
        # 6. Draw info panel
        debug_frame = self._draw_info_panel(debug_frame, yolo_detections, 
                                           tracked_objects, roi_events)
        
        return debug_frame
    
    def _draw_roi(self, frame: np.ndarray, roi_polygon: List[Tuple[int, int]]) -> np.ndarray:
        """Draw ROI polygon on frame."""
        if not roi_polygon:
            return frame
        
        # Draw ROI polygon
        roi_points = np.array(roi_polygon, np.int32)
        cv2.polylines(frame, [roi_points], True, config.roi.color, config.roi.line_thickness)
        
        # Fill with semi-transparent color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], config.roi.color)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Add ROI label
        cv2.putText(frame, "ROI", 
                   (roi_polygon[0][0], roi_polygon[0][1] - 10),
                   self.font, 0.7, config.roi.color, 2)
        
        return frame
    
    def _draw_sam_masks(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Draw SAM masks with random colors."""
        if not masks:
            return frame
        
        overlay = frame.copy()
        
        for i, mask in enumerate(masks):
            if mask is not None:
                color = self.colors[i % len(self.colors)]
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = color
                
                # Blend with overlay
                cv2.addWeighted(overlay, 1 - self.mask_alpha, 
                               colored_mask, self.mask_alpha, 0, overlay)
                
                # Draw contours
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 1)
        
        return overlay
    
    def _draw_yolo_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detection boxes."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                       self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        
        return frame
    
    def _draw_tracked_objects(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """Draw tracked objects with IDs and features."""
        for obj in tracked_objects:
            track_id = obj['track_id']
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            features = obj.get('features', {})
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this track
            color = self.colors[hash(str(track_id)) % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label with ID and class
            label = f"ID:{track_id} {class_name}"
            if confidence:
                label += f" {confidence:.2f}"
            
            # Add feature information if available
            if config.visualization.show_object_info and features:
                feature_text = self._format_features(features)
                if feature_text:
                    label += f" | {feature_text}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                       self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        return frame
    
    def _format_features(self, features: Dict) -> str:
        """Format features for display."""
        parts = []
        
        # Color information
        color_info = features.get('dominant_color', {})
        if color_info.get('color_name'):
            parts.append(f"Color:{color_info['color_name']}")
        
        # Size information
        size_cat = features.get('size_category')
        if size_cat:
            parts.append(f"Size:{size_cat}")
        
        return " ".join(parts)
    
    def _draw_roi_events(self, frame: np.ndarray, roi_events: List[Dict]) -> np.ndarray:
        """Draw ROI crossing events."""
        for event in roi_events:
            center = event['center']
            event_type = event['event_type']
            track_id = event['track_id']
            
            # Draw event indicator
            color = (0, 255, 0) if event_type == "ENTER" else (0, 0, 255)
            cv2.circle(frame, (int(center[0]), int(center[1])), 15, color, 3)
            
            # Draw event text
            event_text = f"{event_type}"
            cv2.putText(frame, event_text, 
                       (int(center[0]) - 30, int(center[1]) - 20),
                       self.font, 0.6, color, 2)
        
        return frame
    
    def _draw_info_panel(self, frame: np.ndarray, 
                        yolo_detections: List[Dict],
                        tracked_objects: List[Dict],
                        roi_events: List[Dict]) -> np.ndarray:
        """Draw information panel with statistics."""
        height, width = frame.shape[:2]
        
        # Create info panel background
        panel_height = 120
        panel_width = 300
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Add statistics text
        y_offset = panel_y + 25
        line_height = 20
        
        info_lines = [
            f"YOLO Detections: {len(yolo_detections)}",
            f"Tracked Objects: {len(tracked_objects)}",
            f"ROI Events: {len(roi_events)}",
            f"FPS Target: {config.video.target_fps}",
            f"Classes: {', '.join(config.target_classes)}"
        ]
        
        for line in info_lines:
            cv2.putText(frame, line, (panel_x + 10, y_offset), 
                       self.font, 0.4, (255, 255, 255), 1)
            y_offset += line_height
        
        return frame
    
    def create_comparison_view(self, original_frame: np.ndarray,
                              debug_frame: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison view."""
        height, width = original_frame.shape[:2]
        
        # Resize frames to fit side by side
        new_width = width // 2
        new_height = height
        
        original_resized = cv2.resize(original_frame, (new_width, new_height))
        debug_resized = cv2.resize(debug_frame, (new_width, new_height))
        
        # Combine frames
        combined = np.hstack([original_resized, debug_resized])
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Debug View", (new_width + 10, 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        
        return combined