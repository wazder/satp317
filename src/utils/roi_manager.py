import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from ..config import config

class ROIManager:
    def __init__(self, roi_points: List[Tuple[int, int]]):
        self.roi_points = roi_points
        self.roi_polygon = Polygon(roi_points)
        self.object_states = {}  # Track which objects are inside/outside ROI
        
        print(f"ROI initialized with {len(roi_points)} points")
        print(f"ROI area: {self.roi_polygon.area:.0f} pixels")
    
    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the ROI."""
        try:
            return self.roi_polygon.contains(Point(point))
        except:
            # Fallback to OpenCV method if Shapely fails
            return cv2.pointPolygonTest(
                np.array(self.roi_points, np.int32), point, False
            ) >= 0
    
    def get_object_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def check_roi_crossings(self, tracked_objects: List[Dict], frame_number: int) -> List[Dict]:
        """
        Check for ROI crossing events and return event list.
        
        Args:
            tracked_objects: List of tracked objects with IDs
            frame_number: Current frame number
            
        Returns:
            List of ROI crossing events
        """
        roi_events = []
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            bbox = obj['bbox']
            center = self.get_object_center(bbox)
            
            # Check if object center is in ROI
            is_in_roi = self.is_point_in_roi(center)
            
            # Get previous state
            previous_state = self.object_states.get(track_id, {
                'in_roi': None,
                'last_frame': -1,
                'entry_time': None,
                'exit_time': None
            })
            
            # Detect crossing events
            event_type = None
            if previous_state['in_roi'] is not None:
                if not previous_state['in_roi'] and is_in_roi:
                    event_type = "ENTER"
                elif previous_state['in_roi'] and not is_in_roi:
                    event_type = "EXIT"
            
            # Update object state
            self.object_states[track_id] = {
                'in_roi': is_in_roi,
                'last_frame': frame_number,
                'entry_time': frame_number if event_type == "ENTER" else previous_state.get('entry_time'),
                'exit_time': frame_number if event_type == "EXIT" else previous_state.get('exit_time'),
                'center': center,
                'bbox': bbox
            }
            
            # Create event if crossing detected
            if event_type:
                event = {
                    'track_id': track_id,
                    'event_type': event_type,
                    'frame_number': frame_number,
                    'object_class': obj['class'],
                    'confidence': obj['confidence'],
                    'center': center,
                    'bbox': bbox,
                    'features': obj.get('features', {})
                }
                roi_events.append(event)
                
                print(f"ROI Event: {event_type} - ID:{track_id} Class:{obj['class']} Frame:{frame_number}")
        
        # Clean up old object states
        self._cleanup_old_states(frame_number)
        
        return roi_events
    
    def _cleanup_old_states(self, current_frame: int, max_age: int = 100):
        """Remove old object states to prevent memory buildup."""
        to_remove = []
        for track_id, state in self.object_states.items():
            if current_frame - state['last_frame'] > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.object_states[track_id]
    
    def draw_roi(self, frame: np.ndarray, thickness: int = 3, 
                 color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw ROI polygon on frame.
        
        Args:
            frame: Input frame
            thickness: Line thickness
            color: Line color (BGR)
            
        Returns:
            Frame with ROI drawn
        """
        if color is None:
            color = config.roi.color
        
        output_frame = frame.copy()
        
        # Draw ROI polygon
        roi_points_array = np.array(self.roi_points, np.int32)
        cv2.polylines(output_frame, [roi_points_array], True, color, thickness)
        
        # Fill ROI with semi-transparent color
        roi_overlay = output_frame.copy()
        cv2.fillPoly(roi_overlay, [roi_points_array], color)
        cv2.addWeighted(output_frame, 0.8, roi_overlay, 0.2, 0, output_frame)
        
        # Add ROI label
        cv2.putText(output_frame, "ROI", 
                   (self.roi_points[0][0], self.roi_points[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output_frame
    
    def get_roi_statistics(self) -> Dict:
        """Get statistics about objects in ROI."""
        total_objects = len(self.object_states)
        objects_in_roi = sum(1 for state in self.object_states.values() if state['in_roi'])
        objects_outside_roi = total_objects - objects_in_roi
        
        return {
            'total_tracked_objects': total_objects,
            'objects_in_roi': objects_in_roi,
            'objects_outside_roi': objects_outside_roi,
            'roi_area': self.roi_polygon.area
        }
    
    def update_roi(self, new_roi_points: List[Tuple[int, int]]):
        """Update ROI with new points."""
        self.roi_points = new_roi_points
        self.roi_polygon = Polygon(new_roi_points)
        # Clear object states when ROI changes
        self.object_states.clear()
        print(f"ROI updated with {len(new_roi_points)} points")