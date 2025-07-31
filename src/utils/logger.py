import csv
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path
import datetime

from ..config import config

class EventLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.events = []
        self.csv_file = None
        self.csv_writer = None
        
        # Initialize CSV file if enabled
        if config.logging.output_csv:
            self.csv_file = open(self.output_dir / "roi_events.csv", 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self._write_csv_header()
        
        print(f"Event logger initialized, output dir: {self.output_dir}")
    
    def _write_csv_header(self):
        """Write CSV header."""
        header = [
            'timestamp',
            'frame_number',
            'track_id',
            'event_type',
            'object_class',
            'confidence',
            'center_x',
            'center_y',
            'bbox_x1',
            'bbox_y1',
            'bbox_x2',
            'bbox_y2'
        ]
        
        # Add feature columns if enabled
        if config.logging.log_features:
            header.extend([
                'dominant_color',
                'color_rgb',
                'color_confidence',
                'pixel_area',
                'width',
                'height',
                'aspect_ratio',
                'size_category',
                'perimeter',
                'compactness'
            ])
        
        self.csv_writer.writerow(header)
    
    def log_events(self, roi_events: List[Dict], timestamp: float):
        """
        Log ROI crossing events.
        
        Args:
            roi_events: List of ROI event dictionaries
            timestamp: Timestamp in seconds
        """
        for event in roi_events:
            # Add timestamp
            event['timestamp'] = timestamp
            
            # Store in memory
            self.events.append(event.copy())
            
            # Write to CSV immediately if enabled
            if self.csv_writer:
                self._write_event_to_csv(event)
            
            # Print event for immediate feedback
            self._print_event(event)
    
    def _write_event_to_csv(self, event: Dict):
        """Write single event to CSV file."""
        row = [
            event['timestamp'],
            event['frame_number'],
            event['track_id'],
            event['event_type'],
            event['object_class'],
            event['confidence'],
            event['center'][0],
            event['center'][1],
            event['bbox'][0],
            event['bbox'][1],
            event['bbox'][2],
            event['bbox'][3]
        ]
        
        # Add feature data if available
        if config.logging.log_features and 'features' in event:
            features = event['features']
            color_info = features.get('dominant_color', {})
            
            row.extend([
                color_info.get('color_name', ''),
                str(color_info.get('rgb', '')),
                color_info.get('confidence', 0.0),
                features.get('pixel_area', 0),
                features.get('width', 0),
                features.get('height', 0),
                features.get('aspect_ratio', 0.0),
                features.get('size_category', ''),
                features.get('perimeter', 0.0),
                features.get('compactness', 0.0)
            ])
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written immediately
    
    def _print_event(self, event: Dict):
        """Print event to console for immediate feedback."""
        features = event.get('features', {})
        color_info = features.get('dominant_color', {})
        
        feature_str = ""
        if color_info:
            color_name = color_info.get('color_name', 'unknown')
            size_cat = features.get('size_category', 'unknown')
            feature_str = f"Color:{color_name} Size:{size_cat}"
        
        print(f"[{event['timestamp']:.1f}s] {event['event_type']} - "
              f"ID:{event['track_id']} {event['object_class']} "
              f"(conf:{event['confidence']:.2f}) {feature_str}")
    
    def save_logs(self):
        """Save all logged events to files."""
        if not self.events:
            print("No events to save")
            return
        
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
        
        # Save JSON file if enabled
        if config.logging.output_json:
            self._save_json_log()
        
        # Generate summary report
        self._generate_summary()
        
        print(f"Saved {len(self.events)} events to log files")
    
    def _save_json_log(self):
        """Save events as JSON file."""
        json_file = self.output_dir / "roi_events.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'metadata': {
                'total_events': len(self.events),
                'generated_at': datetime.datetime.now().isoformat(),
                'config': {
                    'target_classes': config.target_classes,
                    'roi_points': config.roi.roi_points,
                    'video_fps': config.video.target_fps
                }
            },
            'events': self.events
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    def _generate_summary(self):
        """Generate summary statistics."""
        if not self.events:
            return
        
        df = pd.DataFrame(self.events)
        
        summary = {
            'total_events': len(self.events),
            'enter_events': len(df[df['event_type'] == 'ENTER']),
            'exit_events': len(df[df['event_type'] == 'EXIT']),
            'unique_objects': df['track_id'].nunique(),
            'class_distribution': df['object_class'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min() if len(df) > 1 else 0
        }
        
        # Save summary
        summary_file = self.output_dir / "event_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print("\n=== EVENT SUMMARY ===")
        print(f"Total events: {summary['total_events']}")
        print(f"Enter events: {summary['enter_events']}")
        print(f"Exit events: {summary['exit_events']}")
        print(f"Unique objects tracked: {summary['unique_objects']}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        print(f"Class distribution: {summary['class_distribution']}")
        print(f"Video duration: {summary['duration_seconds']:.1f} seconds")
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        if not self.events:
            return {'total_events': 0}
        
        df = pd.DataFrame(self.events)
        
        return {
            'total_events': len(self.events),
            'enter_events': len(df[df['event_type'] == 'ENTER']),
            'exit_events': len(df[df['event_type'] == 'EXIT']),
            'unique_objects': df['track_id'].nunique(),
            'class_counts': df['object_class'].value_counts().to_dict()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()