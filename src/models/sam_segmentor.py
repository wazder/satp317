import torch
import numpy as np
from typing import List, Tuple
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os

from ..config import config

class SAMSegmentor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAM using device: {self.device}")
        
        # Load SAM model
        sam_checkpoint_path = os.path.join("weights", config.models.sam_checkpoint)
        
        # Check if checkpoint exists
        if not os.path.exists(sam_checkpoint_path):
            print(f"Warning: SAM checkpoint not found at {sam_checkpoint_path}")
            print("Please download the SAM checkpoint from: https://github.com/facebookresearch/segment-anything")
            # For now, create a dummy segmentor that will be replaced
            self.mask_generator = None
            return
        
        try:
            sam = sam_model_registry[config.models.sam_model_type](checkpoint=sam_checkpoint_path)
            sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            print("SAM model loaded successfully")
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            self.mask_generator = None
    
    def segment_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Generate instance masks for all objects in the frame using SAM.
        
        Args:
            frame: Input image as numpy array (BGR)
            
        Returns:
            Tuple of (masks, bounding_boxes)
            - masks: List of binary masks as numpy arrays
            - bounding_boxes: List of bounding boxes [x1, y1, x2, y2]
        """
        if self.mask_generator is None:
            # Return empty results if SAM is not loaded
            return [], []
        
        try:
            # Convert BGR to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks_data = self.mask_generator.generate(rgb_frame)
            
            # Extract masks and bounding boxes
            masks = []
            bounding_boxes = []
            
            # Sort by area (largest first) to prioritize larger objects
            masks_data.sort(key=lambda x: x['area'], reverse=True)
            
            for mask_data in masks_data:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, w, h]
                
                # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
                
                masks.append(mask)
                bounding_boxes.append(bbox_xyxy)
            
            print(f"SAM generated {len(masks)} masks")
            return masks, bounding_boxes
            
        except Exception as e:
            print(f"Error in SAM segmentation: {e}")
            return [], []
    
    def create_dummy_masks(self, frame: np.ndarray, detections: List[dict]) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Create dummy masks from YOLO detections when SAM is not available.
        This is a fallback method for testing.
        
        Args:
            frame: Input image
            detections: YOLO detections
            
        Returns:
            Tuple of (masks, bounding_boxes)
        """
        masks = []
        bounding_boxes = []
        
        height, width = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Create a binary mask for the bounding box area
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            
            masks.append(mask.astype(bool))
            bounding_boxes.append([x1, y1, x2, y2])
        
        return masks, bounding_boxes
    
    def draw_masks(self, frame: np.ndarray, masks: List[np.ndarray], 
                   alpha: float = 0.3) -> np.ndarray:
        """
        Draw masks on frame with random colors.
        
        Args:
            frame: Input image
            masks: List of binary masks
            alpha: Transparency for mask overlay
            
        Returns:
            Frame with drawn masks
        """
        output_frame = frame.copy()
        
        # Generate random colors for each mask
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if mask is not None:
                color = colors[i].tolist()
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = color
                
                # Blend with original frame
                output_frame = cv2.addWeighted(output_frame, 1 - alpha, 
                                             colored_mask, alpha, 0)
                
                # Draw mask contours
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(output_frame, contours, -1, color, 2)
        
        return output_frame