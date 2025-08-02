import numpy as np
import cv2
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
import colorsys

class FeatureExtractor:
    def __init__(self):
        self.color_names = {
            'red': (0, 100, 100),
            'orange': (15, 100, 100),
            'yellow': (30, 100, 100),
            'green': (60, 100, 100),
            'cyan': (90, 100, 100),
            'blue': (120, 100, 100),
            'purple': (150, 100, 100),
            'pink': (180, 100, 100),
            'brown': (15, 50, 30),
            'black': (0, 0, 10),
            'white': (0, 0, 90),
            'gray': (0, 0, 50)
        }
    
    def extract_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract visual features from masked object region.
        
        Args:
            frame: Original frame (BGR)
            mask: Binary mask for the object
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Extract dominant color
        dominant_color = self.extract_dominant_color(frame, mask)
        features['dominant_color'] = dominant_color
        
        # Extract size information
        size_info = self.extract_size_features(mask)
        features.update(size_info)
        
        # Extract shape features
        shape_info = self.extract_shape_features(mask)
        features.update(shape_info)
        
        return features
    
    def extract_dominant_color(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract dominant color from masked region.
        
        Args:
            frame: Original frame (BGR)
            mask: Binary mask
            
        Returns:
            Dictionary with color information
        """
        try:
            # Get pixels within mask
            masked_pixels = frame[mask > 0]
            
            if masked_pixels.size == 0:
                return {'color_name': 'unknown', 'rgb': (0, 0, 0), 'confidence': 0.0}
            
            # Convert BGR to RGB
            rgb_pixels = masked_pixels[:, [2, 1, 0]]
            
            # Use KMeans to find dominant colors
            n_colors = min(3, len(masked_pixels))
            if n_colors == 0:
                return {'color_name': 'unknown', 'rgb': (0, 0, 0), 'confidence': 0.0}
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(rgb_pixels)
            
            # Get the most frequent color (largest cluster)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            dominant_color_idx = np.argmax(label_counts)
            dominant_color_rgb = colors[dominant_color_idx].astype(int)
            
            # Calculate confidence based on cluster size
            confidence = label_counts[dominant_color_idx] / len(labels)
            
            # Map to color name
            color_name = self.rgb_to_color_name(dominant_color_rgb)
            
            return {
                'color_name': color_name,
                'rgb': tuple(dominant_color_rgb),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"Error extracting color: {e}")
            return {'color_name': 'unknown', 'rgb': (0, 0, 0), 'confidence': 0.0}
    
    def rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """
        Convert RGB values to closest color name.
        
        Args:
            rgb: RGB color values
            
        Returns:
            Color name string
        """
        r, g, b = rgb
        
        # Convert to HSV for better color matching
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h = h * 180  # Convert to degrees
        s = s * 100  # Convert to percentage
        v = v * 100  # Convert to percentage
        
        # Special cases
        if v < 20:
            return 'black'
        elif s < 20 and v > 80:
            return 'white'
        elif s < 20:
            return 'gray'
        
        # Find closest color by hue
        closest_color = 'unknown'
        min_distance = float('inf')
        
        for color_name, (target_h, target_s, target_v) in self.color_names.items():
            if color_name in ['black', 'white', 'gray']:
                continue
            
            # Calculate hue distance (circular)
            hue_diff = min(abs(h - target_h), 360 - abs(h - target_h))
            distance = hue_diff
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    def extract_size_features(self, mask: np.ndarray) -> Dict:
        """
        Extract size-related features from mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with size features
        """
        # Calculate area
        area = np.sum(mask > 0)
        
        # Get bounding box dimensions
        coords = np.where(mask > 0)
        if coords[0].size == 0:
            return {
                'pixel_area': 0,
                'width': 0,
                'height': 0,
                'aspect_ratio': 0.0,
                'size_category': 'unknown'
            }
        
        min_y, max_y = np.min(coords[0]), np.max(coords[0])
        min_x, max_x = np.min(coords[1]), np.max(coords[1])
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        aspect_ratio = width / height if height > 0 else 0.0
        
        # Categorize size
        size_category = self.categorize_size(area)
        
        return {
            'pixel_area': int(area),
            'width': int(width),
            'height': int(height),
            'aspect_ratio': float(aspect_ratio),
            'size_category': size_category
        }
    
    def categorize_size(self, area: int) -> str:
        """
        Categorize object size based on pixel area.
        
        Args:
            area: Pixel area
            
        Returns:
            Size category string
        """
        if area < 1000:
            return 'very_small'
        elif area < 5000:
            return 'small'
        elif area < 20000:
            return 'medium'
        elif area < 50000:
            return 'large'
        else:
            return 'very_large'
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict:
        """
        Extract shape-related features from mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with shape features
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return {
                    'perimeter': 0.0,
                    'compactness': 0.0,
                    'contour_count': 0
                }
            
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate compactness (circularity)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
            else:
                compactness = 0.0
            
            return {
                'perimeter': float(perimeter),
                'compactness': float(compactness),
                'contour_count': len(contours)
            }
            
        except Exception as e:
            print(f"Error extracting shape features: {e}")
            return {
                'perimeter': 0.0,
                'compactness': 0.0,
                'contour_count': 0
            }