"""
Quarter detection and heads/tails classification using OpenCV template matching.
"""

import cv2
import numpy as np
from pathlib import Path


def debug_show_image(image, title="Image", wait_ms=0):
    """
    Display an image for debugging. Useful when paused at breakpoints.
    
    Args:
        image: OpenCV image (BGR format)
        title: Window title
        wait_ms: Milliseconds to wait (0 = wait for keypress)
    """
    if image is None or image.size == 0:
        print(f"Cannot display: image is empty")
        return
    
    print(f"Displaying image: {title} (shape={image.shape})")
    cv2.imshow(title, image)
    cv2.waitKey(wait_ms)


class QuarterDetector:
    """Detect quarters and classify heads vs tails using template matching."""
    
    def __init__(self, heads_template=None, threshold=0.6):
        """
        Initialize the detector.
        
        Args:
            heads_template: Path to heads reference template image
            threshold: Confidence threshold for heads classification (0.0-1.0)
        """
        self.heads_template = None
        self.threshold = threshold
        
        if heads_template and Path(heads_template).exists():
            self.load_template(heads_template, 'heads')
    
    def load_template(self, template_path, coin_type='heads'):
        """
        Load a reference heads template image.
        
        Args:
            template_path: Path to template image
            coin_type: Type label (default 'heads')
        """
        template = cv2.imread(str(template_path))
        if template is None:
            print(f"Error: Could not load template from {template_path}")
            return
        
        # Resize template to standard size
        template = cv2.resize(template, (200, 200))
        self.heads_template = template
        print(f"Heads template loaded from {template_path}")
    
    def detect_coin(self, image):
        """
        Detect circular coin in image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (x, y, radius) for detected coin, or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Hough Circle Detection
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the largest circle (assuming it's the main coin)
            circle = circles[0][0]
            return tuple(circle)
        
        return None
    
    def extract_coin_region(self, image, coin_info):
        """
        Extract the circular coin region from the image.
        
        Args:
            image: Input image
            coin_info: Tuple of (x, y, radius)
            
        Returns:
            Extracted coin region (square image) or None if invalid
        """
        x, y, radius = coin_info
        
        # Extract a square region around the coin
        size = int(radius * 2.2)
        if size < 50:  # Coin too small
            print(f"Warning: Detected coin too small (size={size}px). Skipping.")
            return None
            
        top_left_x = max(0, x - radius - 10)
        top_left_y = max(0, y - radius - 10)
        bottom_right_x = min(image.shape[1], top_left_x + size)
        bottom_right_y = min(image.shape[0], top_left_y + size)
        
        coin_region = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        # Validate region
        if coin_region.size == 0:
            print("Warning: Extracted coin region is empty.")
            return None
            
        return coin_region
    
    def classify_heads_tails(self, coin_image):
        """
        Classify coin as heads or tails using heads template matching.
        If confidence exceeds threshold, it's heads. Otherwise, it's tails.
        
        Args:
            coin_image: Extracted coin image
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.heads_template is None:
            print("Error: Heads template not loaded. Please provide heads template.")
            return None, 0.0
        
        if coin_image is None or coin_image.size == 0:
            print("Error: Invalid coin image.")
            return None, 0.0
        
        # Resize coin image to match template size
        coin_resized = cv2.resize(coin_image, (200, 200))
        
        # Convert to grayscale for matching
        coin_gray = cv2.cvtColor(coin_resized, cv2.COLOR_BGR2GRAY) if len(coin_resized.shape) == 3 else coin_resized
        heads_gray = cv2.cvtColor(self.heads_template, cv2.COLOR_BGR2GRAY) if len(self.heads_template.shape) == 3 else self.heads_template
        
        # Normalize images for better matching
        coin_gray = cv2.normalize(coin_gray, None, 0, 255, cv2.NORM_MINMAX)
        heads_gray = cv2.normalize(heads_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Compute template matching score using correlation coefficient
        heads_match = cv2.matchTemplate(coin_gray, heads_gray, cv2.TM_CCOEFF_NORMED)
        heads_score = np.max(heads_match)
        
        # Classify based on threshold
        if heads_score >= self.threshold:
            label = "Heads"
        else:
            label = "Tails"
        
        return label, heads_score
    
    def process_image(self, image_path):
        """
        Process an image file and classify the quarter.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with results or None if coin not detected
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Detect coin
        coin_info = self.detect_coin(image)
        if coin_info is None:
            print("No coin detected in image")
            return None
        
        # Extract coin region
        coin_region = self.extract_coin_region(image, coin_info)
        if coin_region is None:
            print("Failed to extract coin region")
            return None
        
        # Classify
        label, confidence = self.classify_heads_tails(coin_region)
        
        result = {
            'image_path': str(image_path),
            'coin_detected': True,
            'coin_position': coin_info,
            'classification': label,
            'confidence': confidence
        }
        
        return result
    
    def process_camera(self, display=True):
        """
        Real-time quarter detection from webcam.
        
        Args:
            display: Whether to display results
        """
        if self.heads_template is None:
            print("Error: Heads template not loaded. Please provide heads template.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect coin
            coin_info = self.detect_coin(frame)
            
            if coin_info is not None:
                x, y, radius = coin_info
                
                # Extract and classify
                coin_region = self.extract_coin_region(frame, coin_info)
                label, confidence = self.classify_heads_tails(coin_region)
                
                # Draw results
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x - 50, y - radius - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No coin detected", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if display:
                cv2.imshow('Quarter Detector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'frame_{len([f for f in Path(".").glob("frame_*.png")])}.png', frame)
                print("Frame saved")
        
        cap.release()
        cv2.destroyAllWindows()
