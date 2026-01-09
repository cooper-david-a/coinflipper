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
    
    def __init__(self, heads_template=None, tails_template=None):
        """
        Initialize the detector.
        
        Args:
            heads_template: Path to heads reference template image
            tails_template: Path to tails reference template image
        """
        self.heads_template = None
        self.tails_template = None
        
        if heads_template and Path(heads_template).exists():
            self.load_template(heads_template, 'heads')
        
        if tails_template and Path(tails_template).exists():
            self.load_template(tails_template, 'tails')
    
    def load_template(self, template_path, coin_type):
        """
        Load a reference template image.
        
        Args:
            template_path: Path to template image
            coin_type: 'heads' or 'tails'
        """
        template = cv2.imread(str(template_path))
        if template is None:
            print(f"Error: Could not load {coin_type} template from {template_path}")
            return
        
        # Resize template to standard size
        template = cv2.resize(template, (200, 200))
        
        if coin_type == 'heads':
            self.heads_template = template
            print(f"Heads template loaded from {template_path}")
        elif coin_type == 'tails':
            self.tails_template = template
            print(f"Tails template loaded from {template_path}")
    
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
            
        top_left_x = np.clip(x.astype(np.int32) - radius - 10,0,65535)
        top_left_y = np.clip(y.astype(np.int32) - radius - 10,0,65535)
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
        Classify coin as heads or tails using template matching.
        Compares against both templates and picks the best match.
        
        Args:
            coin_image: Extracted coin image
            
        Returns:
            Tuple of (label, confidence, heads_score, tails_score)
        """
        if self.heads_template is None or self.tails_template is None:
            print("Error: Both heads and tails templates required.")
            return None, 0.0, 0.0, 0.0
        
        if coin_image is None or coin_image.size == 0:
            print("Error: Invalid coin image.")
            return None, 0.0, 0.0, 0.0
        
        # Resize coin image to match template size
        coin_resized = cv2.resize(coin_image, (200, 200))
        
        # Convert to grayscale for matching
        coin_gray = cv2.cvtColor(coin_resized, cv2.COLOR_BGR2GRAY) if len(coin_resized.shape) == 3 else coin_resized
        heads_gray = cv2.cvtColor(self.heads_template, cv2.COLOR_BGR2GRAY) if len(self.heads_template.shape) == 3 else self.heads_template
        tails_gray = cv2.cvtColor(self.tails_template, cv2.COLOR_BGR2GRAY) if len(self.tails_template.shape) == 3 else self.tails_template
        
        # Normalize images for better matching
        coin_gray = cv2.normalize(coin_gray, None, 0, 255, cv2.NORM_MINMAX)
        heads_gray = cv2.normalize(heads_gray, None, 0, 255, cv2.NORM_MINMAX)
        tails_gray = cv2.normalize(tails_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Compute template matching scores using correlation coefficient
        heads_match = cv2.matchTemplate(coin_gray, heads_gray, cv2.TM_CCOEFF_NORMED)
        tails_match = cv2.matchTemplate(coin_gray, tails_gray, cv2.TM_CCOEFF_NORMED)
        
        heads_score = np.max(heads_match)
        tails_score = np.max(tails_match)
        
        # Classify based on best match
        if heads_score > tails_score:
            label = "Heads"
            confidence = heads_score
        else:
            label = "Tails"
            confidence = tails_score
        
        return label, confidence, heads_score, tails_score
    
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
        label, confidence, heads_score, tails_score = self.classify_heads_tails(coin_region)
        
        result = {
            'image_path': str(image_path),
            'coin_detected': True,
            'coin_position': coin_info,
            'classification': label,
            'confidence': confidence,
            'heads_score': heads_score,
            'tails_score': tails_score
        }
        
        return result
    
    def process_camera(self, display=True):
        """
        Real-time quarter detection from webcam.
        
        Args:
            display: Whether to display results
        """
        if self.heads_template is None or self.tails_template is None:
            print("Error: Both heads and tails templates required.")
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
                label, confidence, heads_score, tails_score = self.classify_heads_tails(coin_region)
                
                # Draw results
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x - 50, y - radius - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"H:{heads_score:.2f} T:{tails_score:.2f}", (x - 50, y - radius - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
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
