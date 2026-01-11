import cv2
import numpy as np
import os
from pathlib import Path

class CoinPreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        self.image_files = sorted([f for f in os.listdir(input_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.current_idx = 0
        
        # Default parameters (set to midpoints of ranges)
        self.params = {
            'clahe_clip': 5,      # Midpoint of 0-10
            'clahe_grid': 8,      # Midpoint of 0-16
            'hough_param1': 100,  # Midpoint of 0-200
            'hough_param2': 50,   # Midpoint of 0-100
            'min_radius': 100,    # Midpoint of 0-200
            'max_radius': 150,    # Midpoint of 0-300
            'target_intensity': 128  # Midpoint of 0-255
        }
        
        # Create windows
        cv2.namedWindow('Original & Result')
        cv2.namedWindow('Controls')
        cv2.namedWindow('1. CLAHE Enhanced')
        cv2.namedWindow('2. Circle Detection')
        cv2.namedWindow('3. Before Autoscale')
        cv2.namedWindow('4. Final (After Autoscale)')
        
        # Create trackbars with proper initial values
        cv2.createTrackbar('CLAHE Clip', 'Controls', self.params['clahe_clip'], 10, self.on_change)
        cv2.createTrackbar('CLAHE Grid', 'Controls', self.params['clahe_grid'], 16, self.on_change)
        cv2.createTrackbar('Hough Param1', 'Controls', self.params['hough_param1'], 200, self.on_change)
        cv2.createTrackbar('Hough Param2', 'Controls', self.params['hough_param2'], 100, self.on_change)
        cv2.createTrackbar('Min Radius', 'Controls', self.params['min_radius'], 200, self.on_change)
        cv2.createTrackbar('Max Radius', 'Controls', self.params['max_radius'], 300, self.on_change)
        cv2.createTrackbar('Target Intensity', 'Controls', self.params['target_intensity'], 255, self.on_change)
        
    def on_change(self, val):
        """Callback for trackbar changes"""
        self.update_params()
        self.process_current_image()
    
    def update_params(self):
        """Read current trackbar values with safety checks"""
        self.params['clahe_clip'] = max(1, cv2.getTrackbarPos('CLAHE Clip', 'Controls'))
        self.params['clahe_grid'] = max(2, cv2.getTrackbarPos('CLAHE Grid', 'Controls'))
        self.params['hough_param1'] = max(1, cv2.getTrackbarPos('Hough Param1', 'Controls'))
        self.params['hough_param2'] = max(1, cv2.getTrackbarPos('Hough Param2', 'Controls'))
        self.params['min_radius'] = max(1, cv2.getTrackbarPos('Min Radius', 'Controls'))
        self.params['max_radius'] = max(1, cv2.getTrackbarPos('Max Radius', 'Controls'))
        self.params['target_intensity'] = cv2.getTrackbarPos('Target Intensity', 'Controls')
    
    def process_current_image(self):
        """Process and display current image"""
        if self.current_idx >= len(self.image_files):
            print("No more images")
            return None
        
        filename = self.image_files[self.current_idx]
        filepath = os.path.join(self.input_dir, filename)
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            print(f"Failed to load {filename}")
            return None
        
        original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: CLAHE Enhancement for circle detection
        grid_size = self.params['clahe_grid']
        if grid_size % 2 == 0:  # Must be even for CLAHE
            grid_size = max(2, grid_size)
        else:
            grid_size = max(2, grid_size - 1)
        
        clahe = cv2.createCLAHE(clipLimit=float(self.params['clahe_clip']), 
                                tileGridSize=(grid_size, grid_size))
        enhanced = clahe.apply(gray)
        cv2.imshow('1. CLAHE Enhanced', enhanced)
        
        # Step 2: Circle Detection
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.params['hough_param1'],
            param2=self.params['hough_param2'],
            minRadius=self.params['min_radius'],
            maxRadius=self.params['max_radius']
        )
        
        if circles is None:
            # Show original with "NO CIRCLE" text
            no_circle_img = original.copy()
            cv2.putText(no_circle_img, 'NO CIRCLE DETECTED', (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Original & Result', no_circle_img)
            cv2.imshow('2. Circle Detection', enhanced)
            return None
        
        # Get circle parameters
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        cx, cy, radius = int(x), int(y), int(r)
        
        # Show circle detection
        circle_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.circle(circle_vis, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(circle_vis, (cx, cy), 2, (0, 0, 255), 3)
        cv2.imshow('2. Circle Detection', circle_vis)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Isolate on black background
        isolated = cv2.bitwise_and(img, img, mask=mask)
        
        # Apply CLAHE enhancement
        gray_final = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
        enhanced_final = clahe.apply(gray_final)
        
        # Show before autoscale
        cv2.imshow('3. Before Autoscale', enhanced_final)
        
        # Step 3: Autoscale to target intensity
        # Calculate current average intensity of coin region
        current_avg = np.mean(enhanced_final[mask == 255])
        
        # Calculate scaling factor
        if current_avg > 0:
            scale_factor = self.params['target_intensity'] / current_avg
            # Apply scaling
            autoscaled = np.clip(enhanced_final.astype(float) * scale_factor, 0, 255).astype(np.uint8)
        else:
            autoscaled = enhanced_final
        
        # Show final autoscaled result
        cv2.imshow('4. Final (After Autoscale)', autoscaled)
        
        # Crop to circle bounding box (use autoscaled image)
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(img.shape[1], cx + radius)
        y2 = min(img.shape[0], cy + radius)
        
        cropped = autoscaled[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        
        # Apply mask to cropped
        cropped_masked = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
        
        # Resize to 100x100
        resized = cv2.resize(cropped_masked, (100, 100), interpolation=cv2.INTER_AREA)
        
        # Display original with ellipse overlay
        original_with_circle = original.copy()
        cv2.circle(original_with_circle, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(original_with_circle, (cx, cy), 2, (0, 0, 255), 3)
        
        # Resize result for display
        resized_display = cv2.resize(resized, (200, 200), interpolation=cv2.INTER_NEAREST)
        
        # Combine original and result side by side
        h1, w1 = original_with_circle.shape[:2]
        h2, w2 = resized_display.shape[:2]
        
        # Make heights match
        if h1 > h2:
            pad = (h1 - h2) // 2
            resized_display = cv2.copyMakeBorder(resized_display, pad, h1-h2-pad, 0, 0, 
                                                cv2.BORDER_CONSTANT, value=0)
        
        # Convert grayscale result to BGR for concatenation
        resized_display_bgr = cv2.cvtColor(resized_display, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack([original_with_circle, resized_display_bgr])
        
        # Calculate average intensities
        avg_before = np.mean(enhanced_final[mask == 255])
        avg_after = np.mean(autoscaled[mask == 255])
        
        # Add filename and intensity text
        cv2.putText(combined, f'{filename} ({self.current_idx+1}/{len(self.image_files)})',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f'Before: {avg_before:.1f} -> After: {avg_after:.1f}',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Original & Result', combined)
        
        return resized
    
    def save_image(self):
        """Save current processed image with original filename"""
        processed = self.process_current_image()
        if processed is None:
            print("Cannot save - no circle detected")
            return
        
        filename = self.image_files[self.current_idx]
        output_path = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(output_path, processed)
        print(f'Saved: {filename}')
        
        # Move to next image
        self.current_idx += 1
        if self.current_idx < len(self.image_files):
            self.process_current_image()
        else:
            print("All images processed!")
    
    def run(self):
        """Main processing loop"""
        print("Controls:")
        print("  s - Save current image")
        print("  n - Next image (skip)")
        print("  p - Previous image")
        print("  q - Quit")
        
        self.process_current_image()
        
        while True:
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('s'):
                self.save_image()
            elif key == ord('n'):
                self.current_idx += 1
                if self.current_idx < len(self.image_files):
                    self.process_current_image()
                else:
                    print("End of images")
                    self.current_idx = len(self.image_files) - 1
            elif key == ord('p'):
                self.current_idx = max(0, self.current_idx - 1)
                self.process_current_image()
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    input_dir = './templates'  # UPDATE THIS
    output_dir = './templates/processed'  # UPDATE THIS
    
    preprocessor = CoinPreprocessor(input_dir, output_dir)
    preprocessor.run()