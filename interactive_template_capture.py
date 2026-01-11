import cv2
import os
from pathlib import Path
import re

class CoinCameraCapture:
    def __init__(self, output_dir, camera_index=0):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Set camera resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Find last image numbers for heads and tails
        self.heads_counter = self.get_last_number('heads')
        self.tails_counter = self.get_last_number('tails')
        
        # Create window
        cv2.namedWindow('Camera Feed')
        
        print(f"Output directory: {output_dir}")
        print(f"Next heads image: heads_{self.heads_counter:02d}.jpg")
        print(f"Next tails image: tails_{self.tails_counter:02d}.jpg")
    
    def get_last_number(self, prefix):
        """Find the highest number for given prefix in output directory"""
        pattern = re.compile(rf'{prefix}_(\d{{2}})\.jpg')
        max_num = 0
        
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                match = pattern.match(filename)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        
        return max_num + 1
    
    def get_crop_bounds(self, frame):
        """Calculate square crop bounds (center crop)"""
        h, w = frame.shape[:2]
        size = min(h, w)
        
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        y_end = y_start + size
        x_end = x_start + size
        
        return x_start, y_start, x_end, y_end
    
    def save_image(self, frame, label):
        """Save image with appropriate counter, cropped to square and resized"""
        # Crop to square (center crop)
        h, w = frame.shape[:2]
        size = min(h, w)
        
        # Calculate crop coordinates (center)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        
        cropped = frame[y_start:y_start+size, x_start:x_start+size]
        
        # Resize if larger than 500px
        if size > 500:
            resized = cv2.resize(cropped, (500, 500), interpolation=cv2.INTER_AREA)
        else:
            resized = cropped
        
        # Save with counter
        if label == 'heads':
            counter = self.heads_counter
            self.heads_counter += 1
        else:  # tails
            counter = self.tails_counter
            self.tails_counter += 1
        
        filename = f'{label}_{counter:02d}.jpg'
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, resized)
        print(f'Saved: {filename} ({resized.shape[1]}x{resized.shape[0]})')
        
        return filename
    
    def run(self):
        """Main capture loop"""
        print("\nControls:")
        print("  h - Capture and save as heads")
        print("  t - Capture and save as tails")
        print("  q - Quit")
        print("\nReady to capture...")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display frame with info overlay
            display_frame = frame.copy()
            
            # Draw crop boundary rectangle
            x_start, y_start, x_end, y_end = self.get_crop_bounds(frame)
            cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), 
                         (0, 255, 0), 2)
            
            # Add status text
            h, w = frame.shape[:2]
            size = min(h, w)
            final_size = min(size, 500)
            
            cv2.putText(display_frame, f'Next: heads_{self.heads_counter:02d} (h)  |  tails_{self.tails_counter:02d} (t)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Crop: {size}x{size} -> Save: {final_size}x{final_size}',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, 'Press Q to quit',
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Feed', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('h'):
                filename = self.save_image(frame, 'heads')
                # Brief flash to show capture
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), 
                            (255, 255, 255), 30)
                cv2.imshow('Camera Feed', flash)
                cv2.waitKey(100)
                
            elif key == ord('t'):
                filename = self.save_image(frame, 'tails')
                # Brief flash to show capture
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), 
                            (255, 255, 255), 30)
                cv2.imshow('Camera Feed', flash)
                cv2.waitKey(100)
                
            elif key == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCapture complete!")


if __name__ == "__main__":
    # Set your output directory here
    output_dir = './templates/'  # UPDATE THIS
    
    # Camera index (0 for default camera, 1 for second camera, etc.)
    camera_index = 2
    
    try:
        capture = CoinCameraCapture(output_dir, camera_index)
        capture.run()
    except Exception as e:
        print(f"Error: {e}")