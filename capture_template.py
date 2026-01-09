"""
Capture template images for coin detection.
Shows camera feed, detects quarters, and saves templates with countdown.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from coin_detector import QuarterDetector


def capture_template(output_path='heads_ref.jpg'):
    """
    Capture a template image of a quarter from the camera.
    
    Args:
        output_path: Path where to save the template image
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print(f"\n--- Coin Template Capture ---")
    print(f"Show a quarter to the camera.")
    print(f"When the coin is recognized, a 3-second countdown will start.")
    print(f"Template will be saved to: {output_path}")
    print(f"Press 'q' to quit without saving.\n")
    
    # Simple detector without template (just for coin detection)
    detector = QuarterDetector(heads_template=None)
    
    countdown_started = False
    countdown_start_time = None
    captured_frame = None
    last_coin_info = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break
        
        # Try to detect coin
        coin_info = detector.detect_coin(frame)
        
        # Keep track of last detected coin
        if coin_info is not None:
            last_coin_info = coin_info
        
        # Draw instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if coin_info is not None and not countdown_started:
            # Coin detected, start countdown
            countdown_started = True
            countdown_start_time = time.time()
            x, y, radius = coin_info
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)
            cv2.putText(frame, "Coin detected! Starting countdown...", (20, 40),
                       font, 0.7, (0, 255, 0), 2)
        
        elif countdown_started and last_coin_info is not None:
            # Show countdown using last detected coin position
            elapsed = time.time() - countdown_start_time
            
            if elapsed < 3.0:
                # Still counting down
                remaining = int(3 - elapsed)
                x, y, radius = last_coin_info
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)
                
                # Large countdown number
                cv2.putText(frame, str(remaining), (frame.shape[1]//2 - 40, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            else:
                # Time to save
                x, y, radius = last_coin_info
                coin_region = detector.extract_coin_region(frame, last_coin_info)
                
                if coin_region is not None:
                    captured_frame = coin_region
                    cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)
                    cv2.putText(frame, "Saving...", (20, 40),
                               font, 1, (0, 255, 0), 2)
                
                # Show frame briefly then break
                cv2.imshow('Capture Template', frame)
                cv2.waitKey(500)
                break
        
        else:
            # No coin detected
            cv2.putText(frame, "Show a quarter to the camera", (20, 40),
                       font, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (20, 80),
                       font, 0.6, (100, 100, 100), 1)
        
        # Display frame
        cv2.imshow('Capture Template', frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the captured image
    if captured_frame is not None:
        cv2.imwrite(output_path, captured_frame)
        print(f"\n✓ Template saved to: {output_path}")
        print(f"  Image size: {captured_frame.shape}")
        return True
    else:
        print("\n✗ Failed to capture template")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture quarter template images from camera")
    parser.add_argument('--output', '-o', default='heads_ref.jpg',
                       help='Output file path for template (default: heads_ref.jpg)')
    parser.add_argument('--multiple', '-m', action='store_true',
                       help='Capture multiple templates (heads, tails, etc.)')
    
    args = parser.parse_args()
    
    if args.multiple:
        # Capture multiple templates continuously
        template_counter = 1
        
        while True:
            output_path = f'template_{template_counter:02d}.jpg'
            
            print(f"\n{'='*50}")
            print(f"Capturing template #{template_counter}")
            print(f"{'='*50}")
            
            success = capture_template(output_path)
            
            if success:
                # Display the saved image
                img = cv2.imread(output_path)
                if img is not None:
                    cv2.imshow(f'Template #{template_counter}', img)
                    cv2.waitKey(1500)
                    cv2.destroyAllWindows()
                template_counter += 1
    else:
        # Capture single template
        capture_template(args.output)


if __name__ == "__main__":
    main()
