import argparse
from pathlib import Path
from coin_detector import QuarterDetector


def main():
    parser = argparse.ArgumentParser(description="Quarter heads/tails detector using template matching")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Detect command for single image
    detect_parser = subparsers.add_parser('detect', help='Detect quarter in image')
    detect_parser.add_argument('image', help='Path to image file')
    detect_parser.add_argument('--heads', help='Path to heads template image', required=True)
    detect_parser.add_argument('--threshold', type=float, default=0.6, 
                              help='Confidence threshold for heads classification (0.0-1.0)')
    
    # Camera command for real-time detection
    camera_parser = subparsers.add_parser('camera', help='Real-time detection from camera')
    camera_parser.add_argument('--heads', help='Path to heads template image', required=True)
    camera_parser.add_argument('--threshold', type=float, default=0.6,
                              help='Confidence threshold for heads classification (0.0-1.0)')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        # Detect mode
        detector = QuarterDetector(heads_template=args.heads, threshold=args.threshold)
        result = detector.process_image(args.image)
        
        if result:
            print(f"\n✓ Coin detected!")
            print(f"  Classification: {result['classification']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Position: {result['coin_position']}")
        else:
            print("\n✗ Could not detect coin in image")
    
    elif args.command == 'camera':
        # Camera mode
        detector = QuarterDetector(heads_template=args.heads, threshold=args.threshold)
        print("Starting camera detection...")
        detector.process_camera()
    
    else:
        # Default: show help if no command given
        parser.print_help()


if __name__ == "__main__":
    main()
