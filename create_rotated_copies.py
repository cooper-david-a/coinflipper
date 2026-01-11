import cv2
import numpy as np
import os
from pathlib import Path

def rotate_image(image, angle):
    """Rotate image by angle (degrees) about center"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    
    return rotated

def augment_images(input_dir, output_dir, num_copies=10):
    """Create rotated copies of all images in input directory"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating {num_copies} rotated copies of each...")
    print(f"Output directory: {output_dir}")
    
    total_created = 0
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to load {filename}, skipping...")
            continue
        
        # Get base name without extension
        base_name, ext = os.path.splitext(filename)
        
        # Create rotated copies
        for i in range(num_copies):
            # Generate random rotation angle (0-360 degrees)
            angle = np.random.uniform(0, 360)
            
            # Rotate image
            rotated = rotate_image(img, angle)
            
            # Create output filename
            output_filename = f"{base_name}_rot{i:02d}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save rotated image
            cv2.imwrite(output_path, rotated)
            total_created += 1
        
        print(f"Processed: {filename} -> {num_copies} copies")
    
    print(f"\nComplete! Created {total_created} rotated images")
    print(f"Average rotation per image: {num_copies} copies")


if __name__ == "__main__":
    # Set your directories here
    input_dir = './templates/processed'      # UPDATE THIS
    output_dir = input_dir + '/rotated'   # UPDATE THIS
    
    # Number of rotated copies per image
    num_copies = 10
    
    augment_images(input_dir, output_dir, num_copies)