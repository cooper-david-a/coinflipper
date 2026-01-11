import cv2
import numpy as np

class CoinClassifier:
    def __init__(self, onnx_model_path):
        """Load ONNX model with OpenCV DNN"""
        self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.class_names = {0: 'heads', 1: 'tails'}
        
        print(f"Loaded model from {onnx_model_path}")
        print(f"Classes: {self.class_names}")
    
    def predict(self, preprocessed_image):
        """Run inference on preprocessed 100x100 image"""
        if preprocessed_image is None:
            return None, 0.0
        
        # Normalize to [-1, 1] (same as training)
        normalized = (preprocessed_image.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Add batch and channel dimensions: (1, 1, 100, 100)
        blob = normalized.reshape(1, 1, 100, 100)
        
        # Set input
        self.net.setInput(blob)
        
        # Forward pass
        output = self.net.forward()
        
        # Get prediction
        pred_class = np.argmax(output[0])
        
        # Apply softmax to get proper probabilities
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)
        
        return self.class_names[pred_class], probabilities[pred_class]


class RealtimeInference:
    """Combines preprocessing and inference for real-time camera"""
    def __init__(self, onnx_model_path, camera_index=0, show_previews=True):
        # Load classifier
        self.classifier = CoinClassifier(onnx_model_path)
        
        # Preview toggle
        self.show_previews = show_previews
        
        # Preprocessing parameters (same as interactive_preprocessing.py)
        self.params = {
            'clahe_clip': 5,
            'clahe_grid': 8,
            'hough_param1': 100,
            'hough_param2': 50,
            'min_radius': 100,
            'max_radius': 150,
            'target_intensity': 128
        }
        
        # Create windows
        cv2.namedWindow('Camera & Result')
        cv2.namedWindow('Controls')
        
        if self.show_previews:
            cv2.namedWindow('Final (After Autoscale)')
        
        # Create trackbars
        cv2.createTrackbar('CLAHE Clip', 'Controls', self.params['clahe_clip'], 10, self.on_change)
        cv2.createTrackbar('CLAHE Grid', 'Controls', self.params['clahe_grid'], 16, self.on_change)
        cv2.createTrackbar('Hough Param1', 'Controls', self.params['hough_param1'], 200, self.on_change)
        cv2.createTrackbar('Hough Param2', 'Controls', self.params['hough_param2'], 100, self.on_change)
        cv2.createTrackbar('Min Radius', 'Controls', self.params['min_radius'], 250, self.on_change)
        cv2.createTrackbar('Max Radius', 'Controls', self.params['max_radius'], 250, self.on_change)
        cv2.createTrackbar('Target Intensity', 'Controls', self.params['target_intensity'], 255, self.on_change)
        cv2.createTrackbar('Show Preview', 'Controls', 1 if self.show_previews else 0, 1, self.on_change)
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Preview windows: {'Enabled' if self.show_previews else 'Disabled'}")
    
    def on_change(self, val):
        """Callback for trackbar changes"""
        self.update_params()
    
    def update_params(self):
        """Read current trackbar values"""
        self.params['clahe_clip'] = max(1, cv2.getTrackbarPos('CLAHE Clip', 'Controls'))
        self.params['clahe_grid'] = max(2, cv2.getTrackbarPos('CLAHE Grid', 'Controls'))
        self.params['hough_param1'] = max(1, cv2.getTrackbarPos('Hough Param1', 'Controls'))
        self.params['hough_param2'] = max(1, cv2.getTrackbarPos('Hough Param2', 'Controls'))
        self.params['min_radius'] = max(1, cv2.getTrackbarPos('Min Radius', 'Controls'))
        self.params['max_radius'] = max(1, cv2.getTrackbarPos('Max Radius', 'Controls'))
        self.params['target_intensity'] = cv2.getTrackbarPos('Target Intensity', 'Controls')
        
        # Handle show_preview toggle
        self.show_previews = cv2.getTrackbarPos('Show Preview', 'Controls') == 1

    
    def process_frame(self, frame):
        """Process frame using same pipeline as interactive preprocessing"""
        # Crop to square and resize to 500x500 first for performance        
        original = frame.copy()
        img = frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE Enhancement
        grid_size = self.params['clahe_grid']
        if grid_size % 2 == 0:
            grid_size = max(2, grid_size)
        else:
            grid_size = max(2, grid_size - 1)
        
        clahe = cv2.createCLAHE(clipLimit=float(self.params['clahe_clip']),
                                tileGridSize=(grid_size, grid_size))
        enhanced = clahe.apply(gray)
        
        # Circle Detection
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
            no_circle_img = original.copy()
            cv2.putText(no_circle_img, 'NO CIRCLE DETECTED', (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera & Result', no_circle_img)
            return None, None
        
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        cx, cy, radius = int(x), int(y), int(r)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Isolate on black background
        isolated = cv2.bitwise_and(img, img, mask=mask)
        
        # Apply CLAHE
        gray_final = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
        enhanced_final = clahe.apply(gray_final)
        
        # Autoscale
        current_avg = np.mean(enhanced_final[mask == 255])
        if current_avg > 0:
            scale_factor = self.params['target_intensity'] / current_avg
            autoscaled = np.clip(enhanced_final.astype(float) * scale_factor, 0, 255).astype(np.uint8)
        else:
            autoscaled = enhanced_final
        
        # Show only final result if preview enabled
        if self.show_previews:
            cv2.imshow('Final (After Autoscale)', autoscaled)
        
        # Crop and resize to 100x100
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(img.shape[1], cx + radius)
        y2 = min(img.shape[0], cy + radius)
        
        cropped = autoscaled[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        cropped_masked = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
        resized = cv2.resize(cropped_masked, (100, 100), interpolation=cv2.INTER_AREA)
        
        return resized, (cx, cy, radius)
    
    def run(self):
        """Main inference loop"""
        print("\nRunning real-time inference...")
        print("Press 'q' to quit")
        
        while True:
            #self.update_params()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            h, w = frame.shape[:2]
            size = min(h, w)
            y_start = (h - size) // 2
            x_start = (w - size) // 2
            frame = frame[y_start:y_start+size, x_start:x_start+size]
            
            # Resize to 500x500 if larger
            if size > 500:
                frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
            
            # Process frame
            preprocessed, circle_info = self.process_frame(frame)
            
            if preprocessed is not None and circle_info is not None:
                # Run inference
                prediction, confidence = self.classifier.predict(preprocessed)
                
                # Display result
                display_frame = frame.copy()
                cx, cy, radius = circle_info
                
                cv2.circle(display_frame, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), 3)
                
                color = (0, 255, 0) if prediction == 'heads' else (255, 0, 0)
                text1 = f'{prediction.upper()}: {confidence*100:.1f}%'
                text2 = f'radius: {radius}px'
                cv2.putText(display_frame, text1, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(display_frame, text2, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                cv2.putText(display_frame, 'Press Q to quit', (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Camera & Result', display_frame)
                
                self.show_previews = cv2.getTrackbarPos('Show Preview', 'Controls') == 1
                
                if self.show_previews:
                    if not cv2.getWindowProperty('Final (After Autoscale)', cv2.WND_PROP_VISIBLE) < 0:
                        cv2.namedWindow('Final (After Autoscale)')
                    cv2.imshow('Final (After Autoscale)', preprocessed)
                else:
                    try:
                        cv2.destroyWindow('Final (After Autoscale)')
                    except:
                        pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Inference stopped")


if __name__ == "__main__":
    onnx_model_path = 'coin_classifier.onnx'  # UPDATE THIS if needed
    camera_index = 0
    
    # Set to False for better performance (disables preview windows)
    show_previews = True  # Change to False to disable preview windows
    
    inference = RealtimeInference(onnx_model_path, camera_index, show_previews)
    inference.run()