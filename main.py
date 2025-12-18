#!/usr/bin/env python3
"""
Trash Bin Detection System for Raspberry Pi 4 64-bit
Uses ONNX Runtime for inference and gpiozero for GPIO control
Buzzer ON when garbage container detected (>=80% confidence)
Buzzer OFF when not detected
"""

import cv2
import numpy as np
import time
import sys
import os

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"  # ONNX model path
BUZZER_PIN = 17           # GPIO17 for buzzer
CONFIDENCE_THRESHOLD = 0.8  # 80% threshold for buzzer
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CLASS_NAMES = ["garbage container"]  # Single class model

# --- SETUP BUZZER (gpiozero) ---
buzzer = None
try:
    from gpiozero import Buzzer
    buzzer = Buzzer(BUZZER_PIN)
    print(f"✓ Buzzer initialized on GPIO{BUZZER_PIN}")
except ImportError:
    print("⚠ gpiozero not installed. Install with: sudo apt install python3-gpiozero")
except Exception as e:
    print(f"⚠ GPIO error: {e}. Running without buzzer.")

def buzzer_on():
    if buzzer:
        buzzer.on()

def buzzer_off():
    if buzzer:
        buzzer.off()

# --- LOAD ONNX MODEL ---
print(f"Loading model: {MODEL_PATH}...")
try:
    import onnxruntime as ort
    
    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        # Try models directory
        if os.path.exists(f"models/{MODEL_PATH}"):
            MODEL_PATH = f"models/{MODEL_PATH}"
        else:
            print(f"✗ Model not found: {MODEL_PATH}")
            sys.exit(1)
    
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # Get input/output info
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    
    model_outputs = session.get_outputs()
    output_names = [o.name for o in model_outputs]
    
    print(f"✓ Model loaded! Input: {input_name} {input_shape}")
    print(f"  Outputs: {output_names}")
    
except ImportError:
    print("✗ onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# --- PREPROCESSING ---
def preprocess(frame):
    """Prepare frame for YOLO ONNX model"""
    # Resize to model input size
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    
    # HWC to CHW format
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# --- POST-PROCESSING ---
def postprocess(outputs, frame_width, frame_height, conf_threshold=0.5):
    """
    Process YOLOv8 ONNX output
    Output shape: (1, 84, 8400) or (1, 5, 8400) for single class
    """
    detections = []
    
    output = outputs[0]  # Shape: (1, num_features, num_boxes)
    
    # Transpose to (num_boxes, num_features)
    if len(output.shape) == 3:
        output = output[0].T  # (8400, 84) or (8400, 5)
    
    # Scale factors
    x_scale = frame_width / INPUT_WIDTH
    y_scale = frame_height / INPUT_HEIGHT
    
    for detection in output:
        # For single class: [cx, cy, w, h, conf]
        # For multi class: [cx, cy, w, h, class1_conf, class2_conf, ...]
        
        if len(detection) == 5:
            # Single class model
            cx, cy, w, h, confidence = detection
            class_id = 0
        else:
            # Multi-class model
            cx, cy, w, h = detection[:4]
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
        
        if confidence >= conf_threshold:
            # Convert center format to corner format
            x1 = int((cx - w/2) * x_scale)
            y1 = int((cy - h/2) * y_scale)
            x2 = int((cx + w/2) * x_scale)
            y2 = int((cy + h/2) * y_scale)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            })
    
    # Non-Maximum Suppression
    if len(detections) > 0:
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.45)
        
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]
        else:
            detections = []
    
    return detections

# --- CAMERA SETUP ---
def init_camera():
    """Initialize camera with multiple fallback options"""
    
    # Option 1: GStreamer with libcamera (Pi Camera)
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "appsink"
    )
    
    print("Trying GStreamer pipeline for Pi Camera...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("✓ Pi Camera initialized via GStreamer")
        return cap
    
    # Option 2: Direct V4L2
    print("Trying /dev/video0...")
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Camera initialized via V4L2")
        return cap
    
    # Option 3: Default OpenCV
    print("Trying default camera (index 0)...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Camera initialized via OpenCV")
        return cap
    
    return None

# --- MAIN ---
def main():
    print("\n" + "="*60)
    print("TRASH BIN DETECTION SYSTEM")
    print("Raspberry Pi 4 64-bit")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Buzzer: GPIO{BUZZER_PIN}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print("="*60 + "\n")
    
    # Initialize camera
    cap = init_camera()
    if cap is None:
        print("✗ Failed to open camera!")
        print("\nTroubleshooting:")
        print("  1. Check camera connection")
        print("  2. Run: sudo raspi-config -> Interface -> Camera -> Enable")
        print("  3. Run: libcamera-hello --list-cameras")
        sys.exit(1)
    
    # Test buzzer
    if buzzer:
        print("Testing buzzer...")
        buzzer_on()
        time.sleep(0.2)
        buzzer_off()
        print("✓ Buzzer test complete")
    
    print("\nStarting detection... Press 'q' to quit\n")
    
    buzzer_state = False
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            
            # Preprocess
            input_tensor = preprocess(frame)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(output_names, {input_name: input_tensor})
            inference_time = (time.time() - start_time) * 1000
            
            # Post-process
            detections = postprocess(outputs, frame_width, frame_height, conf_threshold=0.5)
            
            # Filter by confidence threshold for buzzer
            high_conf_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
            
            # Buzzer control
            if len(high_conf_detections) > 0:
                if not buzzer_state:
                    buzzer_on()
                    buzzer_state = True
                    print(f"🔔 BUZZER ON - Detected: {high_conf_detections[0]['class_name']} ({high_conf_detections[0]['confidence']*100:.1f}%)")
            else:
                if buzzer_state:
                    buzzer_off()
                    buzzer_state = False
                    print("🔕 BUZZER OFF - No detection")
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                name = det['class_name']
                
                # Green if above threshold, yellow if below
                color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name}: {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Draw status
            status_color = (0, 0, 255) if buzzer_state else (0, 255, 0)
            status_text = "BUZZER: ON" if buzzer_state else "BUZZER: OFF"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Trash Bin Detection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        buzzer_off()
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Cleanup complete")

if __name__ == "__main__":
    main()
