#!/usr/bin/env python3
"""
Trash Bin Detection System for Raspberry Pi 4 64-bit
Uses ONNX Runtime + Flask Web Server for visualization
Buzzer ON when garbage container detected (>=70% confidence)
Buzzer OFF when not detected
"""

import cv2
import numpy as np
import time
import sys
import os
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
BUZZER_PIN = 17
CONFIDENCE_THRESHOLD = 0.7  # 70% threshold
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CLASS_NAMES = ["garbage container"]
WEB_PORT = 5000

# --- GLOBAL STATE ---
current_frame = None
frame_lock = Lock()
buzzer_state = False
detection_info = {"count": 0, "fps": 0, "detections": []}

# --- FLASK APP ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trash Bin Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white; 
            margin: 0; 
            padding: 20px;
            text-align: center;
        }
        h1 { color: #00ff88; }
        .container { max-width: 900px; margin: 0 auto; }
        .video-container { 
            background: #16213e; 
            padding: 10px; 
            border-radius: 10px;
            margin: 20px 0;
        }
        img { max-width: 100%; border-radius: 5px; }
        .status { 
            padding: 15px; 
            border-radius: 10px; 
            margin: 10px 0;
            font-size: 18px;
        }
        .buzzer-on { background: #ff4444; }
        .buzzer-off { background: #44aa44; }
        .info { 
            display: flex; 
            justify-content: space-around; 
            flex-wrap: wrap;
        }
        .info-box { 
            background: #16213e; 
            padding: 15px 30px; 
            border-radius: 10px; 
            margin: 10px;
        }
        .info-value { font-size: 24px; color: #00ff88; }
    </style>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('buzzer-status').className = 
                        'status ' + (data.buzzer_on ? 'buzzer-on' : 'buzzer-off');
                    document.getElementById('buzzer-status').innerText = 
                        'BUZZER: ' + (data.buzzer_on ? 'ON - DETECTED!' : 'OFF');
                    document.getElementById('fps').innerText = data.fps.toFixed(1);
                    document.getElementById('detections').innerText = data.detection_count;
                });
        }
        setInterval(updateStatus, 500);
    </script>
</head>
<body>
    <div class="container">
        <h1>🗑️ Trash Bin Detection System</h1>
        <p>Raspberry Pi 4 | Threshold: 70% | Buzzer: GPIO17</p>
        
        <div id="buzzer-status" class="status buzzer-off">BUZZER: OFF</div>
        
        <div class="video-container">
            <img src="/video_feed" alt="Video Stream">
        </div>
        
        <div class="info">
            <div class="info-box">
                <div>FPS</div>
                <div class="info-value" id="fps">0</div>
            </div>
            <div class="info-box">
                <div>Detections</div>
                <div class="info-value" id="detections">0</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify({
        'buzzer_on': buzzer_state,
        'fps': detection_info['fps'],
        'detection_count': detection_info['count'],
        'detections': detection_info['detections']
    })

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            frame = current_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- SETUP BUZZER ---
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
session = None
input_name = None
output_names = None

def load_model():
    global session, input_name, output_names, MODEL_PATH
    
    print(f"Loading model: {MODEL_PATH}...")
    try:
        import onnxruntime as ort
        
        if not os.path.exists(MODEL_PATH):
            if os.path.exists(f"models/{MODEL_PATH}"):
                MODEL_PATH = f"models/{MODEL_PATH}"
            else:
                print(f"✗ Model not found: {MODEL_PATH}")
                sys.exit(1)
        
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        
        model_outputs = session.get_outputs()
        output_names = [o.name for o in model_outputs]
        
        print(f"✓ Model loaded! Input: {input_name}")
        return True
        
    except ImportError:
        print("✗ onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

# --- PREPROCESSING ---
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

# --- POST-PROCESSING ---
def postprocess(outputs, frame_width, frame_height, conf_threshold=0.5):
    detections = []
    output = outputs[0]
    
    if len(output.shape) == 3:
        output = output[0].T
    
    x_scale = frame_width / INPUT_WIDTH
    y_scale = frame_height / INPUT_HEIGHT
    
    for detection in output:
        if len(detection) == 5:
            cx, cy, w, h, confidence = detection
            class_id = 0
        else:
            cx, cy, w, h = detection[:4]
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
        
        if confidence >= conf_threshold:
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
    """Initialize camera with Picamera2 first, then fallbacks"""
    
    # Option 1: Try Picamera2 directly (best for Pi Camera)
    try:
        from picamera2 import Picamera2
        print("Trying Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)  # Warm up
        print("✓ Pi Camera initialized via Picamera2")
        return ('picamera2', picam2)
    except Exception as e:
        print(f"Picamera2 failed: {e}")
    
    # Option 2: GStreamer pipeline
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=15/1 ! "
        "videoconvert ! "
        "appsink drop=1"
    )
    print("Trying GStreamer...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print("✓ Camera initialized via GStreamer")
            return ('opencv', cap)
        cap.release()
    
    # Option 3: V4L2 with different settings
    print("Trying V4L2...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)
        ret, _ = cap.read()
        if ret:
            print("✓ Camera initialized via V4L2")
            return ('opencv', cap)
        cap.release()
    
    # Option 4: Default
    print("Trying default camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, _ = cap.read()
        if ret:
            print("✓ Camera initialized")
            return ('opencv', cap)
        cap.release()
    
    return (None, None)

def read_frame(camera_type, camera):
    """Read frame from camera"""
    if camera_type == 'picamera2':
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame
    else:
        return camera.read()

# --- DETECTION LOOP ---
def detection_loop(camera_type, camera):
    global current_frame, buzzer_state, detection_info
    
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    total_detections = 0
    
    print("\n" + "="*60)
    print("Starting detection loop...")
    print(f"Web interface: http://<raspberry_pi_ip>:{WEB_PORT}")
    print("Press Ctrl+C to quit")
    print("="*60 + "\n")
    
    while True:
        try:
            ret, frame = read_frame(camera_type, camera)
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            
            # Preprocess and run inference
            input_tensor = preprocess(frame)
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # Post-process
            detections = postprocess(outputs, frame_width, frame_height, conf_threshold=0.5)
            
            # Filter by threshold
            high_conf = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
            
            # Buzzer control
            if len(high_conf) > 0:
                if not buzzer_state:
                    buzzer_on()
                    buzzer_state = True
                    print(f"🔔 BUZZER ON - {high_conf[0]['class_name']} ({high_conf[0]['confidence']*100:.1f}%)")
                total_detections += 1
            else:
                if buzzer_state:
                    buzzer_off()
                    buzzer_state = False
                    print("🔕 BUZZER OFF")
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']}: {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate FPS
            if frame_count % 15 == 0:
                fps = 15 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Draw status
            status_color = (0, 0, 255) if buzzer_state else (0, 255, 0)
            cv2.putText(frame, f"BUZZER: {'ON' if buzzer_state else 'OFF'}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update global state
            with frame_lock:
                current_frame = frame
            
            detection_info['fps'] = fps
            detection_info['count'] = total_detections
            detection_info['detections'] = [{'name': d['class_name'], 'conf': d['confidence']} for d in high_conf]
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

# --- MAIN ---
def main():
    print("\n" + "="*60)
    print("TRASH BIN DETECTION SYSTEM")
    print("Raspberry Pi 4 64-bit + Web Server")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Buzzer: GPIO{BUZZER_PIN}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Web Port: {WEB_PORT}")
    print("="*60 + "\n")
    
    # Load model
    if not load_model():
        sys.exit(1)
    
    # Initialize camera
    camera_type, camera = init_camera()
    if camera is None:
        print("✗ Failed to open camera!")
        print("\nTroubleshooting:")
        print("  1. sudo pkill -9 libcamera")
        print("  2. sudo raspi-config -> Interface -> Camera -> Enable")
        print("  3. sudo reboot")
        sys.exit(1)
    
    # Test buzzer
    if buzzer:
        print("Testing buzzer...")
        buzzer_on()
        time.sleep(0.2)
        buzzer_off()
        print("✓ Buzzer test complete")
    
    # Start detection thread
    detection_thread = Thread(target=detection_loop, args=(camera_type, camera), daemon=True)
    detection_thread.start()
    
    # Start web server
    try:
        print(f"\n🌐 Web server starting on port {WEB_PORT}...")
        print(f"   Open http://<raspberry_pi_ip>:{WEB_PORT} in browser")
        app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        buzzer_off()
        if camera_type == 'picamera2':
            camera.stop()
            camera.close()
        else:
            camera.release()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
