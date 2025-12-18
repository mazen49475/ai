#!/usr/bin/env python3
"""
Trash Bin Detection System for Raspberry Pi 4 64-bit
ONNX Runtime + Flask Web Server with Adjustable Threshold
Buzzer ON when confidence >= threshold, OFF when below
"""

import cv2
import numpy as np
import time
import sys
import os
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify, request

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
BUZZER_PIN = 17
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CLASS_NAMES = ["garbage container"]
WEB_PORT = 5000

# --- ADJUSTABLE THRESHOLD (default 0.8 = 80%) ---
confidence_threshold = 0.8
threshold_lock = Lock()

# --- GLOBAL STATE ---
current_frame = None
frame_lock = Lock()
buzzer_state = False
detection_info = {"count": 0, "fps": 0, "detections": [], "threshold": 0.8}

# --- FLASK APP ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trash Bin Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white; 
            margin: 0; 
            padding: 10px;
            text-align: center;
        }
        h1 { color: #00ff88; margin: 10px 0; font-size: 1.5em; }
        .container { max-width: 900px; margin: 0 auto; }
        .video-container { 
            background: #16213e; 
            padding: 5px; 
            border-radius: 10px;
            margin: 10px 0;
        }
        img { max-width: 100%; border-radius: 5px; }
        .status { 
            padding: 12px; 
            border-radius: 10px; 
            margin: 8px 0;
            font-size: 16px;
            font-weight: bold;
        }
        .buzzer-on { background: #ff4444; animation: pulse 0.5s infinite; }
        .buzzer-off { background: #44aa44; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
        .info { 
            display: flex; 
            justify-content: space-around; 
            flex-wrap: wrap;
        }
        .info-box { 
            background: #16213e; 
            padding: 10px 20px; 
            border-radius: 10px; 
            margin: 5px;
            min-width: 100px;
        }
        .info-value { font-size: 20px; color: #00ff88; }
        .threshold-control {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .slider-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin: 10px 0;
        }
        input[type="range"] {
            width: 200px;
            height: 20px;
            cursor: pointer;
        }
        .threshold-value {
            font-size: 28px;
            color: #ffaa00;
            font-weight: bold;
            min-width: 80px;
        }
        .btn {
            background: #00ff88;
            color: #1a1a2e;
            border: none;
            padding: 10px 25px;
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer;
            margin: 5px;
            font-weight: bold;
        }
        .btn:hover { background: #00cc66; }
        .btn-preset { background: #4488ff; color: white; }
        .btn-preset:hover { background: #3366cc; }
    </style>
    <script>
        let currentThreshold = 0.8;
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('buzzer-status').className = 
                        'status ' + (data.buzzer_on ? 'buzzer-on' : 'buzzer-off');
                    document.getElementById('buzzer-status').innerText = 
                        'BUZZER: ' + (data.buzzer_on ? '🔔 ON - DETECTED!' : '🔕 OFF');
                    document.getElementById('fps').innerText = data.fps.toFixed(1);
                    document.getElementById('detections').innerText = data.detection_count;
                    currentThreshold = data.threshold;
                    document.getElementById('threshold-display').innerText = (data.threshold * 100).toFixed(0) + '%';
                    document.getElementById('threshold-slider').value = data.threshold * 100;
                });
        }
        
        function setThreshold(value) {
            const threshold = value / 100;
            fetch('/set_threshold', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({threshold: threshold})
            }).then(() => {
                document.getElementById('threshold-display').innerText = value + '%';
            });
        }
        
        function presetThreshold(value) {
            document.getElementById('threshold-slider').value = value;
            setThreshold(value);
        }
        
        setInterval(updateStatus, 500);
    </script>
</head>
<body>
    <div class="container">
        <h1>🗑️ Trash Bin Detection</h1>
        
        <div id="buzzer-status" class="status buzzer-off">🔕 BUZZER: OFF</div>
        
        <div class="threshold-control">
            <div style="margin-bottom: 10px;">⚙️ <b>Confidence Threshold</b></div>
            <div class="slider-container">
                <span>50%</span>
                <input type="range" id="threshold-slider" min="50" max="95" value="80" 
                       onchange="setThreshold(this.value)" oninput="document.getElementById('threshold-display').innerText = this.value + '%'">
                <span>95%</span>
            </div>
            <div class="threshold-value" id="threshold-display">80%</div>
            <div style="margin-top: 10px;">
                <button class="btn btn-preset" onclick="presetThreshold(70)">70%</button>
                <button class="btn btn-preset" onclick="presetThreshold(80)">80%</button>
                <button class="btn btn-preset" onclick="presetThreshold(90)">90%</button>
            </div>
        </div>
        
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
        
        <p style="color: #888; font-size: 12px; margin-top: 15px;">
            Raspberry Pi 4 | GPIO17 Buzzer | ONNX Runtime
        </p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    with threshold_lock:
        thresh = confidence_threshold
    return jsonify({
        'buzzer_on': buzzer_state,
        'fps': detection_info['fps'],
        'detection_count': detection_info['count'],
        'detections': detection_info['detections'],
        'threshold': thresh
    })

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global confidence_threshold
    data = request.get_json()
    new_threshold = float(data.get('threshold', 0.8))
    new_threshold = max(0.5, min(0.95, new_threshold))  # Clamp between 50-95%
    with threshold_lock:
        confidence_threshold = new_threshold
    print(f"⚙️ Threshold changed to: {new_threshold*100:.0f}%")
    return jsonify({'success': True, 'threshold': new_threshold})

@app.route('/get_threshold')
def get_threshold():
    with threshold_lock:
        thresh = confidence_threshold
    return jsonify({'threshold': thresh})

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            frame = current_frame.copy()
        
        # Lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~20fps stream

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
    print("⚠ gpiozero not installed")
except Exception as e:
    print(f"⚠ GPIO error: {e}")

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
        
        # Set optimization options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Use 4 cores
        sess_options.inter_op_num_threads = 4
        
        if not os.path.exists(MODEL_PATH):
            if os.path.exists(f"models/{MODEL_PATH}"):
                MODEL_PATH = f"models/{MODEL_PATH}"
            else:
                print(f"✗ Model not found: {MODEL_PATH}")
                return False
        
        session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        
        print(f"✓ Model loaded! (Optimized for 4 cores)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

# --- PREPROCESSING (Optimized) ---
def preprocess(frame):
    # Resize with faster interpolation
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0/255.0)  # Faster than /255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])
    return img

# --- POST-PROCESSING (Optimized) ---
def postprocess(outputs, frame_width, frame_height, conf_threshold=0.5):
    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0].T
    
    # Fast filtering using numpy
    if len(output[0]) == 5:
        confidences = output[:, 4]
    else:
        confidences = np.max(output[:, 4:], axis=1)
    
    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []
    
    filtered = output[mask]
    filtered_conf = confidences[mask]
    
    x_scale = frame_width / INPUT_WIDTH
    y_scale = frame_height / INPUT_HEIGHT
    
    detections = []
    for i, det in enumerate(filtered):
        cx, cy, w, h = det[:4]
        
        if len(det) == 5:
            class_id = 0
            conf = det[4]
        else:
            class_id = int(np.argmax(det[4:]))
            conf = filtered_conf[i]
        
        x1 = int((cx - w/2) * x_scale)
        y1 = int((cy - h/2) * y_scale)
        x2 = int((cx + w/2) * x_scale)
        y2 = int((cy + h/2) * y_scale)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        })
    
    # NMS
    if len(detections) > 1:
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.45)
        if len(indices) > 0:
            detections = [detections[i] for i in indices.flatten()]
    
    return detections

# --- CAMERA SETUP ---
def init_camera():
    # Try Picamera2 first
    try:
        from picamera2 import Picamera2
        print("Trying Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=2
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("✓ Pi Camera initialized")
        return ('picamera2', picam2)
    except Exception as e:
        print(f"Picamera2 failed: {e}")
    
    # Fallback to OpenCV
    for idx in [0, 1, 2]:
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)
            ret, _ = cap.read()
            if ret:
                print(f"✓ Camera {idx} initialized")
                return ('opencv', cap)
            cap.release()
    
    return (None, None)

def read_frame(camera_type, camera):
    if camera_type == 'picamera2':
        frame = camera.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        return camera.read()

# --- DETECTION LOOP ---
def detection_loop(camera_type, camera):
    global current_frame, buzzer_state, detection_info, confidence_threshold
    
    frame_count = 0
    fps = 0
    fps_start = time.time()
    total_detections = 0
    skip_frames = 0  # Process every frame for responsiveness
    
    print("\n" + "="*60)
    print("Detection loop started")
    print(f"Web: http://<pi_ip>:{WEB_PORT}")
    print("="*60 + "\n")
    
    while True:
        try:
            ret, frame = read_frame(camera_type, camera)
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Get current threshold
            with threshold_lock:
                thresh = confidence_threshold
            
            # Run inference
            input_tensor = preprocess(frame)
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # Post-process with lower threshold to show all detections
            detections = postprocess(outputs, w, h, conf_threshold=0.5)
            
            # Filter for buzzer (only >= threshold)
            high_conf = [d for d in detections if d['confidence'] >= thresh]
            
            # Buzzer control
            if len(high_conf) > 0:
                if not buzzer_state:
                    buzzer_on()
                    buzzer_state = True
                    print(f"🔔 ON: {high_conf[0]['class_name']} ({high_conf[0]['confidence']*100:.0f}%) >= {thresh*100:.0f}%")
                total_detections += 1
            else:
                if buzzer_state:
                    buzzer_off()
                    buzzer_state = False
                    print(f"🔕 OFF: No detection >= {thresh*100:.0f}%")
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                # Green if >= threshold, yellow if below
                color = (0, 255, 0) if conf >= thresh else (0, 200, 255)
                thickness = 3 if conf >= thresh else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{conf*100:.0f}%"
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # FPS calculation
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Draw status
            cv2.putText(frame, f"{'BUZZER ON' if buzzer_state else 'BUZZER OFF'}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if buzzer_state else (0,255,0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | Thresh: {thresh*100:.0f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Update global state
            with frame_lock:
                current_frame = frame
            
            detection_info['fps'] = fps
            detection_info['count'] = total_detections
            detection_info['threshold'] = thresh
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

# --- MAIN ---
def main():
    global confidence_threshold
    
    print("\n" + "="*60)
    print("TRASH BIN DETECTION SYSTEM")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Buzzer: GPIO{BUZZER_PIN}")
    print(f"Default Threshold: {confidence_threshold*100:.0f}%")
    print(f"Web Port: {WEB_PORT}")
    print("="*60 + "\n")
    
    if not load_model():
        sys.exit(1)
    
    camera_type, camera = init_camera()
    if camera is None:
        print("✗ Camera failed!")
        print("Run: sudo pkill -9 libcamera && sudo reboot")
        sys.exit(1)
    
    if buzzer:
        print("Testing buzzer...")
        buzzer_on()
        time.sleep(0.15)
        buzzer_off()
    
    # Start detection thread
    Thread(target=detection_loop, args=(camera_type, camera), daemon=True).start()
    
    # Start web server
    try:
        print(f"\n🌐 Web server: http://<pi_ip>:{WEB_PORT}")
        app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        buzzer_off()
        if camera_type == 'picamera2':
            camera.stop()
            camera.close()
        else:
            camera.release()
        print("✓ Done")

if __name__ == "__main__":
    main()
