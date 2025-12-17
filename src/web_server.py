"""
Trash Bin Detection - Web Server Module
Flask-based web interface for monitoring and control
"""

import logging
from threading import Thread
from datetime import datetime

from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)


class WebServer:
    """Web server for live monitoring and control."""
    
    def __init__(self, config: dict, detector):
        """Initialize web server."""
        self.config = config
        self.detector = detector
        self.server_config = config.get('server', {})
        
        self.host = self.server_config.get('host', '0.0.0.0')
        self.port = self.server_config.get('port', 5000)
        self.debug = self.server_config.get('debug', False)
        
        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder='../templates',
            static_folder='../static'
        )
        self.app.config['SECRET_KEY'] = 'trash-bin-detector-secret'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Register routes
        self._register_routes()
        
        self.thread = None
        self.running = False
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/status')
        def api_status():
            """Get current status."""
            return jsonify(self.detector.get_status())
        
        @self.app.route('/api/detections')
        def api_detections():
            """Get current detections."""
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'detections': self.detector.current_detections
            })
        
        @self.app.route('/api/config')
        def api_config():
            """Get current configuration."""
            return jsonify(self.config)
        
        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update configuration."""
            try:
                new_config = request.json
                # Update relevant config sections
                if 'confidence_threshold' in new_config:
                    self.config['model']['confidence_threshold'] = float(new_config['confidence_threshold'])
                    self.detector.model.confidence = float(new_config['confidence_threshold'])
                
                return jsonify({'success': True, 'message': 'Configuration updated'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/api/snapshot')
        def api_snapshot():
            """Capture and save current frame."""
            try:
                frame = self.detector.current_frame
                if frame is not None:
                    import cv2
                    from pathlib import Path
                    
                    output_dir = Path(self.config['output']['output_dir'])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{timestamp}.jpg"
                    filepath = output_dir / filename
                    
                    cv2.imwrite(str(filepath), frame)
                    
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'path': str(filepath)
                    })
                else:
                    return jsonify({'success': False, 'error': 'No frame available'}), 400
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/system')
        def api_system():
            """Get system information."""
            from src.utils import get_system_info
            return jsonify(get_system_info())
    
    def _generate_frames(self):
        """Generate frames for video streaming."""
        while self.running:
            frame_bytes = self.detector.get_frame()
            
            if frame_bytes is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            else:
                # Send a placeholder frame
                import time
                time.sleep(0.1)
    
    def start(self):
        """Start the web server in a separate thread."""
        self.running = True
        self.thread = Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"Web server started at http://{self.host}:{self.port}")
    
    def _run_server(self):
        """Run the Flask server."""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            logger.error(f"Web server error: {e}")
    
    def stop(self):
        """Stop the web server."""
        self.running = False
        logger.info("Web server stopped")
