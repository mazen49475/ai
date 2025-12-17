# Trash Bin Detection System

Real-time trash bin detection using YOLOv8 on Raspberry Pi 4 B with Pi Camera.

## Features

- Real-time trash bin detection using custom YOLOv8 model
- Optimized for Raspberry Pi 4 B
- Support for Raspberry Pi Camera Module
- Web interface for live monitoring
- Configurable detection parameters
- Logging and alert system

## Hardware Requirements

- Raspberry Pi 4 Model B (4GB+ RAM recommended)
- Raspberry Pi Camera Module (v2 or HQ Camera)
- MicroSD Card (32GB+ recommended)
- Power Supply (5V 3A)
- Optional: Case with camera mount

## Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- OpenCV
- Ultralytics YOLOv8
- Picamera2

## Quick Start

### On Raspberry Pi

1. Clone this repository:
```bash
git clone https://github.com/mazen49475/ai.git
cd ai
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Start detection:
```bash
source venv/bin/activate
python src/detector.py
```

### Web Interface

After starting the detector, access the web interface at:
```
http://<raspberry-pi-ip>:5000
```

## Project Structure

```
ai/
├── README.md
├── setup.sh                 # Auto-installation script for Raspberry Pi
├── requirements.txt         # Python dependencies
├── config/
│   └── config.yaml         # Configuration file
├── models/
│   └── .gitkeep            # Place best.pt here
├── src/
│   ├── __init__.py
│   ├── detector.py         # Main detection script
│   ├── camera.py           # Camera handling
│   ├── model_loader.py     # Model loading utilities
│   ├── utils.py            # Helper functions
│   └── web_server.py       # Flask web interface
├── static/
│   └── style.css           # Web interface styles
├── templates/
│   └── index.html          # Web interface template
├── logs/
│   └── .gitkeep            # Detection logs
└── output/
    └── .gitkeep            # Detection outputs/screenshots
```

## Configuration

Edit `config/config.yaml` to customize:

- Detection confidence threshold
- Camera resolution
- Frame rate
- Alert settings
- Output options

## License

MIT License

## Author

Mazen
