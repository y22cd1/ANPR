# ANPR System - Quick Reference

## Installation (One-Time Setup)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python demo.py
```

## Common Commands

### Running the System

```bash
# Real-time detection from webcam
python run.py --mode realtime

# Process video file
python run.py --mode video --input path/to/video.mp4

# Process single image
python run.py --mode image --input path/to/image.jpg

# Start web interface
python run.py --mode web
# Then open: http://localhost:5000
```

### Training

```bash
# Train model
python train.py --mode train

# Evaluate model
python train.py --mode eval

# Export model (ONNX, TorchScript)
python train.py --mode export

# Complete pipeline (train + eval + export)
python train.py --mode all
```

### Demo

```bash
# Quick system test
python demo.py
```

## Keyboard Shortcuts (Real-time Mode)

- `q` - Quit application
- `s` - Save current frame

## Configuration

Edit `config.yaml` for settings:

```yaml
# Quick settings
model:
  name: "yolov8n"              # Model: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  confidence_threshold: 0.5    # Detection confidence (0-1)
  device: "cuda"               # Device: cuda or cpu

ocr:
  engine: "easyocr"            # OCR: easyocr or pytesseract
  min_confidence: 0.6          # OCR confidence (0-1)

video:
  input_source: 0              # 0=webcam, 1=external camera, or video path
  fps: 30
  resolution: [1280, 720]
```

## Project Structure

```
AutomaticNumberPlateDetection/
├── src/
│   ├── detection/          # YOLO detection module
│   ├── ocr/               # OCR engine
│   ├── preprocessing/     # Data augmentation
│   └── utils/             # Video processor, logger
├── data/                  # Dataset (train/val/test)
├── models/                # Model weights
├── outputs/               # Detection results
├── logs/                  # System logs
├── templates/             # Web UI templates
├── static/                # CSS, JavaScript
├── config.yaml           # Main configuration
├── run.py                # Main application
├── train.py              # Training script
├── app.py                # Web application
└── demo.py               # Quick demo
```

## Python API

### Basic Usage

```python
import yaml
import cv2
from src.detection import LicensePlateDetector
from src.ocr import OCREngine

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize
detector = LicensePlateDetector(config)
ocr_engine = OCREngine(config)

# Process image
image = cv2.imread('image.jpg')
detections = detector.detect(image)

# Extract plate text
for det in detections:
    if 'plate' in det['class_name'].lower():
        x1, y1, x2, y2 = det['bbox']
        plate = image[y1:y2, x1:x2]
        result = ocr_engine.read_text(plate)
        print(f"Plate: {result['text']}")
```

### Video Processing

```python
from src.utils import VideoProcessor

processor = VideoProcessor(config, detector, ocr_engine)
processor.run(display=True)
stats = processor.get_statistics()
```

## File Formats

### YOLO Label Format

Each image has a `.txt` file with:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.2    # vehicle at center
1 0.52 0.58 0.1 0.05 # plate slightly below center
```

Coordinates are normalized (0-1).

Classes:
- `0` = vehicle
- `1` = license_plate

## API Endpoints (Web Mode)

- `GET /` - Dashboard
- `GET /video_feed` - Live stream
- `GET /api/detections` - Recent detections (JSON)
- `GET /api/statistics` - Statistics (JSON)

Example:
```bash
curl http://localhost:5000/api/statistics
```

## Performance Tips

### Speed Up Detection

1. Use smaller model: `yolov8n` instead of `yolov8l`
2. Lower resolution in config
3. Enable GPU: `device: "cuda"`
4. Reduce confidence threshold

### Improve Accuracy

1. Use larger model: `yolov8m` or `yolov8l`
2. Increase confidence threshold
3. Collect more training data
4. Train longer (more epochs)

### Reduce Memory Usage

1. Smaller batch size
2. Smaller model
3. Lower resolution
4. Use CPU instead of GPU (slower but uses less VRAM)

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| CUDA not available | Set `device: "cpu"` in config |
| Webcam not working | Try `input_source: 1` or `2` |
| Out of memory | Reduce `batch_size` to 8 or 4 |
| Low FPS | Use `yolov8n` model |
| Poor OCR | Try `engine: "pytesseract"` |
| Port in use | Change `port: 5001` in config |

## Dataset Requirements

### Minimum
- 500+ images
- Clear bounding box annotations
- Mix of day/night scenes

### Recommended
- 2000+ images
- Multiple angles and distances
- Various weather conditions
- Different lighting scenarios
- Mix of vehicle types

## Model Sizes Comparison

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n | 6MB | Fastest | Good |
| yolov8s | 22MB | Fast | Better |
| yolov8m | 52MB | Medium | Great |
| yolov8l | 87MB | Slow | Excellent |
| yolov8x | 136MB | Slowest | Best |

Start with `yolov8n` for development, upgrade to `yolov8s` or `yolov8m` for production.

## Environment Variables

Optional environment variables:

```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=-1

# Set log level
export LOG_LEVEL=DEBUG

# Custom config file
export ANPR_CONFIG=custom_config.yaml
```

## Batch Processing

Process multiple files:

```bash
# Process all videos in folder
for video in videos/*.mp4; do
    python run.py --mode video --input "$video"
done

# Process all images
for img in images/*.jpg; do
    python run.py --mode image --input "$img"
done
```

## Docker Support (Future)

```bash
# Build image
docker build -t anpr-system .

# Run with webcam
docker run --device=/dev/video0 -p 5000:5000 anpr-system

# Run with GPU
docker run --gpus all -p 5000:5000 anpr-system
```

## Useful Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)

## Getting Help

1. Check logs: `logs/anpr_system.log`
2. Run demo: `python demo.py`
3. See detailed guides:
   - README.md - Full documentation
   - SETUP_GUIDE.md - Installation guide
4. Open GitHub issue with error details

## Version Info

Check installed versions:

```bash
python --version
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
```

## License

MIT License - See LICENSE file for details.
