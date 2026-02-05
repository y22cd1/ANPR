# Getting Started with ANPR System

Welcome! This guide will help you get up and running in 5 minutes.

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd AutomaticNumberPlateDetection

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Test the System (1 minute)

```bash
# Run demo to verify installation
python demo.py
```

If you see "✓ Detector initialized successfully" and "✓ OCR engine initialized successfully", you're good to go!

## Step 3: Choose Your Mode (2 minutes)

### Option A: Real-Time Detection (Recommended for First Try)

```bash
python run.py --mode realtime
```

This will:
- Open your webcam
- Detect vehicles and license plates in real-time
- Display FPS and detection results
- Press 'q' to quit, 's' to save frame

### Option B: Web Interface (Best for Monitoring)

```bash
python run.py --mode web
```

Then open your browser to: http://localhost:5000

You'll see:
- Live video feed
- Detection statistics
- Recent plate detections
- Performance metrics

### Option C: Process a File

```bash
# Process an image
python run.py --mode image --input path/to/your/image.jpg

# Process a video
python run.py --mode video --input path/to/your/video.mp4
```

## What You Should See

### Real-Time Mode
- Video window showing live feed
- Green boxes around license plates
- Blue boxes around vehicles
- Detected plate text displayed
- FPS counter in top-left

### Web Mode
- Professional dashboard
- Live video stream
- Statistics cards (total detections, unique plates)
- Scrolling list of recent detections with timestamps

## Common First-Time Issues

### "CUDA not available"
This is normal if you don't have NVIDIA GPU. The system will use CPU (slower but works fine).

### "Failed to open video source"
Try changing the camera index in config.yaml:
```yaml
video:
  input_source: 1  # Try 0, 1, 2, etc.
```

### "No detections found"
The pretrained model may not detect all plates perfectly. For best results:
1. Ensure good lighting
2. Keep plates clearly visible
3. Consider training a custom model (see below)

## Next Steps

### For Quick Testing
Just keep using the system with the pretrained model:
```bash
python run.py --mode realtime
```

### For Production Use
Train a custom model on your specific data:

1. Collect 500-1000 images with vehicles/plates
2. Annotate them using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg)
3. Export in YOLO format
4. Place in data/images/train and data/labels/train
5. Run: `python train.py --mode train`

### For Integration
Use the Python API:

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
ocr = OCREngine(config)

# Process
image = cv2.imread('car.jpg')
detections = detector.detect(image)

# Print results
for det in detections:
    if 'plate' in det['class_name'].lower():
        print(f"Found plate at {det['bbox']}")
```

## Configuration Tips

Edit `config.yaml` to customize:

```yaml
# For faster speed (lower accuracy)
model:
  name: "yolov8n"
  confidence_threshold: 0.3

# For better accuracy (slower)
model:
  name: "yolov8m"
  confidence_threshold: 0.6

# To use CPU instead of GPU
model:
  device: "cpu"

# To change OCR engine
ocr:
  engine: "pytesseract"  # or "easyocr"
```

## Understanding the Output

### Detection Format
```
class_name: license_plate
confidence: 0.87  (87% sure it's a plate)
bbox: [100, 200, 300, 250]  (x1, y1, x2, y2)
plate_text: "ABC1234"  (extracted text)
ocr_confidence: 0.92  (92% sure about text)
```

### Performance Metrics
- **FPS**: Frames processed per second (higher is better)
- **Latency**: Time to process one frame in milliseconds (lower is better)
- **Confidence**: How sure the model is (0-1 scale)

## Keyboard Shortcuts

When running in real-time mode:
- `q` - Quit
- `s` - Save current frame to detections/

## File Locations

- **Saved frames**: `detections/`
- **Processed videos**: `outputs/`
- **Logs**: `logs/anpr_system.log`
- **Model weights**: `models/best.pt`

## Need Help?

1. **Check the logs**: `cat logs/anpr_system.log`
2. **Read full docs**: See [README.md](README.md)
3. **Setup issues**: See [SETUP_GUIDE.md](SETUP_GUIDE.md)
4. **Quick commands**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## Tips for Best Results

1. **Good Lighting**: Ensure plates are well-lit and visible
2. **Proper Distance**: Not too far, not too close (3-15 meters ideal)
3. **Clean Lens**: Keep camera lens clean
4. **Stable Mount**: Reduce motion blur with stable camera
5. **Appropriate Angle**: Front or rear view works best

## Performance Expectations

With default settings (YOLOv8n):

| Hardware | FPS | Latency |
|----------|-----|---------|
| CPU (i7) | 10-15 | ~80ms |
| GTX 1060 | 40-60 | ~20ms |
| RTX 3060 | 80-120 | ~10ms |

## What's Next?

- ✅ System is running
- ⬜ Collect your own data
- ⬜ Train custom model
- ⬜ Integrate with your application
- ⬜ Deploy to production

Congratulations! You now have a working ANPR system.

Start experimenting and customize it for your needs!
