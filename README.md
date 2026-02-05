# Automatic Number Plate Recognition (ANPR) System

A real-time object detection system using YOLO (You Only Look Once) for detecting vehicles and license plates from live video streams with OCR capabilities.

## Features

- **Real-time Detection**: Low-latency inference system optimized for traffic monitoring
- **YOLO-based Detection**: State-of-the-art object detection for vehicles and license plates
- **OCR Integration**: Text extraction from license plates using EasyOCR/Pytesseract
- **Data Augmentation**: Comprehensive preprocessing pipeline for diverse lighting and traffic conditions
- **Web Interface**: Flask-based monitoring dashboard
- **Multiple Input Modes**: Support for webcam, video files, and images
- **Performance Metrics**: FPS and latency monitoring

## Architecture

```
├── src/
│   ├── detection/       # YOLO-based detection module
│   ├── ocr/            # OCR engine for plate text extraction
│   ├── preprocessing/  # Data augmentation and dataset utilities
│   └── utils/          # Video processor and logging utilities
├── templates/          # Web interface templates
├── static/             # CSS and JavaScript files
├── data/               # Dataset directory
├── models/             # Trained model weights
├── outputs/            # Detection results
└── logs/               # System logs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
cd AutomaticNumberPlateDetection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract (if using Pytesseract):
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

5. **Add a video file for testing:**
   - Place your test video file (with vehicles and license plates) in the project root directory
   - Name it `sample.mp4` or update the path in `config.yaml`
   - Recommended: Use videos with clear license plate visibility

## Windows Setup Guide

### Step-by-Step Instructions for Windows Users

#### 1. Install Python
- Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
- **Important:** During installation, check "Add Python to PATH"
- Verify installation:
  ```cmd
  python --version
  ```

#### 2. Install Git (Optional but recommended)
- Download Git from [git-scm.com](https://git-scm.com/download/win)
- Install with default settings
- Verify installation:
  ```cmd
  git --version
  ```

#### 3. Clone the Repository
Open Command Prompt or PowerShell:
```cmd
git clone https://github.com/y22cd1/ANPR.git
cd vehicleDetection
```

Or download as ZIP:
- Click "Code" → "Download ZIP" on GitHub
- Extract to your desired location
- Open Command Prompt in that folder

#### 4. Create Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command prompt.

#### 5. Install Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This will download approximately 2-3 GB of packages including PyTorch, OpenCV, and other ML libraries. Ensure you have a stable internet connection.

#### 6. Add a Test Video File
- Place a video file with vehicles in the project root directory
- Rename it to `sample.mp4` or update `config.yaml`:
  ```yaml
  video:
    input_source: "path/to/your/video.mp4"
  ```

#### 7. Verify Model File
The pre-trained license plate detection model (`models/best.pt`) should already be included. If missing, download it from the releases page.

#### 8. Run the Application

**Web Interface (Recommended):**
```cmd
python run.py --mode web
```
Then open your browser to: `http://localhost:5000`

**Process a Video File:**
```cmd
python run.py --mode video --input sample.mp4
```

**Use Webcam:**
```cmd
python run.py --mode realtime
```

**Process an Image:**
```cmd
python run.py --mode image --input path/to/image.jpg
```

#### 9. Troubleshooting Windows Issues

**Issue: "python is not recognized"**
- Python is not in PATH. Reinstall Python with "Add to PATH" checked
- Or manually add Python to PATH in System Environment Variables

**Issue: pip install fails with SSL errors**
```cmd
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Issue: PyTorch installation fails**
- Install PyTorch separately first:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Then install remaining requirements:
```cmd
pip install -r requirements.txt
```

**Issue: "No module named 'cv2'"**
```cmd
pip install opencv-python
```

**Issue: Port 5000 already in use**
- Change the port in `config.yaml`:
```yaml
api:
  port: 8080  # or any other available port
```

**Issue: Video not displaying in web interface**
- Ensure your video file path is correct in `config.yaml`
- Check that the video file is in a supported format (mp4, avi, mov)
- Look at the console logs for error messages

**Issue: GPU not detected**
- The system will automatically use CPU if CUDA is not available
- To use GPU, install CUDA toolkit and GPU-enabled PyTorch:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 10. Deactivating Virtual Environment
When you're done:
```cmd
deactivate
```

### System Requirements (Windows)

**Minimum:**
- Windows 10 or 11
- Python 3.8+
- 8 GB RAM
- 5 GB free disk space
- Intel i5 or equivalent

**Recommended:**
- Windows 10/11
- Python 3.9+
- 16 GB RAM
- 10 GB free disk space
- NVIDIA GPU with CUDA support
- Intel i7 or equivalent

### File Paths on Windows

When specifying file paths in `config.yaml`, use one of these formats:
```yaml
# Forward slashes (recommended)
input_source: "C:/Users/YourName/Videos/sample.mp4"

# Double backslashes
input_source: "C:\\Users\\YourName\\Videos\\sample.mp4"

# Raw string with single backslash (in Python code only)
# Not applicable in YAML files
```

## Quick Start

### 1. Real-time Detection (Webcam)

```bash
python run.py --mode realtime
```

Press 'q' to quit, 's' to save current frame.

### 2. Process Video File

```bash
python run.py --mode video --input path/to/video.mp4
```

### 3. Process Single Image

```bash
python run.py --mode image --input path/to/image.jpg
```

### 4. Web Interface

```bash
python run.py --mode web
```

Then open your browser to: `http://localhost:5000`

## Training Custom Model

### Prepare Dataset

1. Organize your dataset in YOLO format:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. Label format (YOLO format - one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized (0-1):
- `class_id`: 0 for vehicle, 1 for license_plate
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Dimensions of bounding box

### Train Model

```bash
python train.py --mode train
```

### Evaluate Model

```bash
python train.py --mode eval
```

### Export Model

```bash
python train.py --mode export
```

## Configuration

Edit `config.yaml` to customize:

- **Model settings**: Model size, confidence threshold, device
- **OCR settings**: Engine selection, language, confidence threshold
- **Video settings**: Input source, resolution, FPS
- **Augmentation**: Brightness, contrast, rotation, noise parameters
- **Training**: Epochs, batch size, learning rate

Example configuration:

```yaml
model:
  name: "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  confidence_threshold: 0.5
  device: "cuda"  # or "cpu"

ocr:
  engine: "easyocr"  # or "pytesseract"
  languages: ['en']
  min_confidence: 0.6

video:
  input_source: 0  # 0 for webcam, or path to video
  fps: 30
  resolution: [1280, 720]
```

## Usage Examples

### Python API

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

# Load image
image = cv2.imread('test_image.jpg')

# Detect
detections = detector.detect(image)

# Extract plate text
for det in detections:
    if 'plate' in det['class_name'].lower():
        x1, y1, x2, y2 = det['bbox']
        plate_img = image[y1:y2, x1:x2]

        processed = detector.preprocess_plate(plate_img)
        result = ocr_engine.read_text(processed)

        print(f"Plate: {result['text']}, Confidence: {result['confidence']}")
```

### Data Augmentation

```python
from src.preprocessing import DataAugmentation

# Initialize augmentation
augmenter = DataAugmentation(config)

# Augment image with bounding boxes
aug_image, aug_bboxes, aug_labels = augmenter.augment_image(
    image=image,
    bboxes=[[x1, y1, x2, y2]],
    class_labels=[1],
    mode='train'
)
```

## Performance

Typical performance on modern hardware:

| Model   | GPU        | FPS  | Latency (ms) |
|---------|------------|------|--------------|
| YOLOv8n | RTX 3060   | 120  | 8.3          |
| YOLOv8s | RTX 3060   | 85   | 11.8         |
| YOLOv8m | RTX 3060   | 60   | 16.7         |
| YOLOv8n | CPU (i7)   | 25   | 40           |

## Dataset Preparation Tips

1. **Diverse Conditions**: Include images from:
   - Different times of day (morning, noon, evening, night)
   - Various weather conditions (sunny, rainy, foggy)
   - Multiple camera angles and distances
   - Different lighting conditions

2. **Quality**:
   - Minimum resolution: 640x640
   - Clear, focused images
   - Proper bounding box annotations

3. **Augmentation**: The system automatically applies:
   - Brightness/contrast adjustments
   - Motion blur and noise
   - Geometric transformations
   - Weather simulations (rain, sun flare, shadows)

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Low FPS

1. Use smaller model (yolov8n instead of yolov8l)
2. Reduce input resolution
3. Enable GPU acceleration
4. Reduce confidence threshold for fewer detections

### Poor OCR Accuracy

1. Adjust preprocessing in `detector.preprocess_plate()`
2. Try different OCR engine (EasyOCR vs Pytesseract)
3. Increase minimum confidence threshold
4. Add more training data with clear plates

### Camera Not Detected

Change input source in `config.yaml`:
```yaml
video:
  input_source: 1  # Try different camera indices
```

## Project Structure Details

### Detection Module
- `detector.py`: YOLO-based detection with preprocessing
- Handles vehicle and license plate detection
- Low-latency inference optimization

### OCR Module
- `ocr_engine.py`: Text extraction from plates
- Supports EasyOCR and Pytesseract
- Post-processing and validation

### Preprocessing Module
- `augmentation.py`: Data augmentation pipeline
- `dataset.py`: Dataset loading and splitting

### Utils
- `video_processor.py`: Real-time video processing
- `logger.py`: Logging configuration

## API Endpoints (Web Mode)

- `GET /`: Main dashboard
- `GET /video_feed`: Live video stream
- `GET /api/detections`: Recent detections (JSON)
- `GET /api/statistics`: System statistics (JSON)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Albumentations](https://github.com/albumentations-team/albumentations)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{anpr_system,
  title = {Automatic Number Plate Recognition System},
  year = {2025},
  author = {Your Name},
  description = {Real-time YOLO-based ANPR with OCR capabilities}
}
```

## Contact

For questions and support, please open an issue on GitHub.

## Future Enhancements

- [ ] Multi-language plate recognition
- [ ] Database integration for plate logging
- [ ] REST API for external integrations
- [ ] Mobile app support
- [ ] Cloud deployment options
- [ ] Advanced analytics dashboard
- [ ] Vehicle make/model recognition
- [ ] Speed estimation

