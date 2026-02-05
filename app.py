"""
Flask web application for ANPR monitoring and control.
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import yaml
import logging
from pathlib import Path
import json
from datetime import datetime

from src.detection import LicensePlateDetector
from src.ocr import OCREngine
from src.utils import setup_logger

app = Flask(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logger = setup_logger(
    name="ANPR_WEB",
    log_file=config['logging']['log_file'],
    level=config['logging']['level']
)

# Initialize components
detector = None
ocr_engine = None
video_capture = None
detection_history = []


def initialize_system():
    """Initialize detection and OCR systems."""
    global detector, ocr_engine, video_capture

    try:
        logger.info("Initializing ANPR system...")

        # Initialize detector
        detector = LicensePlateDetector(config)

        # Initialize OCR
        ocr_engine = OCREngine(config)

        # Initialize video capture
        video_source = config['video']['input_source']
        video_capture = cv2.VideoCapture(video_source)

        logger.info("ANPR system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return False


def generate_frames():
    """Generate video frames with detections."""
    global detection_history

    while True:
        success, frame = video_capture.read()

        if not success:
            # Loop the video by resetting to the beginning
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect vehicles and plates
        detections = detector.detect(frame)

        # Process license plates
        for det in detections:
            if det['class_id'] == 1 or 'plate' in det['class_name'].lower():
                x1, y1, x2, y2 = det['bbox']
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size > 0:
                    # Preprocess and run OCR
                    processed = detector.preprocess_plate(plate_img)
                    ocr_result = ocr_engine.read_text(processed)

                    if ocr_result['text']:
                        det['plate_text'] = ocr_result['text']
                        det['ocr_confidence'] = ocr_result['confidence']

                        # Add to history
                        detection_entry = {
                            'text': ocr_result['text'],
                            'confidence': ocr_result['confidence'],
                            'timestamp': datetime.now().isoformat(),
                            'bbox': det['bbox']
                        }
                        detection_history.append(detection_entry)

                        # Keep only recent detections
                        if len(detection_history) > 100:
                            detection_history = detection_history[-100:]

        # Draw detections
        annotated = draw_detections(frame, detections)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def draw_detections(frame, detections):
    """Draw detections on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']

        # Color based on class
        color = (0, 255, 0) if 'plate' in class_name.lower() else (255, 0, 0)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{class_name}: {confidence:.2f}"
        if 'plate_text' in det:
            label += f" | {det['plate_text']}"

        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return annotated


@app.route('/')
def index():
    """Main page."""
    # Initialize system if not already initialized
    if detector is None:
        initialize_system()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/detections')
def get_detections():
    """Get recent detections."""
    return jsonify({
        'detections': detection_history[-20:],
        'total': len(detection_history)
    })


@app.route('/api/statistics')
def get_statistics():
    """Get system statistics."""
    unique_plates = len(set(d['text'] for d in detection_history))

    return jsonify({
        'total_detections': len(detection_history),
        'unique_plates': unique_plates,
        'recent_detections': detection_history[-10:]
    })


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration."""
    if request.method == 'GET':
        return jsonify(config)
    else:
        # Update configuration
        new_config = request.json
        # TODO: Implement config update logic
        return jsonify({'status': 'success'})


if __name__ == '__main__':
    # Initialize system
    if initialize_system():
        # Run Flask app
        app.run(
            host=config['api']['host'],
            port=config['api']['port'],
            debug=config['api']['debug']
        )
    else:
        logger.error("Failed to start application")
