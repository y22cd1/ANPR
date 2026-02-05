"""
Quick demo script to test ANPR system with sample image.
"""

import cv2
import yaml
import numpy as np
from pathlib import Path

from src.detection import LicensePlateDetector
from src.ocr import OCREngine
from src.utils import setup_logger

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logger = setup_logger(name="ANPR_DEMO", level="INFO")


def create_test_image():
    """Create a simple test image with text."""
    # Create blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Add text
    text = "ANPR System Demo"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (150, 240), font, 1.5, (0, 0, 0), 3)

    # Add instruction
    instruction = "Add your own image to test detection"
    cv2.putText(img, instruction, (100, 300), font, 0.7, (100, 100, 100), 2)

    return img


def main():
    logger.info("=" * 60)
    logger.info("ANPR System Demo")
    logger.info("=" * 60)

    try:
        # Initialize detector
        logger.info("\n1. Initializing YOLO detector...")
        detector = LicensePlateDetector(config)
        logger.info("   ✓ Detector initialized successfully")

        # Initialize OCR
        logger.info("\n2. Initializing OCR engine...")
        ocr_engine = OCREngine(config)
        logger.info("   ✓ OCR engine initialized successfully")

        # Create or load test image
        logger.info("\n3. Loading test image...")
        test_img_path = Path("test_image.jpg")

        if test_img_path.exists():
            image = cv2.imread(str(test_img_path))
            logger.info(f"   ✓ Loaded image from {test_img_path}")
        else:
            image = create_test_image()
            cv2.imwrite(str(test_img_path), image)
            logger.info(f"   ✓ Created test image: {test_img_path}")
            logger.info("   Note: Replace this with an actual vehicle image for real detection")

        # Run detection
        logger.info("\n4. Running detection...")
        detections = detector.detect(image)
        logger.info(f"   ✓ Found {len(detections)} detections")

        # Process detections
        if detections:
            logger.info("\n5. Detection Results:")
            for i, det in enumerate(detections, 1):
                logger.info(f"\n   Detection {i}:")
                logger.info(f"   - Class: {det['class_name']}")
                logger.info(f"   - Confidence: {det['confidence']:.2%}")
                logger.info(f"   - Bounding Box: {det['bbox']}")

                # If it's a license plate, run OCR
                if 'plate' in det['class_name'].lower():
                    x1, y1, x2, y2 = det['bbox']
                    plate_img = image[y1:y2, x1:x2]

                    if plate_img.size > 0:
                        processed = detector.preprocess_plate(plate_img)
                        ocr_result = ocr_engine.read_text(processed)

                        if ocr_result['text']:
                            logger.info(f"   - Plate Text: {ocr_result['text']}")
                            logger.info(f"   - OCR Confidence: {ocr_result['confidence']:.2%}")

            # Draw detections
            annotated = detector.draw_detections(image, detections)

            # Save result
            output_path = Path("demo_output.jpg")
            cv2.imwrite(str(output_path), annotated)
            logger.info(f"\n   ✓ Saved result to: {output_path}")

        else:
            logger.info("\n   No detections found.")
            logger.info("   This is normal for the demo image.")
            logger.info("   Try with an actual vehicle image!")

        # Show capabilities
        logger.info("\n" + "=" * 60)
        logger.info("System Capabilities:")
        logger.info("=" * 60)
        logger.info("✓ Real-time video processing")
        logger.info("✓ Vehicle detection")
        logger.info("✓ License plate detection")
        logger.info("✓ OCR text extraction")
        logger.info("✓ Data augmentation")
        logger.info("✓ Custom model training")
        logger.info("✓ Web interface monitoring")

        logger.info("\n" + "=" * 60)
        logger.info("Next Steps:")
        logger.info("=" * 60)
        logger.info("\n1. Test with webcam:")
        logger.info("   python run.py --mode realtime")
        logger.info("\n2. Process video file:")
        logger.info("   python run.py --mode video --input your_video.mp4")
        logger.info("\n3. Web interface:")
        logger.info("   python run.py --mode web")
        logger.info("\n4. Train custom model:")
        logger.info("   python train.py --mode train")
        logger.info("\nSee README.md for detailed documentation.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        logger.info("\nTroubleshooting:")
        logger.info("1. Check if all dependencies are installed: pip install -r requirements.txt")
        logger.info("2. Verify Python version (3.8+): python --version")
        logger.info("3. See SETUP_GUIDE.md for detailed installation instructions")


if __name__ == '__main__':
    main()
