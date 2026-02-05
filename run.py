"""
Main script to run ANPR system in different modes.
"""

import yaml
import argparse
import logging
from pathlib import Path

from src.detection import LicensePlateDetector
from src.ocr import OCREngine
from src.utils import VideoProcessor, setup_logger

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logger = setup_logger(
    name="ANPR",
    log_file=config['logging']['log_file'],
    level=config['logging']['level']
)


def run_realtime():
    """Run real-time detection from webcam."""
    logger.info("Starting real-time detection mode")

    # Initialize detector
    detector = LicensePlateDetector(config)

    # Initialize OCR
    ocr_engine = OCREngine(config)

    # Initialize video processor
    processor = VideoProcessor(config, detector, ocr_engine)

    # Run processing
    processor.run(display=True)

    # Print statistics
    stats = processor.get_statistics()
    logger.info("\nSession Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def run_video(video_path: str):
    """Run detection on video file."""
    logger.info(f"Processing video: {video_path}")

    # Update config with video path
    config['video']['input_source'] = video_path

    # Initialize components
    detector = LicensePlateDetector(config)
    ocr_engine = OCREngine(config)
    processor = VideoProcessor(config, detector, ocr_engine)

    # Run processing
    processor.run(display=True)

    # Print statistics
    stats = processor.get_statistics()
    logger.info("\nProcessing Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def run_image(image_path: str):
    """Run detection on single image."""
    import cv2

    logger.info(f"Processing image: {image_path}")

    # Initialize detector
    detector = LicensePlateDetector(config)
    ocr_engine = OCREngine(config)

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    # Detect
    detections = detector.detect(image)

    logger.info(f"Found {len(detections)} detections")

    # Process license plates
    for i, det in enumerate(detections):
        if det['class_id'] == 1 or 'plate' in det['class_name'].lower():
            x1, y1, x2, y2 = det['bbox']
            plate_img = image[y1:y2, x1:x2]

            if plate_img.size > 0:
                # Preprocess and run OCR
                processed = detector.preprocess_plate(plate_img)
                ocr_result = ocr_engine.read_text(processed)

                logger.info(f"Plate {i + 1}: {ocr_result['text']} "
                            f"(confidence: {ocr_result['confidence']:.2f})")

                det['plate_text'] = ocr_result['text']
                det['ocr_confidence'] = ocr_result['confidence']

    # Draw detections
    annotated = detector.draw_detections(image, detections)

    # Add OCR results
    for det in detections:
        if 'plate_text' in det:
            x1, y1, _, _ = det['bbox']
            cv2.putText(
                annotated,
                det['plate_text'],
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Save result
    output_path = Path(config['paths']['outputs_dir']) / f"result_{Path(image_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Result saved to: {output_path}")

    # Display
    cv2.imshow('Detection Result', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_web():
    """Run web application."""
    logger.info("Starting web application")

    from app import app as flask_app

    flask_app.run(
        host=config['api']['host'],
        port=config['api']['port'],
        debug=config['api']['debug']
    )


def main():
    parser = argparse.ArgumentParser(
        description='Automatic Number Plate Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Real-time detection:
    python run.py --mode realtime

  Process video file:
    python run.py --mode video --input path/to/video.mp4

  Process image:
    python run.py --mode image --input path/to/image.jpg

  Web interface:
    python run.py --mode web
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['realtime', 'video', 'image', 'web'],
        help='Operation mode'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (for video or image mode)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ['video', 'image'] and not args.input:
        parser.error(f"--input is required for {args.mode} mode")

    try:
        if args.mode == 'realtime':
            run_realtime()

        elif args.mode == 'video':
            run_video(args.input)

        elif args.mode == 'image':
            run_image(args.input)

        elif args.mode == 'web':
            run_web()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
