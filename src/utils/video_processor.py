import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Real-time video processing system for ANPR with low-latency inference.
    """

    def __init__(self, config: Dict, detector, ocr_engine):
        """
        Initialize video processor.

        Args:
            config: Configuration dictionary
            detector: LicensePlateDetector instance
            ocr_engine: OCREngine instance
        """
        self.config = config
        self.video_config = config['video']
        self.detector = detector
        self.ocr_engine = ocr_engine

        self.input_source = self.video_config['input_source']
        self.output_path = self.video_config['output_path']
        self.target_fps = self.video_config['fps']
        self.resolution = tuple(self.video_config['resolution'])

        self.cap = None
        self.writer = None
        self.running = False

        # Performance metrics
        self.fps_buffer = deque(maxlen=30)
        self.frame_times = deque(maxlen=100)

        # Detection history
        self.detection_history = []

    def initialize(self) -> bool:
        """
        Initialize video capture and writer.

        Returns:
            True if successful
        """
        try:
            # Initialize capture
            if isinstance(self.input_source, int):
                self.cap = cv2.VideoCapture(self.input_source)
            else:
                self.cap = cv2.VideoCapture(str(self.input_source))

            if not self.cap.isOpened():
                logger.error("Failed to open video source")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Get actual resolution
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video source initialized: {width}x{height}")

            # Initialize writer if output path specified
            if self.output_path:
                output_path = Path(self.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.target_fps,
                    (width, height)
                )

                logger.info(f"Video writer initialized: {self.output_path}")

            self.running = True
            return True

        except Exception as e:
            logger.error(f"Error initializing video processor: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame with detection and OCR.

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated_frame, detections, plate_texts)
        """
        start_time = time.time()

        # Detect vehicles and license plates
        detections = self.detector.detect(frame)

        # Extract and process license plates
        plate_texts = []

        for det in detections:
            # Check if it's a license plate
            if det['class_id'] == 1 or 'plate' in det['class_name'].lower():
                x1, y1, x2, y2 = det['bbox']

                # Extract plate region
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size > 0:
                    # Preprocess plate
                    processed_plate = self.detector.preprocess_plate(plate_img)

                    # Run OCR
                    ocr_result = self.ocr_engine.read_text(processed_plate)

                    if ocr_result['text']:
                        plate_texts.append({
                            'text': ocr_result['text'],
                            'confidence': ocr_result['confidence'],
                            'bbox': det['bbox'],
                            'timestamp': datetime.now().isoformat()
                        })

                        # Add text to detection
                        det['plate_text'] = ocr_result['text']
                        det['ocr_confidence'] = ocr_result['confidence']

        # Draw detections
        annotated_frame = self._draw_detections_with_text(frame, detections)

        # Calculate processing time
        process_time = time.time() - start_time
        self.frame_times.append(process_time)

        # Calculate FPS
        if process_time > 0:
            fps = 1.0 / process_time
            self.fps_buffer.append(fps)

        return annotated_frame, detections, plate_texts

    def _draw_detections_with_text(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw detections with OCR text on frame.

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Choose color
            if 'plate' in class_name.lower():
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"{class_name}: {confidence:.2f}"

            if 'plate_text' in det:
                label += f" | {det['plate_text']}"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated

    def add_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add FPS and performance metrics overlay.

        Args:
            frame: Input frame

        Returns:
            Frame with overlay
        """
        if self.fps_buffer:
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

            # Add semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Add text
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if self.frame_times:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                cv2.putText(
                    frame,
                    f"Latency: {avg_time*1000:.1f}ms",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        return frame

    def run(self, display: bool = True, callback: Optional[Callable] = None):
        """
        Run real-time video processing loop.

        Args:
            display: Whether to display video
            callback: Optional callback function for each frame
        """
        if not self.initialize():
            logger.error("Failed to initialize video processor")
            return

        logger.info("Starting video processing...")

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    break

                # Process frame
                annotated_frame, detections, plate_texts = self.process_frame(frame)

                # Add performance overlay
                annotated_frame = self.add_performance_overlay(annotated_frame)

                # Save detections
                if plate_texts:
                    self.detection_history.extend(plate_texts)
                    logger.info(f"Detected plates: {[p['text'] for p in plate_texts]}")

                # Write to file
                if self.writer:
                    self.writer.write(annotated_frame)

                # Display
                if display:
                    cv2.imshow('ANPR System', annotated_frame)

                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested stop")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        self.save_frame(annotated_frame)

                # Callback
                if callback:
                    callback(annotated_frame, detections, plate_texts)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def save_frame(self, frame: np.ndarray, prefix: str = "detection"):
        """
        Save a frame to disk.

        Args:
            frame: Frame to save
            prefix: Filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = Path(self.config['paths']['detections_dir']) / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), frame)

        logger.info(f"Saved frame: {filepath}")

    def cleanup(self):
        """Release resources."""
        logger.info("Cleaning up resources...")

        if self.cap:
            self.cap.release()

        if self.writer:
            self.writer.release()

        cv2.destroyAllWindows()

        self.running = False

        logger.info("Cleanup complete")

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_detections': len(self.detection_history),
            'unique_plates': len(set(d['text'] for d in self.detection_history)),
        }

        if self.fps_buffer:
            stats['avg_fps'] = sum(self.fps_buffer) / len(self.fps_buffer)

        if self.frame_times:
            stats['avg_latency_ms'] = (sum(self.frame_times) / len(self.frame_times)) * 1000

        return stats
