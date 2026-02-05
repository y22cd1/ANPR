import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    YOLO-based license plate and vehicle detector with real-time inference capabilities.
    """

    def __init__(self, config: Dict):
        """
        Initialize the detector with configuration.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_config = config['model']
        self.confidence_threshold = self.model_config['confidence_threshold']
        self.iou_threshold = self.model_config['iou_threshold']
        self.device = self.model_config['device']
        self.img_size = self.model_config['img_size']

        # Initialize model
        self.model = None
        self.load_model()

        logger.info(f"LicensePlateDetector initialized with device: {self.device}")

    def load_model(self):
        """Load YOLO model from weights or pretrained model."""
        try:
            weights_path = self.model_config['weights_path']
            model_name = self.model_config['name']

            # Try loading custom weights first
            try:
                self.model = YOLO(weights_path)
                logger.info(f"Loaded custom model from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load custom weights: {e}")
                logger.info(f"Loading pretrained {model_name} model")
                self.model = YOLO(f'{model_name}.pt')

            # Set device
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = 'cpu'

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles and license plates in a frame.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries containing bbox, confidence, and class
        """
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )

            detections = []

            # Parse results
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.model.names[class_id] if hasattr(self.model, 'names') else f'class_{class_id}'
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def detect_license_plates(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect and extract license plate regions from frame.

        Args:
            frame: Input image

        Returns:
            Tuple of (license_plate_images, detection_info)
        """
        detections = self.detect(frame)

        license_plates = []
        plate_detections = []

        for det in detections:
            # Filter for license plate class (assuming class_id 1 or checking class_name)
            if det['class_id'] == 1 or 'plate' in det['class_name'].lower() or 'license' in det['class_name'].lower():
                x1, y1, x2, y2 = det['bbox']

                # Extract plate region
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size > 0:
                    license_plates.append(plate_img)
                    plate_detections.append(det)

        return license_plates, plate_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Choose color based on class
            if 'plate' in class_name.lower() or 'license' in class_name.lower():
                color = (0, 255, 0)  # Green for license plates
            else:
                color = (255, 0, 0)  # Blue for vehicles

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Background for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return annotated_frame

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR accuracy.

        Args:
            plate_img: License plate image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Resize for better OCR
        height, width = thresh.shape
        if height < 100:
            scale = 100 / height
            new_width = int(width * scale)
            thresh = cv2.resize(thresh, (new_width, 100), interpolation=cv2.INTER_CUBIC)

        return thresh
