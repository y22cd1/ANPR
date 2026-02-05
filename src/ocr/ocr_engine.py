import cv2
import numpy as np
import re
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("Pytesseract not available")


class OCREngine:
    """
    OCR Engine for extracting text from license plate images.
    Supports both EasyOCR and Pytesseract backends.
    """

    def __init__(self, config: Dict):
        """
        Initialize OCR engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ocr_config = config['ocr']
        self.engine_type = self.ocr_config['engine']
        self.languages = self.ocr_config['languages']
        self.allowed_chars = self.ocr_config['allowed_chars']
        self.min_confidence = self.ocr_config['min_confidence']

        self.reader = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the OCR engine based on configuration."""
        if self.engine_type == 'easyocr':
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR is not installed. Install with: pip install easyocr")

            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info("EasyOCR initialized successfully")

        elif self.engine_type == 'pytesseract':
            if not PYTESSERACT_AVAILABLE:
                raise ImportError("Pytesseract is not installed. Install with: pip install pytesseract")

            logger.info("Using Pytesseract engine")

        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_type}")

    def read_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from license plate image.

        Args:
            image: Preprocessed license plate image

        Returns:
            Dictionary with extracted text and confidence
        """
        if self.engine_type == 'easyocr':
            return self._read_easyocr(image)
        elif self.engine_type == 'pytesseract':
            return self._read_pytesseract(image)
        else:
            return {'text': '', 'confidence': 0.0}

    def _read_easyocr(self, image: np.ndarray) -> Dict:
        """
        Read text using EasyOCR.

        Args:
            image: Input image

        Returns:
            Dictionary with text and confidence
        """
        try:
            results = self.reader.readtext(image)

            if not results:
                return {'text': '', 'confidence': 0.0}

            # Combine all detected text
            all_text = []
            confidences = []

            for bbox, text, confidence in results:
                if confidence >= self.min_confidence:
                    cleaned_text = self._clean_text(text)
                    if cleaned_text:
                        all_text.append(cleaned_text)
                        confidences.append(confidence)

            if all_text:
                combined_text = ''.join(all_text)
                avg_confidence = sum(confidences) / len(confidences)

                return {
                    'text': combined_text,
                    'confidence': avg_confidence
                }

            return {'text': '', 'confidence': 0.0}

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return {'text': '', 'confidence': 0.0}

    def _read_pytesseract(self, image: np.ndarray) -> Dict:
        """
        Read text using Pytesseract.

        Args:
            image: Input image

        Returns:
            Dictionary with text and confidence
        """
        try:
            # Configure Tesseract for license plate recognition
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=' + self.allowed_chars

            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)

            # Get confidence
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            cleaned_text = self._clean_text(text)

            if avg_confidence >= self.min_confidence and cleaned_text:
                return {
                    'text': cleaned_text,
                    'confidence': avg_confidence
                }

            return {'text': '', 'confidence': 0.0}

        except Exception as e:
            logger.error(f"Pytesseract error: {e}")
            return {'text': '', 'confidence': 0.0}

    def _clean_text(self, text: str) -> str:
        """
        Clean and format extracted text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        # Remove whitespace
        text = text.strip().replace(' ', '').replace('\n', '')

        # Keep only allowed characters
        text = ''.join(char for char in text if char in self.allowed_chars)

        # Convert to uppercase
        text = text.upper()

        return text

    def validate_plate_format(self, text: str, patterns: Optional[List[str]] = None) -> bool:
        """
        Validate if extracted text matches expected license plate patterns.

        Args:
            text: Extracted text
            patterns: List of regex patterns to match

        Returns:
            True if text matches any pattern
        """
        if not patterns:
            # Default patterns (US format examples)
            patterns = [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[A-Z]{2}[0-9]{5}$',   # AB12345
                r'^[0-9]{3}[A-Z]{3}$',   # 123ABC
                r'^[A-Z]{1}[0-9]{6}$',   # A123456
            ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True

        return False

    def post_process(self, text: str) -> str:
        """
        Apply post-processing rules to improve accuracy.

        Args:
            text: Extracted text

        Returns:
            Post-processed text
        """
        # Common OCR mistakes
        replacements = {
            'O': '0',  # Context-dependent
            'I': '1',
            'S': '5',
            'B': '8',
            'G': '6',
        }

        # Apply replacements based on position (numbers usually at the end)
        processed = text

        # More sophisticated logic can be added here based on plate format

        return processed
