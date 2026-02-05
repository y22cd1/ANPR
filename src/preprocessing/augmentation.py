import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Data augmentation pipeline for license plate detection.
    Handles diverse lighting and traffic conditions.
    """

    def __init__(self, config: Dict):
        """
        Initialize augmentation pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.aug_config = config['augmentation']

        # Training augmentation pipeline
        self.train_transform = A.Compose([
            # Brightness and contrast adjustments for different lighting
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_config['brightness_range'],
                contrast_limit=self.aug_config['contrast_range'],
                p=0.8
            ),

            # Simulate different weather conditions
            A.OneOf([
                A.MotionBlur(blur_limit=self.aug_config['blur_limit'], p=1.0),
                A.GaussianBlur(blur_limit=self.aug_config['blur_limit'], p=1.0),
                A.MedianBlur(blur_limit=self.aug_config['blur_limit'], p=1.0),
            ], p=0.5),

            # Add noise for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=self.aug_config['noise_prob']),

            # Geometric transformations
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=self.aug_config['scale_limit'],
                rotate_limit=self.aug_config['rotation_limit'],
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),

            # Perspective transformation for different viewing angles
            A.Perspective(scale=(0.05, 0.1), p=0.5),

            # Color adjustments
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            ),

            # Shadow simulation
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),

            # Sun flare for daytime conditions
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=2,
                p=0.2
            ),

            # Rain simulation
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=10,
                drop_width=1,
                p=0.2
            ),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

        # Validation transform (minimal augmentation)
        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

        logger.info("Data augmentation pipeline initialized")

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: list,
        class_labels: list,
        mode: str = 'train'
    ) -> Tuple[np.ndarray, list, list]:
        """
        Apply augmentation to image and bounding boxes.

        Args:
            image: Input image (BGR format)
            bboxes: List of bounding boxes in [x_min, y_min, x_max, y_max] format
            class_labels: List of class labels for each bbox
            mode: 'train' or 'val'

        Returns:
            Tuple of (augmented_image, augmented_bboxes, class_labels)
        """
        try:
            # Convert BGR to RGB for albumentations
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Choose transform based on mode
            transform = self.train_transform if mode == 'train' else self.val_transform

            # Apply augmentation
            transformed = transform(
                image=image_rgb,
                bboxes=bboxes,
                class_labels=class_labels
            )

            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']

            # Convert back to BGR for OpenCV
            if isinstance(aug_image, np.ndarray):
                aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

            return aug_image, aug_bboxes, aug_labels

        except Exception as e:
            logger.error(f"Augmentation error: {e}")
            return image, bboxes, class_labels

    def preprocess_for_yolo(
        self,
        image: np.ndarray,
        target_size: int = 640
    ) -> np.ndarray:
        """
        Preprocess image for YOLO input.

        Args:
            image: Input image
            target_size: Target image size

        Returns:
            Preprocessed image
        """
        # Resize with aspect ratio preservation
        h, w = image.shape[:2]
        scale = target_size / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        # Center the image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        num_augmentations: int = 5
    ):
        """
        Augment entire dataset.

        Args:
            input_dir: Directory with original images
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmentations per image
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))

        logger.info(f"Augmenting {len(image_files)} images...")

        for img_path in image_files:
            image = cv2.imread(str(img_path))

            if image is None:
                logger.warning(f"Could not read {img_path}")
                continue

            # TODO: Load corresponding bounding boxes from annotation file

            for i in range(num_augmentations):
                # Apply augmentation
                aug_image, _, _ = self.augment_image(
                    image,
                    [],  # Add actual bboxes
                    [],  # Add actual labels
                    mode='train'
                )

                # Save augmented image
                output_path = output_dir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
                cv2.imwrite(str(output_path), aug_image)

        logger.info(f"Augmentation complete. Saved to {output_dir}")

    def visualize_augmentation(
        self,
        image: np.ndarray,
        bboxes: list,
        class_labels: list,
        num_samples: int = 4
    ):
        """
        Visualize augmentation effects.

        Args:
            image: Input image
            bboxes: Bounding boxes
            class_labels: Class labels
            num_samples: Number of augmented samples to generate
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))

        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Augmented samples
        for i in range(num_samples):
            aug_image, aug_bboxes, _ = self.augment_image(
                image,
                bboxes,
                class_labels,
                mode='train'
            )

            axes[i + 1].imshow(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f'Augmented {i + 1}')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.savefig('augmentation_samples.png')
        logger.info("Saved augmentation visualization to augmentation_samples.png")
