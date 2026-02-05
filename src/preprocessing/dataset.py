import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ANPRDataset(Dataset):
    """
    Dataset class for ANPR with YOLO format annotations.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        img_size: int = 640
    ):
        """
        Initialize dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            transform: Augmentation transform
            img_size: Target image size
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.img_size = img_size

        # Get all image files
        self.image_files = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.png'))
        )

        logger.info(f"Loaded {len(self.image_files)} images from {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Dictionary with image and labels
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))

        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            # Return empty sample
            return {
                'image': torch.zeros((3, self.img_size, self.img_size)),
                'labels': torch.zeros((0, 5)),
                'img_path': str(img_path)
            }

        h, w = image.shape[:2]

        # Load labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert from YOLO format (normalized) to pixel coordinates
                        x_min = int((x_center - width / 2) * w)
                        y_min = int((y_center - height / 2) * h)
                        x_max = int((x_center + width / 2) * w)
                        y_max = int((y_center + height / 2) * h)

                        bboxes.append([x_min, y_min, x_max, y_max])
                        class_labels.append(class_id)

        # Apply augmentation
        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception as e:
                logger.warning(f"Transform failed for {img_path}: {e}")

        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Prepare labels tensor
        if bboxes:
            labels = []
            for bbox, class_id in zip(bboxes, class_labels):
                x_min, y_min, x_max, y_max = bbox

                # Convert back to YOLO format (normalized)
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                labels.append([class_id, x_center, y_center, width, height])

            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'image': image,
            'labels': labels,
            'img_path': str(img_path)
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples

        Returns:
            Batched data
        """
        images = torch.stack([item['image'] for item in batch])

        # Labels need special handling due to variable number of objects
        labels = [item['labels'] for item in batch]

        img_paths = [item['img_path'] for item in batch]

        return {
            'images': images,
            'labels': labels,
            'img_paths': img_paths
        }


def create_yolo_dataset_yaml(
    data_dir: str,
    output_path: str,
    class_names: List[str]
):
    """
    Create YOLO dataset YAML file.

    Args:
        data_dir: Root data directory
        output_path: Output YAML path
        class_names: List of class names
    """
    data_dir = Path(data_dir)

    yaml_content = f"""
# ANPR Dataset Configuration

path: {data_dir.absolute()}  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images (optional)

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"Created dataset YAML: {output_path}")


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Split dataset into train/val/test sets.

    Args:
        images_dir: Directory with images
        labels_dir: Directory with labels
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    import shutil
    from sklearn.model_selection import train_test_split

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Get all image files
    image_files = sorted(
        list(images_dir.glob('*.jpg')) +
        list(images_dir.glob('*.png'))
    )

    # Split into train/temp
    train_files, temp_files = train_test_split(
        image_files,
        test_size=(1 - train_ratio),
        random_state=42
    )

    # Split temp into val/test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=(1 - val_size),
        random_state=42
    )

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        logger.info(f"Copying {len(files)} files to {split_name}")

        for img_path in files:
            # Copy image
            dst_img = output_dir / 'images' / split_name / img_path.name
            shutil.copy(img_path, dst_img)

            # Copy label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = output_dir / 'labels' / split_name / label_path.name
                shutil.copy(label_path, dst_label)

    logger.info(f"Dataset split complete: {output_dir}")
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
