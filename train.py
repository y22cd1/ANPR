"""
Training script for YOLO-based ANPR model.
Fine-tunes model across diverse lighting and traffic conditions.
"""

import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from datetime import datetime

from src.utils import setup_logger
from src.preprocessing import DataAugmentation, create_yolo_dataset_yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
log_file = Path(config['paths']['logs_dir']) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logger(
    name="ANPR_TRAINING",
    log_file=str(log_file),
    level=config['logging']['level']
)


def prepare_dataset():
    """Prepare dataset for training."""
    logger.info("Preparing dataset...")

    data_dir = Path(config['paths']['data_dir'])

    # Create dataset YAML
    dataset_yaml = data_dir / 'dataset.yaml'

    if not dataset_yaml.exists():
        create_yolo_dataset_yaml(
            data_dir=str(data_dir),
            output_path=str(dataset_yaml),
            class_names=['vehicle', 'license_plate']
        )

    logger.info(f"Dataset YAML: {dataset_yaml}")
    return dataset_yaml


def train_model():
    """Train YOLO model for ANPR."""
    logger.info("Starting training...")

    # Prepare dataset
    dataset_yaml = prepare_dataset()

    # Initialize model
    model_name = config['model']['name']
    logger.info(f"Initializing {model_name} model")

    model = YOLO(f'{model_name}.pt')

    # Training parameters
    train_config = config['training']

    # Train the model
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {train_config['epochs']}")
    logger.info(f"  Batch size: {train_config['batch_size']}")
    logger.info(f"  Learning rate: {train_config['learning_rate']}")
    logger.info(f"  Image size: {config['model']['img_size']}")

    results = model.train(
        data=str(dataset_yaml),
        epochs=train_config['epochs'],
        batch=train_config['batch_size'],
        imgsz=config['model']['img_size'],
        device=config['model']['device'],
        lr0=train_config['learning_rate'],
        patience=train_config['patience'],
        save=True,
        project=config['paths']['models_dir'],
        name='anpr_model',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        # Data augmentation parameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )

    logger.info("Training complete!")

    # Save best model
    best_model_path = Path(config['paths']['models_dir']) / 'anpr_model' / 'weights' / 'best.pt'

    if best_model_path.exists():
        output_path = Path(config['model']['weights_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(best_model_path, output_path)

        logger.info(f"Best model saved to: {output_path}")

    # Print results
    logger.info("\nTraining Results:")
    logger.info(f"  Results saved to: {config['paths']['models_dir']}/anpr_model")

    return results


def evaluate_model():
    """Evaluate trained model."""
    logger.info("Evaluating model...")

    weights_path = config['model']['weights_path']

    if not Path(weights_path).exists():
        logger.error(f"Model weights not found: {weights_path}")
        return

    model = YOLO(weights_path)

    # Prepare dataset
    dataset_yaml = prepare_dataset()

    # Validate
    results = model.val(
        data=str(dataset_yaml),
        batch=config['training']['batch_size'],
        imgsz=config['model']['img_size'],
        device=config['model']['device'],
        split='test',
        save_json=True,
        save_hybrid=False,
        conf=config['model']['confidence_threshold'],
        iou=config['model']['iou_threshold'],
        max_det=300,
        half=False,
        plots=True
    )

    logger.info("\nEvaluation Results:")
    logger.info(f"  mAP50: {results.box.map50:.4f}")
    logger.info(f"  mAP50-95: {results.box.map:.4f}")
    logger.info(f"  Precision: {results.box.mp:.4f}")
    logger.info(f"  Recall: {results.box.mr:.4f}")

    return results


def export_model():
    """Export model to different formats."""
    logger.info("Exporting model...")

    weights_path = config['model']['weights_path']

    if not Path(weights_path).exists():
        logger.error(f"Model weights not found: {weights_path}")
        return

    model = YOLO(weights_path)

    # Export to ONNX
    onnx_path = model.export(format='onnx', dynamic=True)
    logger.info(f"ONNX model exported: {onnx_path}")

    # Export to TorchScript
    torchscript_path = model.export(format='torchscript')
    logger.info(f"TorchScript model exported: {torchscript_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train ANPR model')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'export', 'all'],
                        help='Operation mode')

    args = parser.parse_args()

    try:
        if args.mode == 'train':
            train_model()

        elif args.mode == 'eval':
            evaluate_model()

        elif args.mode == 'export':
            export_model()

        elif args.mode == 'all':
            train_model()
            evaluate_model()
            export_model()

    except Exception as e:
        logger.error(f"Error during {args.mode}: {e}", exc_info=True)
        raise

    logger.info("Done!")
