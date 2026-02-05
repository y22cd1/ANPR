# Models Directory

This directory stores trained model weights.

## Pre-trained Models

On first run, the system will automatically download YOLOv8 pre-trained weights.

## Custom Models

After training your own model, the best weights will be saved here as `best.pt`.

## Model Files

- `best.pt` - Best model from training (used by default)
- `last.pt` - Last checkpoint from training
- `*.onnx` - Exported ONNX models
- `*.torchscript` - Exported TorchScript models

## Usage

Update `config.yaml` to specify which model to use:

```yaml
model:
  weights_path: "models/best.pt"
```
