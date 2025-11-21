"""
YOLOv8 Training Script
Fine-tune YOLOv8 on MOT17 + PETS2009 for improved people detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch


def train_yolov8(
    data_config: str = "train/config.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "models/finetuned",
    name: str = "yolov8_mot17",
    resume: bool = False
):
    """
    Train YOLOv8 model.
    
    Args:
        data_config: Path to data configuration YAML
        model: Base model or checkpoint to resume from
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
        device: Device for training
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
    """
    print("=" * 70)
    print("YOLOv8 Training Script")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model}")
    yolo_model = YOLO(model)
    
    # Check device
    if device == "cpu":
        print("Training on CPU (this will be slow!)")
    else:
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Data config: {data_config}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}")
    print()
    
    results = yolo_model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        resume=resume,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=True,
        workers=8,
        # Save settings
        save=True,
        save_period=5,
        # Validation
        val=True,
        plots=True,
        # Augmentation (can override config)
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        # degrees=0.0,
        # translate=0.1,
        # scale=0.5,
        # fliplr=0.5,
        # mosaic=1.0,
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Print results
    print(f"\nBest model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {results.save_dir / 'weights' / 'last.pt'}")
    
    # Validation metrics
    print(f"\nFinal metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    return results


def validate_model(model_path: str, data_config: str):
    """Run validation on trained model."""
    print(f"\nValidating model: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=data_config)
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model_path: str, format: str = "onnx"):
    """Export trained model to different formats."""
    print(f"\nExporting model to {format.upper()}...")
    
    model = YOLO(model_path)
    exported = model.export(format=format)
    
    print(f"✓ Model exported: {exported}")
    
    return exported


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for people detection')
    
    # Training arguments
    parser.add_argument('--data', type=str, default='train/config.yaml', help='Data config path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--project', type=str, default='models/finetuned', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov8_mot17', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # Other actions
    parser.add_argument('--validate', type=str, help='Validate a model checkpoint')
    parser.add_argument('--export', type=str, help='Export model to format (onnx, engine, etc.)')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_model(args.validate, args.data)
    elif args.export:
        model_path = args.model if Path(args.model).exists() else f"{args.project}/{args.name}/weights/best.pt"
        export_model(model_path, args.export)
    else:
        train_yolov8(
            data_config=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume
        )
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
