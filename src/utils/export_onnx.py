"""
Model Export Utilities
Export YOLOv8 models to ONNX and TensorRT formats.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def export_to_onnx(
    model_path: str,
    output_path: str = None,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 12
):
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model (.pt)
        output_path: Optional output path (default: same name with .onnx)
        dynamic: Enable dynamic input shapes
        simplify: Simplify ONNX model
        opset: ONNX opset version
    
    Returns:
        Path to exported ONNX model
    """
    print("=" * 70)
    print("ONNX Export")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Export
    print(f"\nExporting to ONNX...")
    print(f"  Dynamic shapes: {dynamic}")
    print(f"  Simplify: {simplify}")
    print(f"  Opset version: {opset}")
    
    exported_path = model.export(
        format='onnx',
        dynamic=dynamic,
        simplify=simplify,
        opset=opset
    )
    
    # Rename if output path specified
    if output_path:
        exported_path = Path(exported_path)
        output_path = Path(output_path)
        exported_path.rename(output_path)
        exported_path = output_path
    
    print(f"\n✅ ONNX model exported: {exported_path}")
    print(f"   File size: {Path(exported_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(exported_path)


def export_to_tensorrt(
    model_path: str,
    output_path: str = None,
    half: bool = True,
    workspace: int = 4
):
    """
    Export YOLOv8 model to TensorRT format (NVIDIA GPUs only).
    
    Args:
        model_path: Path to PyTorch model (.pt)
        output_path: Optional output path
        half: Use FP16 precision
        workspace: Maximum workspace size (GB)
    
    Returns:
        Path to exported TensorRT engine
    """
    print("=" * 70)
    print("TensorRT Export")
    print("=" * 70)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ Error: CUDA not available. TensorRT requires an NVIDIA GPU.")
        return None
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Export
    print(f"\nExporting to TensorRT...")
    print(f"  Precision: {'FP16' if half else 'FP32'}")
    print(f"  Workspace: {workspace} GB")
    print("\nNote: This may take several minutes...")
    
    try:
        exported_path = model.export(
            format='engine',
            half=half,
            workspace=workspace
        )
        
        # Rename if output path specified
        if output_path:
            exported_path = Path(exported_path)
            output_path = Path(output_path)
            exported_path.rename(output_path)
            exported_path = output_path
        
        print(f"\n✅ TensorRT engine exported: {exported_path}")
        print(f"   File size: {Path(exported_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(exported_path)
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\nMake sure TensorRT is properly installed:")
        print("  pip install nvidia-tensorrt")
        return None


def benchmark_model(model_path: str, imgsz: int = 640):
    """
    Benchmark model inference speed.
    
    Args:
        model_path: Path to model
        imgsz: Input image size
    """
    print("=" * 70)
    print("Model Benchmark")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    print(f"\nModel: {model_path}")
    print(f"Input size: {imgsz}x{imgsz}")
    
    # Run benchmark
    metrics = model.val(data='coco128.yaml', imgsz=imgsz, batch=1)
    
    print("\nPerformance:")
    print(f"  Inference: {metrics.speed['inference']:.2f} ms")
    print(f"  Preprocess: {metrics.speed['preprocess']:.2f} ms")
    print(f"  Postprocess: {metrics.speed['postprocess']:.2f} ms")
    print(f"  Total: {sum(metrics.speed.values()):.2f} ms")
    print(f"  FPS: {1000 / sum(metrics.speed.values()):.1f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Export YOLOv8 models')
    parser.add_argument('--model', type=str, required=True, help='Model path (.pt)')
    parser.add_argument('--format', type=str, choices=['onnx', 'tensorrt', 'both'], default='onnx')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic shapes (ONNX)')
    parser.add_argument('--half', action='store_true', help='FP16 precision (TensorRT)')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark exported model')
    
    args = parser.parse_args()
    
    # Export
    if args.format in ['onnx', 'both']:
        onnx_output = args.output if args.output and args.format == 'onnx' else None
        onnx_path = export_to_onnx(
            model_path=args.model,
            output_path=onnx_output,
            dynamic=args.dynamic
        )
        
        if args.benchmark and onnx_path:
            benchmark_model(onnx_path)
    
    if args.format in ['tensorrt', 'both']:
        trt_output = args.output if args.output and args.format == 'tensorrt' else None
        trt_path = export_to_tensorrt(
            model_path=args.model,
            output_path=trt_output,
            half=args.half
        )
        
        if args.benchmark and trt_path:
            benchmark_model(trt_path)
    
    print("\n✅ Export complete!")


if __name__ == '__main__':
    main()
