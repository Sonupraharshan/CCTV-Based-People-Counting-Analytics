"""
YOLO to MOT Format Converter
Converts YOLO detection results to MOT Challenge format for evaluation.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


class YOLOToMOTConverter:
    """Convert YOLO detections to MOT format."""
    
    def __init__(self, yolo_dir: Path, output_dir: Path):
        """
        Initialize converter.
        
        Args:
            yolo_dir: Directory with YOLO format labels
            output_dir: Directory to save MOT format files
        """
        self.yolo_dir = Path(yolo_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"YOLO to MOT Converter")
        print(f"Input: {self.yolo_dir}")
        print(f"Output: {self.output_dir}")
    
    def convert_detections(
        self,
        label_file: Path,
        frame_id: int,
        img_width: int,
        img_height: int
    ) -> list:
        """
        Convert YOLO detections for a single frame to MOT format.
        
        Args:
            label_file: Path to YOLO label file
            frame_id: Frame number
            img_width: Image width
            img_height: Image height
        
        Returns:
            List of MOT format rows
        """
        if not label_file.exists():
            return []
        
        mot_rows = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO format: class x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to top-left corner
                x = x_center - width / 2
                y = y_center - height / 2
                
                # MOT format: frame, id, x, y, w, h, conf, class, visibility
                # Use -1 for track_id (detections only, no tracking)
                # Use 1.0 for confidence (or extract if available)
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                
                mot_row = [
                    frame_id,      # Frame number
                    -1,            # Track ID (-1 for detections only)
                    x,             # Top-left x
                    y,             # Top-left y
                    width,         # Width
                    height,        # Height
                    conf,          # Confidence
                    class_id + 1,  # Class (MOT uses 1-indexed, 1=pedestrian)
                    1.0            # Visibility
                ]
                
                mot_rows.append(mot_row)
        
        return mot_rows
    
    def convert_sequence(
        self,
        sequence_name: str,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        """
        Convert a complete sequence.
        
        Args:
            sequence_name: Name of the sequence
            img_width: Image width
            img_height: Image height
        """
        print(f"\nConverting sequence: {sequence_name}")
        
        # Input label directory
        label_dir = self.yolo_dir / sequence_name
        
        if not label_dir.exists():
            print(f"  Error: Label directory not found: {label_dir}")
            return
        
        # Get all label files
        label_files = sorted(label_dir.glob("*.txt"))
        
        if len(label_files) == 0:
            print(f"  Warning: No label files found in {label_dir}")
            return
        
        # Output file
        output_file = self.output_dir / f"{sequence_name}.txt"
        
        all_detections = []
        
        # Process each frame
        for frame_id, label_file in enumerate(tqdm(label_files, desc=f"  {sequence_name}"), start=1):
            detections = self.convert_detections(
                label_file, frame_id, img_width, img_height
            )
            all_detections.extend(detections)
        
        # Write to file
        if all_detections:
            np.savetxt(
                output_file,
                all_detections,
                fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%.2f',
                delimiter=','
            )
            print(f"  ✓ Saved {len(all_detections)} detections to {output_file}")
        else:
            print(f"  Warning: No detections found")
    
    def convert_all(self, sequences: list = None):
        """
        Convert all sequences.
        
        Args:
            sequences: List of sequence names (if None, auto-detect)
        """
        if sequences is None:
            # Auto-detect sequences
            sequences = [d.name for d in self.yolo_dir.iterdir() if d.is_dir()]
        
        print(f"\nFound {len(sequences)} sequences")
        
        for sequence in sequences:
            self.convert_sequence(sequence)
        
        print("\n✅ Conversion complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Convert YOLO to MOT format')
    parser.add_argument('--input', type=str, required=True, help='YOLO labels directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--sequences', type=str, nargs='+', help='Specific sequences')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')
    
    args = parser.parse_args()
    
    converter = YOLOToMOTConverter(
        yolo_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    
    converter.convert_all(sequences=args.sequences)
    
    print(f"\n✅ MOT format files saved in: {args.output}")


if __name__ == '__main__':
    main()
