"""
MOT to YOLO Format Converter
Converts MOT Challenge annotations to YOLO format for detector training.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil


class MOTToYOLOConverter:
    """Convert MOT format annotations to YOLO format."""
    
    def __init__(self, mot_dir: Path, output_dir: Path):
        """
        Initialize converter.
        
        Args:
            mot_dir: Path to MOT dataset directory
            output_dir: Path to save YOLO format dataset
        """
        self.mot_dir = Path(mot_dir)
        self.output_dir = Path(output_dir)
        
        print(f"MOT to YOLO Converter")
        print(f"Input: {self.mot_dir}")
        print(f"Output: {self.output_dir}")
    
    def convert_sequence(self, sequence_path: Path):
        """Convert a single MOT sequence to YOLO format."""
        seq_name = sequence_path.name
        print(f"\nConverting {seq_name}...")
        
        # Paths
        img_dir = sequence_path / "img1"
        gt_file = sequence_path / "gt" / "gt.txt"
        seqinfo_file = sequence_path / "seqinfo.ini"
        
        if not gt_file.exists():
            print(f"  ⚠️  Ground truth not found, skipping")
            return
        
        # Read sequence info
        img_width, img_height = self._read_seqinfo(seqinfo_file)
        
        # Create output directories
        split_name = "train" if "train" in str(sequence_path) else "test"
        out_img_dir = self.output_dir / split_name / "images" / seq_name
        out_label_dir = self.output_dir / split_name / "labels" / seq_name
        
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Read ground truth
        gt_data = np.loadtxt(gt_file, delimiter=',')
        
        # Group by frame
        frames = {}
        for row in gt_data:
            frame_id = int(row[0])
            if frame_id not in frames:
                frames[frame_id] = []
            
            # MOT format: frame, id, x, y, w, h, conf, class, visibility
            track_id = int(row[1])
            x, y, w, h = row[2:6]
            conf = row[6] if len(row) > 6 else 1.0
            class_id = int(row[7]) if len(row) > 7 else 1
            visibility = row[8] if len(row) > 8 else 1.0
            
            # Filter: only keep pedestrians (class 1) with good visibility
            if class_id == 1 and visibility > 0.3:
                frames[frame_id].append((x, y, w, h, conf))
        
        # Convert each frame
        for frame_id, detections in tqdm(frames.items(), desc=f"  {seq_name}"):
            # Image file
            img_file = img_dir / f"{frame_id:06d}.jpg"
            if not img_file.exists():
                continue
            
            # Copy image
            out_img_file = out_img_dir / f"{frame_id:06d}.jpg"
            if not out_img_file.exists():
                shutil.copy(img_file, out_img_file)
            
            # Create YOLO label file
            out_label_file = out_label_dir / f"{frame_id:06d}.txt"
            
            with open(out_label_file, 'w') as f:
                for x, y, w, h, conf in detections:
                    # Convert to YOLO format (class, x_center, y_center, width, height) - normalized
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Class 0 for person in COCO/YOLO
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        print(f"  ✓ Converted {len(frames)} frames")
    
    def _read_seqinfo(self, seqinfo_file: Path) -> tuple:
        """Read image dimensions from seqinfo.ini."""
        if not seqinfo_file.exists():
            # Default MOT17 resolution
            return 1920, 1080
        
        width, height = 1920, 1080
        
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if 'imWidth' in line:
                    width = int(line.split('=')[1].strip())
                elif 'imHeight' in line:
                    height = int(line.split('=')[1].strip())
        
        return width, height
    
    def convert_all(self, split: str = 'train'):
        """
        Convert all sequences in a split.
        
        Args:
            split: Dataset split ('train' or 'test')
        """
        split_dir = self.mot_dir / split
        
        if not split_dir.exists():
            print(f"Error: {split_dir} not found")
            return
        
        # Find all sequences
        sequences = list(split_dir.glob("MOT17-*-*"))
        
        print(f"\nFound {len(sequences)} sequences in {split} split")
        
        for seq in sequences:
            self.convert_sequence(seq)
        
        # Create dataset.yaml for YOLO
        self._create_dataset_yaml()
        
        print("\n✅ Conversion complete!")
    
    def _create_dataset_yaml(self):
        """Create YOLO dataset.yaml configuration."""
        yaml_content = f"""# MOT17 Dataset in YOLO Format
# Auto-generated by MOT to YOLO converter

path: {self.output_dir.absolute()}  # Dataset root dir
train: train/images  # Train images (relative to 'path')
val: train/images    # Val images (can splitlater)
test: test/images    # Test images

# Classes
names:
  0: person

# Number of classes
nc: 1

# Additional info
info:
  dataset: MOT17
  format: YOLO
  task: detection
  description: MOT Challenge 2017 converted to YOLO format
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created dataset.yaml: {yaml_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Convert MOT to YOLO format')
    parser.add_argument('--input', type=str, required=True, help='MOT dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for YOLO format')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'both'])
    
    args = parser.parse_args()
    
    converter = MOTToYOLOConverter(
        mot_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    
    if args.split == 'both':
        converter.convert_all('train')
        converter.convert_all('test')
    else:
        converter.convert_all(args.split)
    
    print("\n✅ Done!")
    print(f"\nDataset ready for YOLO training at: {args.output}")
    print("Use the generated dataset.yaml file for training.")


if __name__ == '__main__':
    main()
