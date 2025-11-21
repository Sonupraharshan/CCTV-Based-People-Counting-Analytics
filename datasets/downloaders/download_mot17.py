"""
MOT17 Dataset Downloader
Downloads and prepares MOT17 dataset for training and tracking evaluation.
"""

import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import shutil


class MOT17Downloader:
    """Download and prepare MOT17 dataset."""
    
    # MOT17 dataset information
    # Note: MOT Challenge requires registration, so provide manual instructions
    DATASET_INFO = {
        'train': {
            'url': 'https://motchallenge.net/data/MOT17.zip',  # Requires auth
            'size': '5.5 GB',
            'sequences': 7
        },
        'test': {
            'url': 'https://motchallenge.net/data/MOT17.zip',
            'size': '5.0 GB',
            'sequences': 7
        }
    }
    
    def __init__(self, output_dir: str = "datasets/MOT17"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"MOT17 Downloader initialized")
        print(f"Output directory: {self.output_dir}")
    
    def show_manual_instructions(self):
        """Show manual download instructions."""
        print("\n" + "=" * 70)
        print("MOT17 Dataset - Manual Download Instructions")
        print("=" * 70)
        print("\nThe MOT17 dataset requires registration on the MOT Challenge website.")
        print("\nSteps to download:")
        print("\n1. Create account at: https://motchallenge.net/")
        print("2. Go to: https://motchallenge.net/data/MOT17/")
        print("3. Download MOT17 dataset (both train and test)")
        print(f"4. Extract the zip file to: {self.output_dir.absolute()}")
        print("\nExpected directory structure:")
        print("  MOT17/")
        print("    ├── train/")
        print("    │   ├── MOT17-02-DPM/")
        print("    │   ├── MOT17-04-DPM/")
        print("    │   └── ...")
        print("    └── test/")
        print("        ├── MOT17-01-DPM/")
        print("        └── ...")
        print("\n" + "=" * 70)
    
    def verify_structure(self) -> bool:
        """Verify dataset structure exists."""
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"
        
        train_exists = train_dir.exists() and any(train_dir.iterdir())
        test_exists = test_dir.exists() and any(test_dir.iterdir())
        
        if train_exists and test_exists:
            print("✅ MOT17 dataset found and verified!")
            self.show_statistics()
            return True
        elif train_exists:
            print("⚠️  MOT17 train split found, but test split missing")
            return False
        elif test_exists:
            print("⚠️  MOT17 test split found, but train split missing")
            return False
        else:
            print("❌ MOT17 dataset not found")
            self.show_manual_instructions()
            return False
    
    def show_statistics(self):
        """Show dataset statistics."""
        print("\n" + "=" * 70)
        print("MOT17 Dataset Statistics")
        print("=" * 70)
        
        for split in ['train', 'test']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                sequences = list(split_dir.glob("MOT17-*"))
                print(f"\n{split.upper()} split:")
                print(f"  - Sequences: {len(sequences)}")
                
                for seq in sorted(sequences):
                    img_dir = seq / "img1"
                    if img_dir.exists():
                        num_frames = len(list(img_dir.glob("*.jpg")))
                        print(f"    • {seq.name}: {num_frames} frames")
        
        print("=" * 70)
    
    def prepare_for_yolo(self):
        """Prepare MOT17 annotations for YOLO training."""
        print("\nPreparing MOT17 for YOLO format...")
        print("Use datasets/converters/mot_to_yolo.py to convert annotations")
    
    def get_info(self):
        """Print dataset information."""
        print("\n" + "=" * 70)
        print("MOT17 Dataset Information")
        print("=" * 70)
        print("\nMOT17 (Multiple Object Tracking Benchmark 2017)")
        print("\nDataset Details:")
        print("  - 14 sequences (7 train, 7 test)")
        print("  - HD resolution (1920x1080)")
        print("  - Varied scenarios: crowded, sparse, indoor, outdoor")
        print("  - Ground truth tracks with IDs")
        print("  - Ideal for training detectors and evaluating trackers")
        print("\nMetrics Supported:")
        print("  - MOTA (Multiple Object Tracking Accuracy)")
        print("  - IDF1 (ID F1 Score)")
        print("  - FP, FN, ID switches")
        print("\nWebsite:")
        print("  https://motchallenge.net/")
        print("=" * 70)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare MOT17 dataset')
    parser.add_argument('--output', type=str, default='datasets/MOT17', help='Output directory')
    parser.add_argument('--verify', action='store_true', help='Verify existing dataset')
    parser.add_argument('--info', action='store_true', help='Show dataset information')
    parser.add_argument('--prepare-yolo', action='store_true', help='Prepare for YOLO training')
    
    args = parser.parse_args()
    
    downloader = MOT17Downloader(output_dir=args.output)
    
    if args.info:
        downloader.get_info()
    elif args.verify:
        downloader.verify_structure()
    elif args.prepare_yolo:
        if downloader.verify_structure():
            downloader.prepare_for_yolo()
    else:
        # Show manual instructions by default
        downloader.show_manual_instructions()
        print("\nRun with --verify to check if dataset is properly installed")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
