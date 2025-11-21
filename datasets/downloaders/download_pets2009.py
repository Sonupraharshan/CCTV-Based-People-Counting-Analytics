"""
PETS2009 S2L1 Dataset Downloader
Downloads and prepares PETS2009 dataset for counting evaluation.
"""

import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import shutil


class PETS2009Downloader:
    """Download and prepare PETS2009 S2L1 dataset."""
    
    # PETS2009 dataset URLs (Note: Update these with actual URLs)
    URLS = {
        'S2L1': 'http://www.cvg.reading.ac.uk/PETS2009/a.zip',  # Example URL
        # Add more sequences as needed
    }
    
    def __init__(self, output_dir: str = "datasets/PETS2009"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PETS2009 Downloader initialized")
        print(f"Output directory: {self.output_dir}")
    
    def download_file(self, url: str, output_path: Path):
        """Download file with progress bar."""
        print(f"Downloading from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"✓ Downloaded: {output_path}")
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP archive."""
        print(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✓ Extracted to: {extract_to}")
    
    def download_sequence(self, sequence: str = 'S2L1'):
        """
        Download a specific PETS2009 sequence.
        
        Args:
            sequence: Sequence name (e.g., 'S2L1')
        """
        if sequence not in self.URLS:
            print(f"Warning: URL for {sequence} not available")
            print(f"Available sequences: {list(self.URLS.keys())}")
            return False
        
        url = self.URLS[sequence]
        zip_path = self.output_dir / f"{sequence}.zip"
        extract_path = self.output_dir / sequence
        
        # Download if not exists
        if not zip_path.exists():
            try:
                self.download_file(url, zip_path)
            except Exception as e:
                print(f"Error downloading: {e}")
                print("\nAlternative: Manual download instructions:")
                print(f"1. Visit: http://www.cvg.reading.ac.uk/PETS2009/")
                print(f"2. Download {sequence} dataset manually")
                print(f"3. Place in {self.output_dir}")
                return False
        
        # Extract
        if not extract_path.exists():
            self.extract_zip(zip_path, self.output_dir)
        else:
            print(f"✓ Already extracted: {extract_path}")
        
        # Organize structure
        self.organize_structure(sequence)
        
        return True
    
    def organize_structure(self, sequence: str):
        """Organize dataset into standard structure."""
        seq_path = self.output_dir / sequence
        
        # Create standard directories
        (seq_path / "images").mkdir(exist_ok=True)
        (seq_path / "annotations").mkdir(exist_ok=True)
        (seq_path / "ground_truth").mkdir(exist_ok=True)
        
        print(f"✓ Organized structure for {sequence}")
    
    def download_all(self):
        """Download all available sequences."""
        for sequence in self.URLS.keys():
            self.download_sequence(sequence)
    
    def get_info(self):
        """Print dataset information."""
        print("\n" + "=" * 60)
        print("PETS2009 Dataset Information")
        print("=" * 60)
        print("\nPETS2009 is a benchmark dataset for people tracking and counting.")
        print("\nS2L1 Sequence:")
        print("  - People walking in different directions")
        print("  - Ground truth annotations available")
        print("  - Ideal for counting accuracy evaluation")
        print("\nFor more information:")
        print("  http://www.cvg.reading.ac.uk/PETS2009/")
        print("=" * 60)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download PETS2009 dataset')
    parser.add_argument('--output', type=str, default='datasets/PETS2009', help='Output directory')
    parser.add_argument('--sequence', type=str, default='S2L1', help='Sequence to download')
    parser.add_argument('--all', action='store_true', help='Download all sequences')
    parser.add_argument('--info', action='store_true', help='Show dataset information')
    
    args = parser.parse_args()
    
    downloader = PETS2009Downloader(output_dir=args.output)
    
    if args.info:
        downloader.get_info()
    elif args.all:
        downloader.download_all()
    else:
        downloader.download_sequence(args.sequence)
    
    print("\n✅ Done!")
    print("\nNOTE: If automatic download fails, please download manually from:")
    print("http://www.cvg.reading.ac.uk/PETS2009/")
    print(f"And extract to: {args.output}")


if __name__ == '__main__':
    main()
