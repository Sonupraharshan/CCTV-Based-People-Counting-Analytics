"""
Tracking Quality Evaluation on MOT17
Computes MOTA, IDF1, and other MOT Challenge metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import motmetrics as mm
from typing import Dict, List
import json

from src.pipeline import PeopleCountingPipeline


class TrackingEvaluator:
    """Evaluate tracking quality using MOT metrics."""
    
    def __init__(self, dataset_dir: Path, model_path: str, tracker_type: str = 'strongsort'):
        """
        Initialize evaluator.
        
        Args:
            dataset_dir: Path to MOT17 dataset
            model_path: Path to YOLOv8 model
            tracker_type: Tracker type ('strongsort' or 'bytetrack')
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_path = model_path
        self.tracker_type = tracker_type
        
        # Initialize pipeline
        self.pipeline = PeopleCountingPipeline(
            detector_model=model_path,
            tracker_type=tracker_type,
            conf_threshold=0.5
        )
        
        print(f"Tracking Evaluator initialized:")
        print(f"  Dataset: {dataset_dir}")
        print(f"  Model: {model_path}")
        print(f"  Tracker: {tracker_type}")
    
    def load_mot_gt(self, sequence_path: Path) -> Dict[int, List]:
        """
        Load MOT format ground truth.
        
        Args:
            sequence_path: Path to sequence directory
        
        Returns:
            Dict mapping frame_id to list of (track_id, bbox) tuples
        """
        gt_file = sequence_path / "gt" / "gt.txt"
        
        if not gt_file.exists():
            print(f"Warning: Ground truth not found at {gt_file}")
            return {}
        
        # Parse ground truth
        # MOT format: frame, id, x, y, w, h, conf, class, visibility
        gt_data = np.loadtxt(gt_file, delimiter=',')
        
        ground_truth = {}
        for row in gt_data:
            frame_id = int(row[0])
            track_id = int(row[1])
            x, y, w, h = row[2:6]
            class_id = int(row[7]) if len(row) > 7 else 1
            visibility = row[8] if len(row) > 8 else 1.0
            
            # Filter: only pedestrians with good visibility
            if class_id == 1 and visibility > 0.3:
                if frame_id not in ground_truth:
                    ground_truth[frame_id] = []
                
                # Convert to [x, y, x+w, y+h] format
                bbox = [x, y, x + w, y + h]
                ground_truth[frame_id].append((track_id, bbox))
        
        return ground_truth
    
    def run_tracking(self, video_path: Path) -> Dict[int, List]:
        """
        Run tracking on video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dict mapping frame_id to list of (track_id, bbox) tuples
        """
        import cv2
        from src.utils.video_reader import VideoReader
        
        predictions = {}
        
        reader = VideoReader(str(video_path))
        reader.start()
        
        frame_id = 1
        
        try:
            while True:
                ret, frame, _ = reader.read(timeout=1.0)
                if not ret:
                    break
                
                # Detect
                boxes, scores, _ = self.pipeline.detector.detect(frame)
                
                # Track
                tracks = self.pipeline.tracker.update(
                    detections=[boxes[i] for i in range(len(boxes))],
                    scores=scores.tolist(),
                    frame=frame
                )
                
                # Store predictions
                predictions[frame_id] = []
                for track in tracks:
                    bbox = track.bbox  # [x, y, w, h]
                    # Convert to [x, y, x+w, y+h]
                    bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    predictions[frame_id].append((track.track_id, bbox_xyxy))
                
                frame_id += 1
                
        finally:
            reader.stop()
        
        return predictions
    
    def compute_mot_metrics(
        self,
        ground_truth: Dict[int, List],
        predictions: Dict[int, List],
        iou_threshold: float = 0.5
    ) -> mm.MOTAccumulator:
        """
        Compute MOT metrics.
        
        Args:
            ground_truth: GT tracks per frame
            predictions: Predicted tracks per frame
            iou_threshold: IoU threshold for matching
        
        Returns:
            MOTAccumulator with computed metrics
        """
        # Create accumulator
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Get all frames
        all_frames = sorted(set(list(ground_truth.keys()) + list(predictions.keys())))
        
        for frame_id in all_frames:
            gt_tracks = ground_truth.get(frame_id, [])
            pred_tracks = predictions.get(frame_id, [])
            
            # Extract IDs and boxes
            gt_ids = [t[0] for t in gt_tracks]
            gt_boxes = [t[1] for t in gt_tracks]
            
            pred_ids = [t[0] for t in pred_tracks]
            pred_boxes = [t[1] for t in pred_tracks]
            
            # Compute IoU distance matrix
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                distances = mm.distances.iou_matrix(
                    gt_boxes, pred_boxes, max_iou=1 - iou_threshold
                )
            else:
                distances = np.array([]).reshape(len(gt_boxes), len(pred_boxes))
            
            # Update accumulator
            acc.update(gt_ids, pred_ids, distances)
        
        return acc
    
    def evaluate_sequence(self, sequence: str) -> Dict:
        """
        Evaluate tracking on a single sequence.
        
        Args:
            sequence: Sequence name
        
        Returns:
            Dict with evaluation metrics
        """
        print(f"\nEvaluating sequence: {sequence}")
        
        # Paths
        sequence_path = self.dataset_dir / "train" / sequence
        video_path = sequence_path / "img1"  # Directory with images
        
        if not sequence_path.exists():
            print(f"  Error: Sequence not found at {sequence_path}")
            return None
        
        # Load ground truth
        print(f"  Loading ground truth...")
        ground_truth = self.load_mot_gt(sequence_path)
        
        # Create video from images if needed
        import cv2
        images = sorted(video_path.glob("*.jpg"))
        if len(images) == 0:
            print(f"  Error: No images found in {video_path}")
            return None
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(images[0]))
        h, w = first_img.shape[:2]
        
        # Create temporary video
        temp_video = sequence_path / f"{sequence}_temp.mp4"
        if not temp_video.exists():
            print(f"  Creating temporary video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_video), fourcc, 25.0, (w, h))
            
            for img_path in images:
                img = cv2.imread(str(img_path))
                out.write(img)
            
            out.release()
        
        # Run tracking
        print(f"  Running tracker...")
        predictions = self.run_tracking(temp_video)
        
        # Compute metrics
        print(f"  Computing metrics...")
        acc = self.compute_mot_metrics(ground_truth, predictions)
        
        # Calculate summary metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_switches', 
                    'num_false_positives', 'num_misses', 'num_detections',
                    'num_objects', 'num_predictions', 'num_unique_objects',
                    'mostly_tracked', 'partially_tracked', 'mostly_lost',
                    'precision', 'recall'],
            name=sequence
        )
        
        # Extract metrics
        metrics = {
            'sequence': sequence,
            'num_frames': int(summary['num_frames'].values[0]),
            'mota': float(summary['mota'].values[0]),
            'motp': float(summary['motp'].values[0]),
            'idf1': float(summary['idf1'].values[0]),
            'num_switches': int(summary['num_switches'].values[0]),
            'false_positives': int(summary['num_false_positives'].values[0]),
            'misses': int(summary['num_misses'].values[0]),
            'precision': float(summary['precision'].values[0]),
            'recall': float(summary['recall'].values[0]),
            'mostly_tracked': int(summary['mostly_tracked'].values[0]),
            'partially_tracked': int(summary['partially_tracked'].values[0]),
            'mostly_lost': int(summary['mostly_lost'].values[0])
        }
        
        print(f"  Results:")
        print(f"    MOTA: {metrics['mota']:.2%}")
        print(f"    IDF1: {metrics['idf1']:.2%}")
        print(f"    ID Switches: {metrics['num_switches']}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall: {metrics['recall']:.2%}")
        
        return metrics
    
    def evaluate_all(self, sequences: List[str] = None) -> pd.DataFrame:
        """
        Evaluate all sequences.
        
        Args:
            sequences: List of sequence names (if None, evaluate all)
        
        Returns:
            DataFrame with all results
        """
        if sequences is None:
            # Auto-detect sequences
            train_dir = self.dataset_dir / "train"
            sequences = [d.name for d in train_dir.glob("MOT17-*-DPM")]  # Use DPM detections
        
        results = []
        
        for sequence in sequences:
            result = self.evaluate_sequence(sequence)
            if result:
                results.append(result)
        
        # Create summary DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Overall statistics
            print("\n" + "=" * 80)
            print("Overall Tracking Statistics")
            print("=" * 80)
            print(f"Average MOTA: {df['mota'].mean():.2%}")
            print(f"Average IDF1: {df['idf1'].mean():.2%}")
            print(f"Average Precision: {df['precision'].mean():.2%}")
            print(f"Average Recall: {df['recall'].mean():.2%}")
            print(f"Total ID Switches: {df['num_switches'].sum()}")
            print(f"Total False Positives: {df['false_positives'].sum()}")
            print(f"Total Misses: {df['misses'].sum()}")
            print("=" * 80)
            
            return df
        
        return pd.DataFrame()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Evaluate tracking metrics on MOT17')
    parser.add_argument('--dataset', type=str, required=True, help='MOT17 dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model checkpoint')
    parser.add_argument('--tracker', type=str, default='strongsort', choices=['strongsort', 'bytetrack'])
    parser.add_argument('--sequences', type=str, nargs='+', help='Specific sequences to evaluate')
    parser.add_argument('--output', type=str, default='eval/tracking_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TrackingEvaluator(
        dataset_dir=args.dataset,
        model_path=args.model,
        tracker_type=args.tracker
    )
    
    # Evaluate
    df = evaluator.evaluate_all(sequences=args.sequences)
    
    # Save results
    if not df.empty:
        # Save CSV
        csv_path = args.output.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        # Save JSON
        results_dict = df.to_dict(orient='records')
        with open(args.output, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✓ Detailed results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
