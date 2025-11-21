"""
Counting Accuracy Evaluation on PETS2009
Compares predicted counts vs ground truth and calculates FP/FN rates.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
import json

from src.pipeline import PeopleCountingPipeline


class CountingEvaluator:
    """Evaluate counting accuracy on PETS2009 dataset."""
    
    def __init__(self, dataset_dir: Path, model_path: str):
        """
        Initialize evaluator.
        
        Args:
            dataset_dir: Path to PETS2009 dataset
            model_path: Path to YOLOv8 model
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_path = model_path
        
        # Initialize pipeline
        self.pipeline = PeopleCountingPipeline(
            detector_model=model_path,
            tracker_type='strongsort',
            conf_threshold=0.5
        )
        
        print(f"Counting Evaluator initialized:")
        print(f"  Dataset: {dataset_dir}")
        print(f"  Model: {model_path}")
    
    def load_ground_truth(self, sequence: str) -> Dict:
        """
        Load ground truth counts for a sequence.
        
        Args:
            sequence: Sequence name
        
        Returns:
            Dict with ground truth information
        """
        # This is a placeholder - actual implementation would load PETS2009 ground truth
        # PETS2009 provides frame-level annotations
        
        gt_file = self.dataset_dir / sequence / "ground_truth" / "counts.txt"
        
        if not gt_file.exists():
            print(f"Warning: Ground truth not found for {sequence}")
            return {'total_in': 0, 'total_out': 0, 'frame_counts': {}}
        
        # Parse ground truth (format may vary)
        # Example format: frame_id,direction,count
        frame_counts = {}
        total_in = 0
        total_out = 0
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    frame_id = int(parts[0])
                    direction = parts[1]
                    count = int(parts[2])
                    
                    if direction == 'IN':
                        total_in += count
                    else:
                        total_out += count
                    
                    frame_counts[frame_id] = {direction: count}
        
        return {
            'total_in': total_in,
            'total_out': total_out,
            'frame_counts': frame_counts
        }
    
    def evaluate_sequence(
        self,
        sequence: str,
        counting_line: tuple
    ) -> Dict:
        """
        Evaluate counting on a single sequence.
        
        Args:
            sequence: Sequence name
            counting_line: Counting line coordinates
        
        Returns:
            Dict with evaluation results
        """
        print(f"\nEvaluating sequence: {sequence}")
        
        # Sequence video
        video_path = self.dataset_dir / sequence / "video.mp4"
        if not video_path.exists():
            # Try alternative path
            video_path = self.dataset_dir / sequence / f"{sequence}.mp4"
        
        if not video_path.exists():
            print(f"  Error: Video not found for {sequence}")
            return None
        
        # Load ground truth
        ground_truth = self.load_ground_truth(sequence)
        
        # Run pipeline
        results = self.pipeline.run(
            source=str(video_path),
            counting_line=counting_line,
            display=False,
            save_logs=False
        )
        
        # Calculate metrics
        pred_in = results['count_in']
        pred_out = results['count_out']
        gt_in = ground_truth['total_in']
        gt_out = ground_truth['total_out']
        
        # False positives and negatives
        fp_in = max(0, pred_in - gt_in)
        fn_in = max(0, gt_in - pred_in)
        fp_out = max(0, pred_out - gt_out)
        fn_out = max(0, gt_out - pred_out)
        
        # Accuracy metrics
        total_pred = pred_in + pred_out
        total_gt = gt_in + gt_out
        
        accuracy = 1 - abs(total_pred - total_gt) / max(total_gt, 1)
        precision = (total_pred - (fp_in + fp_out)) / max(total_pred, 1)
        recall = (total_gt - (fn_in + fn_out)) / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        eval_results = {
            'sequence': sequence,
            'ground_truth': {
                'in': gt_in,
                'out': gt_out,
                'total': total_gt
            },
            'predicted': {
                'in': pred_in,
                'out': pred_out,
                'total': total_pred
            },
            'metrics': {
                'fp_in': fp_in,
                'fn_in': fn_in,
                'fp_out': fp_out,
                'fn_out': fn_out,
                'total_fp': fp_in + fp_out,
                'total_fn': fn_in + fn_out,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }
        
        print(f"  Results:")
        print(f"    Ground Truth - IN: {gt_in}, OUT: {gt_out}, Total: {total_gt}")
        print(f"    Predicted    - IN: {pred_in}, OUT: {pred_out}, Total: {total_pred}")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    FP: {fp_in + fp_out}, FN: {fn_in + fn_out}")
        
        return eval_results
    
    def evaluate_all(self, counting_lines: Dict[str, tuple]) -> pd.DataFrame:
        """
        Evaluate all sequences.
        
        Args:
            counting_lines: Dict mapping sequence names to counting lines
        
        Returns:
            DataFrame with all results
        """
        results = []
        
        for sequence, line in counting_lines.items():
            result = self.evaluate_sequence(sequence, line)
            if result:
                results.append(result)
        
        # Create summary DataFrame
        if results:
            summary_data = []
            for r in results:
                summary_data.append({
                    'Sequence': r['sequence'],
                    'GT_IN': r['ground_truth']['in'],
                    'GT_OUT': r['ground_truth']['out'],
                    'GT_Total': r['ground_truth']['total'],
                    'Pred_IN': r['predicted']['in'],
                    'Pred_OUT': r['predicted']['out'],
                    'Pred_Total': r['predicted']['total'],
                    'FP': r['metrics']['total_fp'],
                    'FN': r['metrics']['total_fn'],
                    'Accuracy': r['metrics']['accuracy'],
                    'Precision': r['metrics']['precision'],
                    'Recall': r['metrics']['recall'],
                    'F1': r['metrics']['f1_score']
                })
            
            df = pd.DataFrame(summary_data)
            
            # Overall statistics
            print("\n" + "=" * 80)
            print("Overall Statistics")
            print("=" * 80)
            print(f"Average Accuracy: {df['Accuracy'].mean():.2%}")
            print(f"Average F1 Score: {df['F1'].mean():.4f}")
            print(f"Total FP: {df['FP'].sum()}")
            print(f"Total FN: {df['FN'].sum()}")
            print(f"FP Rate: {df['FP'].sum() / df['GT_Total'].sum():.2%}")
            print(f"FN Rate: {df['FN'].sum() / df['GT_Total'].sum():.2%}")
            print("=" * 80)
            
            return df, results
        
        return pd.DataFrame(), []


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Evaluate counting accuracy on PETS2009')
    parser.add_argument('--dataset', type=str, required=True, help='PETS2009 dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model checkpoint')
    parser.add_argument('--output', type=str, default='eval/counting_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CountingEvaluator(
        dataset_dir=args.dataset,
        model_path=args.model
    )
    
    # Define counting lines for each sequence (example)
    # These should be tuned for each camera view
    counting_lines = {
        'S2L1': ((100, 400), (900, 400)),  # Horizontal line at y=400
        # Add more sequences as needed
    }
    
    # Evaluate
    df, results = evaluator.evaluate_all(counting_lines)
    
    # Save results
    if not df.empty:
        # Save CSV
        csv_path = args.output.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        # Save JSON
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Detailed results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
