"""
Main Processing Pipeline
Orchestrates detector → tracker → counter workflow with live visualization.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pathlib import Path

from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter, CountEvent
from src.utils.video_reader import VideoReader, VideoWriter
from src.utils.overlay import (
    draw_tracks,
    draw_counting_line,
    draw_count_overlay,
    draw_fps
)


class PeopleCountingPipeline:
    """
    End-to-end pipeline for people counting from video/RTSP streams.
    
    Pipeline: Video → Detector → Tracker → Counter → Visualization
    """
    
    def __init__(
        self,
        detector_model: str = "yolov8n.pt",
        tracker_type: str = "strongsort",
        conf_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize pipeline components.
        
        Args:
            detector_model: Path to YOLOv8 model
            tracker_type: Tracker type ('strongsort' or 'bytetrack')
            conf_threshold: Detection confidence threshold
            device: Device for inference
        """
        print("=" * 50)
        print("Initializing People Counting Pipeline")
        print("=" * 50)
        
        # Initialize detector
        self.detector = Detector(
            model_path=detector_model,
            device=device,
            conf_threshold=conf_threshold
        )
        
        # Initialize tracker
        self.tracker = Tracker(
            tracker_type=tracker_type,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Counter will be set later
        self.counter: Optional[LineCounter] = None
        
        # Stats
        self.frame_count = 0
        self.fps = 0.0
        
        print("=" * 50)
        print("Pipeline initialized successfully")
        print("=" * 50)
    
    def set_counting_line(
        self,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
        min_track_length: int = 10
    ):
        """Set or update the counting line."""
        self.counter = LineCounter(
            line=line,
            min_track_length=min_track_length
        )
        print(f"Counting line set: {line}")
    
    def run(
        self,
        source: str,
        counting_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        output_path: Optional[str] = None,
        display: bool = True,
        save_logs: bool = True,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            source: Video file path or RTSP URL
            counting_line: Optional counting line ((x1, y1), (x2, y2))
            output_path: Optional path to save output video
            display: Whether to display live preview
            save_logs: Whether to save event logs
            max_frames: Maximum frames to process (None for all)
        
        Returns:
            Dict with statistics and results
        """
        # Setup counting line
        if counting_line is not None:
            self.set_counting_line(counting_line)
        
        if self.counter is None:
            raise ValueError("Counting line not set. Call set_counting_line() first.")
        
        # Initialize video reader
        print(f"\nOpening video source: {source}")
        video_reader = VideoReader(source)
        video_reader.start()
        
        # Initialize video writer if needed
        video_writer = None
        if output_path is not None:
            video_writer = VideoWriter(
                output_path,
                fps=video_reader.fps,
                frame_size=(video_reader.width, video_reader.height)
            )
        
        # Processing loop
        print("\nStarting processing...")
        print("Press 'Q' to quit\n")
        
        fps_counter = FPSCounter()
        
        try:
            while video_reader.is_opened():
                # Read frame
                ret, frame, frame_num = video_reader.read(timeout=2.0)
                
                if not ret:
                    break
                
                if max_frames and frame_num >= max_frames:
                    break
                
                # Start FPS timer
                fps_counter.start()
                
                # Run detection
                boxes, scores, class_ids = self.detector.detect(frame)
                
                # Update tracker
                tracks = self.tracker.update(
                    detections=boxes if len(boxes) > 0 else [],
                    scores=scores if len(scores) > 0 else [],
                    frame=frame
                )
                
                # Update counter
                events = self.counter.update(tracks)
                
                # Draw visualizations
                vis_frame = frame.copy()
                
                # Draw tracks
                vis_frame = draw_tracks(
                    vis_frame,
                    tracks,
                    draw_trajectory=True,
                    trajectory_length=30
                )
                
                # Draw counting line
                vis_frame = draw_counting_line(vis_frame, self.counter.line)
                
                # Draw count overlay
                vis_frame = draw_count_overlay(
                    vis_frame,
                    self.counter.count_in,
                    self.counter.count_out,
                    position='top-right'
                )
                
                # Draw FPS
                fps_counter.update()
                fps = fps_counter.get_fps()
                vis_frame = draw_fps(vis_frame, fps)
                
                # Log events
                for event in events:
                    print(f"Frame {frame_num}: Track {event.track_id} crossed {event.direction} "
                          f"(IN={self.counter.count_in}, OUT={self.counter.count_out})")
                
                # Display
                if display:
                    cv2.imshow('People Counting', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\nUser quit")
                        break
                
                # Write to output video
                if video_writer:
                    video_writer.write(vis_frame)
                
                # Update stats
                self.frame_count += 1
                
                # Progress update
                if frame_num % 100 == 0:
                    print(f"Processed {frame_num} frames | FPS: {fps:.1f} | "
                          f"IN: {self.counter.count_in} | OUT: {self.counter.count_out}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            video_reader.stop()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        results = {
            'total_frames': self.frame_count,
            'count_in': self.counter.count_in,
            'count_out': self.counter.count_out,
            'net_count': self.counter.get_net_count(),
            'total_count': self.counter.get_total_count(),
            'events': self.counter.export_events_dict(),
            'avg_fps': fps_counter.get_fps()
        }
        
        print("\n" + "=" * 50)
        print("Processing Complete")
        print("=" * 50)
        print(f"Total frames: {results['total_frames']}")
        print(f"IN count: {results['count_in']}")
        print(f"OUT count: {results['count_out']}")
        print(f"NET count: {results['net_count']}")
        print(f"Average FPS: {results['avg_fps']:.2f}")
        print("=" * 50)
        
        return results
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        if self.counter:
            self.counter.reset()
        self.frame_count = 0
        print("Pipeline reset")


class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            buffer_size: Number of frames to average over
        """
        self.buffer_size = buffer_size
        self.times = []
        self.start_time = None
    
    def start(self):
        """Start timing for current frame."""
        self.start_time = time.time()
    
    def update(self):
        """Update FPS calculation."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            
            if len(self.times) > self.buffer_size:
                self.times = self.times[-self.buffer_size:]
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.times) == 0:
            return 0.0
        
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CCTV People Counting Pipeline')
    parser.add_argument('--source', type=str, required=True, help='Video file or RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--tracker', type=str, default='strongsort', choices=['strongsort', 'bytetrack'])
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--line', type=str, help='Counting line: x1,y1,x2,y2')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable live display')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Parse counting line
    counting_line = None
    if args.line:
        coords = list(map(float, args.line.split(',')))
        if len(coords) != 4:
            raise ValueError("Line must be: x1,y1,x2,y2")
        counting_line = ((coords[0], coords[1]), (coords[2], coords[3]))
    else:
        # Default line (horizontal at middle)
        print("Warning: No counting line specified, using default")
        counting_line = ((100, 400), (900, 400))
    
    # Initialize pipeline
    pipeline = PeopleCountingPipeline(
        detector_model=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf
    )
    
    # Run
    results = pipeline.run(
        source=args.source,
        counting_line=counting_line,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
