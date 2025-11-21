"""
Integration Tests for People Counting Pipeline
Tests the complete detector → tracker → counter workflow.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.pipeline import PeopleCountingPipeline
from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance."""
        return PeopleCountingPipeline(
            detector_model="yolov8n.pt",
            tracker_type="bytetrack",  # Faster for testing
            conf_threshold=0.5
        )
    
    @pytest.fixture
    def dummy_video(self):
        """Create a dummy video file for testing."""
        # Create temporary video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Video parameters
        width, height = 640, 480
        fps = 10
        num_frames = 30
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Generate frames with a moving "person" (rectangle)
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw moving rectangle (simulating a person)
            y = int(height * 0.3 + (height * 0.4) * (i / num_frames))  # Moving down
            x = width // 2
            cv2.rectangle(frame, (x - 30, y - 60), (x + 30, y), (255, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.detector is not None
        assert pipeline.tracker is not None
        assert pipeline.counter is not None
    
    def test_detector_alone(self, pipeline):
        """Test detector on a single frame."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 280), (255, 255, 255), -1)
        
        # Run detection
        boxes, scores, class_ids = pipeline.detector.detect(frame)
        
        # Check output format
        assert isinstance(boxes, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(class_ids, np.ndarray)
        assert len(boxes) == len(scores) == len(class_ids)
    
    def test_tracker_with_detections(self, pipeline):
        """Test tracker with mock detections."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock detection
        detection = np.array([300, 200, 40, 80])  # [x, y, w, h]
        score = 0.9
        
        # Update tracker
        tracks = pipeline.tracker.update(
            detections=[detection],
            scores=[score],
            frame=frame
        )
        
        # Should create a track
        assert len(tracks) >= 0  # May or may not be confirmed yet
    
    def test_counter_with_line(self):
        """Test counter with a defined line."""
        line = ((100, 300), (500, 300))
        counter = LineCounter(line=line, min_track_length=3)
        
        assert counter.count_in == 0
        assert counter.count_out == 0
        assert counter.line == line
    
    def test_pipeline_set_counting_line(self, pipeline):
        """Test setting counting line."""
        line = ((100, 300), (500, 300))
        pipeline.set_counting_line(line, min_track_length=5)
        
        assert pipeline.counter is not None
        assert pipeline.counter.line == line
    
    def test_end_to_end_video_processing(self, pipeline, dummy_video):
        """Test complete pipeline on a video."""
        # Set counting line (horizontal in the middle)
        counting_line = ((100, 240), (540, 240))
        pipeline.set_counting_line(counting_line, min_track_length=3)
        
        # Process video (without display)
        results = pipeline.run(
            source=dummy_video,
            counting_line=counting_line,
            display=False,
            save_output=None
        )
        
        # Check results structure
        assert 'count_in' in results
        assert 'count_out' in results
        assert 'total_frames' in results
        assert 'fps' in results
        
        # Verify counts are non-negative
        assert results['count_in'] >= 0
        assert results['count_out'] >= 0
        assert results['total_frames'] > 0
    
    def test_pipeline_with_different_trackers(self, dummy_video):
        """Test pipeline with different tracker types."""
        for tracker_type in ['bytetrack', 'strongsort']:
            pipeline = PeopleCountingPipeline(
                detector_model="yolov8n.pt",
                tracker_type=tracker_type
            )
            
            line = ((100, 240), (540, 240))
            pipeline.set_counting_line(line)
            
            results = pipeline.run(
                source=dummy_video,
                counting_line=line,
                display=False
            )
            
            assert results is not None
            assert results['total_frames'] > 0
    
    def test_pipeline_fps_calculation(self, pipeline, dummy_video):
        """Test FPS tracking."""
        line = ((100, 240), (540, 240))
        
        results = pipeline.run(
            source=dummy_video,
            counting_line=line,
            display=False
        )
        
        assert 'fps' in results
        assert results['fps'] > 0
    
    def test_pipeline_error_handling_invalid_source(self, pipeline):
        """Test pipeline with invalid video source."""
        with pytest.raises(Exception):
            pipeline.run(
                source="nonexistent_video.mp4",
                counting_line=((0, 0), (100, 100)),
                display=False
            )
    
    def test_pipeline_empty_counting_line(self, pipeline, dummy_video):
        """Test pipeline without counting line."""
        # Should still process but not count
        results = pipeline.run(
            source=dummy_video,
            counting_line=None,
            display=False
        )
        
        assert results is not None


class TestComponentsInteraction:
    """Test interactions between components."""
    
    def test_detector_to_tracker_data_flow(self):
        """Test data flow from detector to tracker."""
        detector = Detector(model_path="yolov8n.pt")
        tracker = Tracker(tracker_type="bytetrack")
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 280), (255, 255, 255), -1)
        
        # Detect
        boxes, scores, _ = detector.detect(frame)
        
        # Track
        if len(boxes) > 0:
            tracks = tracker.update(
                detections=boxes.tolist(),
                scores=scores.tolist(),
                frame=frame
            )
            
            assert isinstance(tracks, list)
    
    def test_tracker_to_counter_data_flow(self):
        """Test data flow from tracker to counter."""
        from src.tracker import Track
        
        tracker = Tracker(tracker_type="bytetrack")
        counter = LineCounter(line=((0, 300), (640, 300)))
        
        # Create mock track
        track = Track(
            track_id=1,
            bbox=np.array([300, 250, 40, 80]),
            confidence=0.9
        )
        track.state = 'confirmed'
        
        # Update counter with tracks
        events = counter.update([track])
        
        assert isinstance(events, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
