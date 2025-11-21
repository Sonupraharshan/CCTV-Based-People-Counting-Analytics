"""
Unit Tests for Counting Logic
Tests for line-crossing detection and anti-double-count mechanisms.
"""

import pytest
import numpy as np
from src.counter import LineCounter, CountEvent
from src.tracker import Track
from src.utils.geometry import (
    line_intersection,
    point_side_of_line,
    trajectory_crosses_line
)


class TestGeometry:
    """Test geometric utilities."""
    
    def test_line_intersection_basic(self):
        """Test basic line intersection."""
        # Intersecting lines
        line1 = ((0, 0), (10, 10))
        line2 = ((0, 10), (10, 0))
        
        intersection = line_intersection(line1, line2)
        
        assert intersection is not None
        assert abs(intersection[0] - 5.0) < 0.01
        assert abs(intersection[1] - 5.0) < 0.01
    
    def test_line_intersection_parallel(self):
        """Test parallel lines (no intersection)."""
        line1 = ((0, 0), (10, 0))
        line2 = ((0, 5), (10, 5))
        
        intersection = line_intersection(line1, line2)
        
        assert intersection is None
    
    def test_line_intersection_non_overlapping(self):
        """Test non-overlapping segments."""
        line1 = ((0, 0), (5, 0))
        line2 = ((10, 0), (15, 0))
        
        intersection = line_intersection(line1, line2)
        
        assert intersection is None
    
    def test_point_side_of_line(self):
        """Test point side determination."""
        # Horizontal line
        line = ((0, 5), (10, 5))
        
        # Point above line (should be positive)
        assert point_side_of_line((5, 10), line) > 0
        
        # Point below line (should be negative)
        assert point_side_of_line((5, 0), line) < 0
        
        # Point on line (should be zero)
        assert point_side_of_line((5, 5), line) == 0
    
    def test_trajectory_crosses_line_simple(self):
        """Test simple trajectory crossing."""
        # Vertical line at x=5
        line = ((5, 0), (5, 10))
        
        # Trajectory crossing from left to right
        trajectory = [(0, 5), (10, 5)]
        
        crossed, direction = trajectory_crosses_line(trajectory, line)
        
        assert crossed is True
        assert direction in ['IN', 'OUT']
    
    def test_trajectory_no_crossing(self):
        """Test trajectory that doesn't cross."""
        line = ((5, 0), (5, 10))
        
        # Trajectory on one side only
        trajectory = [(0, 5), (3, 5)]
        
        crossed, direction = trajectory_crosses_line(trajectory, line)
        
        assert crossed is False
        assert direction is None


class TestLineCounter:
    """Test LineCounter class."""
    
    @pytest.fixture
    def counter(self):
        """Create a counter instance."""
        # Horizontal line in middle of frame
        line = ((100, 400), (900, 400))
        return LineCounter(line=line, min_track_length=5, debounce_frames=10)
    
    @pytest.fixture
    def mock_track(self):
        """Create a mock track."""
        def _create_track(track_id, bbox, confidence=0.9):
            track = Track(track_id=track_id, bbox=np.array(bbox), confidence=confidence)
            return track
        return _create_track
    
    def test_counter_initialization(self, counter):
        """Test counter initialization."""
        assert counter.count_in == 0
        assert counter.count_out == 0
        assert counter.min_track_length == 5
        assert counter.debounce_frames == 10
    
    def test_single_crossing_in(self, counter, mock_track):
        """Test single track crossing IN."""
        track = mock_track(1, [400, 350, 450, 450])  # bbox above line
        
        # Simulate track moving down (crossing IN)
        for y in range(350, 450, 10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        # Should count as IN
        assert counter.count_in == 1
        assert counter.count_out == 0
    
    def test_single_crossing_out(self, counter, mock_track):
        """Test single track crossing OUT."""
        track = mock_track(2, [400, 450, 450, 550])  # bbox below line
        
        # Simulate track moving up (crossing OUT)
        for y in range(450, 350, -10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        # Should count as OUT
        assert counter.count_in == 0
        assert counter.count_out == 1
    
    def test_min_track_length_filter(self, counter, mock_track):
        """Test that short tracks are not counted."""
        track = mock_track(3, [400, 350, 450, 450])
        
        # Move track only a few frames
        for y in range(350, 400, 20):  # Only 3 frames
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        # Should not count (min_track_length = 5)
        assert counter.count_in == 0
        assert counter.count_out == 0
    
    def test_debounce_prevents_double_count(self, counter, mock_track):
        """Test debounce mechanism."""
        track = mock_track(4, [400, 350, 450, 450])
        
        # First crossing
        for y in range(350, 450, 10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        assert counter.count_in == 1
        
        # Try to cross again immediately (within debounce window)
        for y in range(450, 350, -10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        # Should still be 1 (debounced)
        assert counter.count_out == 0  # Debounce prevents this
    
    def test_multiple_tracks_simultaneously(self, counter, mock_track):
        """Test counting multiple tracks at once."""
        track1 = mock_track(5, [200, 350, 250, 450])
        track2 = mock_track(6, [600, 450, 650, 550])
        
        # Move both tracks
        for i in range(15):
            # Track 1: crossing IN
            y1 = 350 + i * 10
            track1.bbox = np.array([200, y1, 250, y1 + 100])
            
            # Track 2: crossing OUT
            y2 = 450 - i * 10
            track2.bbox = np.array([600, y2, 650, y2 + 100])
            
            counter.update([track1, track2])
        
        # Both should be counted
        assert counter.count_in == 1
        assert counter.count_out == 1
    
    def test_reset(self, counter):
        """Test counter reset."""
        counter.count_in = 10
        counter.count_out = 5
        
        counter.reset()
        
        assert counter.count_in == 0
        assert counter.count_out == 0
        assert len(counter.events) == 0
    
    def test_get_events(self, counter, mock_track):
        """Test event retrieval."""
        track = mock_track(7, [400, 350, 450, 450])
        
        # Create crossing
        for y in range(350, 450, 10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        events = counter.get_events()
        
        assert len(events) == 1
        assert events[0].track_id == 7
        assert events[0].direction == 'IN'


class TestEdgeCases:
    """Test edge cases."""
    
    def test_track_parallel_to_line(self):
        """Test track moving parallel to counting line."""
        line = ((0, 400), (1000, 400))
        counter = LineCounter(line=line)
        
        # Create track moving horizontally (parallel to line)
        track = Track(track_id=8, bbox=np.array([0, 300, 50, 350]), confidence=0.9)
        
        for x in range(0, 500, 50):
            track.bbox = np.array([x, 300, x + 50, 350])
            counter.update([track])
        
        # Should not count (never crosses)
        assert counter.count_in == 0
        assert counter.count_out == 0
    
    def test_track_touches_line_but_not_crosses(self):
        """Test track that touches but doesn't cross."""
        line = ((0, 400), (1000, 400))
        counter = LineCounter(line=line)
        
        track = Track(track_id=9, bbox=np.array([400, 300, 450, 400]), confidence=0.9)
        
        # Move towards line
        for y in range(300, 395, 10):
            track.bbox = np.array([400, y, 450, y + 100])
            counter.update([track])
        
        # Should not count (touches but doesn't cross)
        assert counter.count_in == 0
        assert counter.count_out == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
