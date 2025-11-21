"""
Line-Crossing Counter with Anti-Double-Count Logic
Robust people counting with direction detection (IN/OUT).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import time
from dataclasses import dataclass, field

from src.utils.geometry import (
    line_intersection,
    point_side_of_line,
    trajectory_crosses_line,
    bbox_bottom_center
)
from src.tracker import Track


@dataclass
class CountEvent:
    """Represents a single counting event."""
    track_id: int
    direction: str  # 'IN' or 'OUT'
    timestamp: float
    frame_number: int
    confidence: float
    crossing_point: Tuple[float, float]


class LineCounter:
    """
    Robust line-crossing counter with anti-double-count mechanisms.
    
    Features:
    - Direction detection (IN/OUT)
    - Per-track state management
    - Minimum track length requirement
    - Debounce logic to prevent double counting
    - Re-identification heuristics
    """
    
    def __init__(
        self,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
        min_track_length: int = 10,
        debounce_frames: int = 30,
        use_bottom_center: bool = True
    ):
        """
        Initialize line counter.
        
        Args:
            line: Counting line ((x1, y1), (x2, y2))
            min_track_length: Minimum frames before allowing count
            debounce_frames: Minimum frames between counts for same track
            use_bottom_center: Use bottom-center of bbox (recommended for people)
        """
        self.line = line
        self.min_track_length = min_track_length
        self.debounce_frames = debounce_frames
        self.use_bottom_center = use_bottom_center
        
        # Counting stats
        self.count_in = 0
        self.count_out = 0
        
        # Track state management
        self.track_states: Dict[int, dict] = {}  # track_id -> state dict
        self.counted_tracks: set = set()  # track_ids that have been counted
        
        # Event history
        self.events: List[CountEvent] = []
        
        # Current frame number
        self.frame_number = 0
        
        print(f"Initialized LineCounter: line={line}, min_length={min_track_length}")
    
    def update_line(self, line: Tuple[Tuple[float, float], Tuple[float, float]]):
        """Update the counting line."""
        self.line = line
        print(f"Updated counting line: {line}")
    
    def update(self, tracks: List[Track]) -> List[CountEvent]:
        """
        Update counter with current frame tracks.
        
        Args:
            tracks: List of active tracks from tracker
        
        Returns:
            List of new counting events in this frame
        """
        self.frame_number += 1
        new_events = []
        
        for track in tracks:
            track_id = track.track_id
            
            # Initialize track state if new
            if track_id not in self.track_states:
                self.track_states[track_id] = {
                    'last_position': None,
                    'last_side': None,
                    'frames_tracked': 0,
                    'last_counted_frame': -999,
                    'trajectory': []
                }
            
            state = self.track_states[track_id]
            state['frames_tracked'] += 1
            
            # Get reference point (center or bottom-center)
            if self.use_bottom_center:
                current_pos = bbox_bottom_center(track.bbox)
            else:
                current_pos = track.get_center()
            
            # Update trajectory
            state['trajectory'].append(current_pos)
            if len(state['trajectory']) > 50:  # Keep last 50 points
                state['trajectory'] = state['trajectory'][-50:]
            
            # Get current side of line
            current_side = point_side_of_line(current_pos, self.line)
            
            # Check for line crossing
            if state['last_position'] is not None and state['last_side'] is not None:
                # Check if track has sufficient history
                if state['frames_tracked'] < self.min_track_length:
                    # Not enough history, skip
                    state['last_position'] = current_pos
                    state['last_side'] = current_side
                    continue
                
                # Check if crossed line (sides are different)
                if current_side != 0 and state['last_side'] != 0 and current_side != state['last_side']:
                    # Check debounce (prevent multiple counts for same track)
                    frames_since_count = self.frame_number - state['last_counted_frame']
                    
                    if frames_since_count >= self.debounce_frames:
                        # Valid crossing detected!
                        # Determine direction
                        if current_side > state['last_side']:
                            direction = 'IN'
                            self.count_in += 1
                        else:
                            direction = 'OUT'
                            self.count_out += 1
                        
                        # Find crossing point
                        segment = (state['last_position'], current_pos)
                        crossing_point = line_intersection(segment, self.line)
                        if crossing_point is None:
                            crossing_point = current_pos
                        
                        # Create event
                        event = CountEvent(
                            track_id=track_id,
                            direction=direction,
                            timestamp=time.time(),
                            frame_number=self.frame_number,
                            confidence=track.confidence,
                            crossing_point=crossing_point
                        )
                        
                        self.events.append(event)
                        new_events.append(event)
                        self.counted_tracks.add(track_id)
                        
                        # Update state
                        state['last_counted_frame'] = self.frame_number
                        
                        print(f"🚶 Count event: Track {track_id} crossed {direction} "
                              f"(Total: IN={self.count_in}, OUT={self.count_out})")
            
            # Update state for next frame
            state['last_position'] = current_pos
            state['last_side'] = current_side
        
        return new_events
    
    def get_total_count(self) -> int:
        """Get total count (IN + OUT)."""
        return self.count_in + self.count_out
    
    def get_net_count(self) -> int:
        """Get net count (IN - OUT)."""
        return self.count_in - self.count_out
    
    def get_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        direction: Optional[str] = None
    ) -> List[CountEvent]:
        """
        Get counting events with optional filtering.
        
        Args:
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            direction: Filter by direction ('IN' or 'OUT')
        
        Returns:
            Filtered list of events
        """
        filtered = self.events
        
        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        if direction is not None:
            filtered = [e for e in filtered if e.direction == direction]
        
        return filtered
    
    def reset(self):
        """Reset counter statistics."""
        self.count_in = 0
        self.count_out = 0
        self.track_states = {}
        self.counted_tracks = set()
        self.events = []
        self.frame_number = 0
        print("Counter reset")
    
    def export_events_dict(self) -> List[dict]:
        """Export events as list of dictionaries for serialization."""
        return [
            {
                'track_id': e.track_id,
                'direction': e.direction,
                'timestamp': e.timestamp,
                'frame_number': e.frame_number,
                'confidence': e.confidence,
                'crossing_point': e.crossing_point
            }
            for e in self.events
        ]


class ZoneCounter:
    """
    Alternative counting method using zone-based approach.
    Counts people entering/exiting defined zones.
    """
    
    def __init__(
        self,
        zone_polygon: List[Tuple[float, float]],
        min_track_length: int = 10
    ):
        """
        Initialize zone counter.
        
        Args:
            zone_polygon: List of (x, y) points defining zone boundary
            min_track_length: Minimum frames before allowing count
        """
        self.zone_polygon = np.array(zone_polygon)
        self.min_track_length = min_track_length
        
        self.count_entered = 0
        self.count_exited = 0
        
        self.track_states: Dict[int, dict] = {}
        self.events: List[CountEvent] = []
        self.frame_number = 0
    
    def _point_in_polygon(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside polygon using ray casting."""
        x, y = point
        polygon = self.zone_polygon
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def update(self, tracks: List[Track]) -> List[CountEvent]:
        """Update zone counter with current tracks."""
        self.frame_number += 1
        new_events = []
        
        for track in tracks:
            track_id = track.track_id
            current_center = track.get_center()
            current_inside = self._point_in_polygon(current_center)
            
            # Initialize state
            if track_id not in self.track_states:
                self.track_states[track_id] = {
                    'was_inside': current_inside,
                    'frames_tracked': 0
                }
                continue
            
            state = self.track_states[track_id]
            state['frames_tracked'] += 1
            
            # Check for zone crossing
            if state['frames_tracked'] >= self.min_track_length:
                if current_inside and not state['was_inside']:
                    # Entered zone
                    self.count_entered += 1
                    event = CountEvent(
                        track_id=track_id,
                        direction='ENTERED',
                        timestamp=time.time(),
                        frame_number=self.frame_number,
                        confidence=track.confidence,
                        crossing_point=current_center
                    )
                    self.events.append(event)
                    new_events.append(event)
                
                elif not current_inside and state['was_inside']:
                    # Exited zone
                    self.count_exited += 1
                    event = CountEvent(
                        track_id=track_id,
                        direction='EXITED',
                        timestamp=time.time(),
                        frame_number=self.frame_number,
                        confidence=track.confidence,
                        crossing_point=current_center
                    )
                    self.events.append(event)
                    new_events.append(event)
            
            state['was_inside'] = current_inside
        
        return new_events
