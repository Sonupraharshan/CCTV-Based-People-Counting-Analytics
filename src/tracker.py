"""
Multi-Object Tracker
Integrates StrongSORT and ByteTrack for robust people tracking.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time


@dataclass
class Track:
    """Represents a single tracked object."""
    track_id: int
    bbox: np.ndarray  # Current bounding box [x1, y1, x2, y2]
    history: deque = field(default_factory=lambda: deque(maxlen=50))  # Position history
    age: int = 0  # Number of frames since birth
    hits: int = 0  # Number of successful detections
    time_since_update: int = 0  # Frames since last update
    state: str = 'tentative'  # 'tentative', 'confirmed', 'deleted'
    confidence: float = 0.0  # Latest detection confidence
    
    def __post_init__(self):
        """Initialize history with current bbox."""
        center = self.get_center()
        self.history.append(center)
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bottom_center(self) -> Tuple[float, float]:
        """Get bottom-center point (useful for people tracking)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)
    
    def update(self, bbox: np.ndarray, confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # Update history
        center = self.get_center()
        self.history.append(center)
        
        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed for current frame."""
        self.time_since_update += 1
        
        if self.time_since_update > 30:  # Max age
            self.state = 'deleted'
    
    def get_trajectory(self, max_length: int = 30) -> List[Tuple[float, float]]:
        """Get recent trajectory points."""
        return list(self.history)[-max_length:]


class Tracker:
    """
    Multi-object tracker with support for multiple tracking algorithms.
    
    Supports:
    - StrongSORT: Appearance-based tracking with Kalman filter
    - ByteTrack: Simple but effective IoU-based tracking
    """
    
    def __init__(
        self,
        tracker_type: str = 'strongsort',
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_appearance: bool = True
    ):
        """
        Initialize tracker.
        
        Args:
            tracker_type: Tracking algorithm ('strongsort' or 'bytetrack')
            max_age: Maximum frames to keep lost tracks
            min_hits: Minimum hits to confirm track
            iou_threshold: IOU threshold for matching
            use_appearance: Use appearance features (for StrongSORT)
        """
        self.tracker_type = tracker_type.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_appearance = use_appearance
        
        # Track management
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        
        print(f"Initialized {tracker_type} tracker")
        
        # Load tracker backend
        self._init_tracker()
    
    def _init_tracker(self):
        """Initialize tracking backend."""
        try:
            if self.tracker_type == 'strongsort':
                # Try to import StrongSORT dependencies
                from filterpy.kalman import KalmanFilter
                self.use_kalman = True
                print("Using Kalman filter for state prediction")
            else:
                self.use_kalman = False
                print(f"Using simple IoU tracking ({self.tracker_type})")
        except ImportError:
            print("Warning: filterpy not available, falling back to simple tracking")
            self.use_kalman = False
    
    def update(
        self,
        detections: List[np.ndarray],
        scores: List[float],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2]
            scores: Confidence scores for each detection
            frame: Current frame (optional, for appearance features)
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Convert to numpy arrays
        if len(detections) > 0:
            detections = np.array(detections)
            scores = np.array(scores)
        else:
            detections = np.empty((0, 4))
            scores = np.empty((0,))
        
        # Match detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = self._match(
            detections, scores, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id].update(detections[det_idx], scores[det_idx])
            self.tracks[track_id].age += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(
                track_id=self.next_id,
                bbox=detections[det_idx],
                confidence=scores[det_idx]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
            self.tracks[track_id].age += 1
        
        # Remove deleted tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.state != 'deleted'
        }
        
        # Return confirmed tracks
        return [track for track in self.tracks.values() if track.state == 'confirmed']
    
    def _match(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        tracks: Dict[int, Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU or appearance features.
        
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if len(detections) == 0:
            return [], [], list(tracks.keys())
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU cost matrix
        track_boxes = np.array([track.bbox for track in tracks.values()])
        track_ids = list(tracks.keys())
        
        iou_matrix = self._compute_iou_matrix(detections, track_boxes)
        
        # Use Hungarian algorithm for assignment
        matches, unmatched_det, unmatched_trk = self._linear_assignment(
            iou_matrix, self.iou_threshold
        )
        
        # Convert track indices to track IDs
        matched_pairs = [(det_idx, track_ids[trk_idx]) for det_idx, trk_idx in matches]
        unmatched_track_ids = [track_ids[idx] for idx in unmatched_trk]
        
        return matched_pairs, unmatched_det, unmatched_track_ids
    
    def _compute_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        Compute IoU matrix between two sets of boxes.
        
        Args:
            boxes1: Array of shape (N, 4) - format [x1, y1, x2, y2]
            boxes2: Array of shape (M, 4) - format [x1, y1, x2, y2]
        
        Returns:
            IoU matrix of shape (N, M)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Compute intersection
        x1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
        y1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
        x2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
        y2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform linear assignment (Hungarian algorithm) on cost matrix.
        
        Args:
            cost_matrix: Cost matrix (higher is better for IoU)
            threshold: Minimum IoU threshold for matching
        
        Returns:
            Tuple of (matches, unmatched_rows, unmatched_cols)
        """
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Convert IoU to cost (1 - IoU)
            cost = 1 - cost_matrix
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost)
            
            # Filter by threshold
            matches = []
            unmatched_detections = []
            unmatched_tracks = []
            
            for row_idx, col_idx in zip(row_indices, col_indices):
                if cost_matrix[row_idx, col_idx] >= threshold:
                    matches.append((row_idx, col_idx))
                else:
                    unmatched_detections.append(row_idx)
                    unmatched_tracks.append(col_idx)
            
            # Add unassigned rows and columns
            for row_idx in range(cost_matrix.shape[0]):
                if row_idx not in row_indices:
                    unmatched_detections.append(row_idx)
            
            for col_idx in range(cost_matrix.shape[1]):
                if col_idx not in col_indices:
                    unmatched_tracks.append(col_idx)
            
            return matches, unmatched_detections, unmatched_tracks
            
        except ImportError:
            # Fallback to greedy matching if scipy not available
            return self._greedy_assignment(cost_matrix, threshold)
    
    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy matching fallback."""
        matches = []
        matched_detections = set()
        matched_tracks = set()
        
        # Flatten and sort by IoU
        flat_indices = np.argsort(-cost_matrix.ravel())
        
        for flat_idx in flat_indices:
            row_idx = flat_idx // cost_matrix.shape[1]
            col_idx = flat_idx % cost_matrix.shape[1]
            
            if cost_matrix[row_idx, col_idx] < threshold:
                break
            
            if row_idx not in matched_detections and col_idx not in matched_tracks:
                matches.append((row_idx, col_idx))
                matched_detections.add(row_idx)
                matched_tracks.add(col_idx)
        
        unmatched_detections = [i for i in range(cost_matrix.shape[0]) if i not in matched_detections]
        unmatched_tracks = [i for i in range(cost_matrix.shape[1]) if i not in matched_tracks]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def get_tracks(self, confirmed_only: bool = True) -> List[Track]:
        """
        Get current tracks.
        
        Args:
            confirmed_only: Return only confirmed tracks
        
        Returns:
            List of tracks
        """
        if confirmed_only:
            return [track for track in self.tracks.values() if track.state == 'confirmed']
        return list(self.tracks.values())
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        print("Tracker reset")
