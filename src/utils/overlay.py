"""
Visualization and Overlay Utilities
Drawing functions for detections, tracks, and counting overlays.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.tracker import Track
from src.counter import LineCounter, CountEvent


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection bounding boxes on frame.
    
    Args:
        frame: Input frame
        boxes: Bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores
        class_ids: Class IDs
        color: Box color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Frame with drawn boxes
    """
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"Person {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    draw_trajectory: bool = True,
    trajectory_length: int = 30,
    color_map: Optional[dict] = None
) -> np.ndarray:
    """
    Draw tracks with IDs and trajectories.
    
    Args:
        frame: Input frame
        tracks: List of tracks
        draw_trajectory: Whether to draw trajectory tails
        trajectory_length: Max trajectory points to draw
        color_map: Optional dict mapping track_id to color
    
    Returns:
        Frame with drawn tracks
    """
    for track in tracks:
        # Get color for track
        if color_map and track.track_id in color_map:
            color = color_map[track.track_id]
        else:
            # Generate consistent color based on track ID
            np.random.seed(track.track_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw bounding box
        x1, y1, x2, y2 = track.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"ID: {track.track_id}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw trajectory
        if draw_trajectory:
            trajectory = track.get_trajectory(trajectory_length)
            if len(trajectory) > 1:
                pts = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
                
                # Draw points
                for pt in pts[-5:]:  # Last 5 points
                    cv2.circle(frame, tuple(pt), 3, color, -1)
    
    return frame


def draw_counting_line(
    frame: np.ndarray,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3,
    draw_arrow: bool = True
) -> np.ndarray:
    """
    Draw counting line on frame.
    
    Args:
        frame: Input frame
        line: Line ((x1, y1), (x2, y2))
        color: Line color (B, G, R)
        thickness: Line thickness
        draw_arrow: Draw direction arrow
    
    Returns:
        Frame with drawn line
    """
    (x1, y1), (x2, y2) = line
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    
    # Draw line
    cv2.line(frame, p1, p2, color, thickness)
    
    # Draw endpoints
    cv2.circle(frame, p1, 8, color, -1)
    cv2.circle(frame, p2, 8, color, -1)
    
    # Draw direction arrow (perpendicular to line)
    if draw_arrow:
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize and rotate 90 degrees
            perp_dx = -dy / length * 30
            perp_dy = dx / length * 30
            
            # Draw arrow
            arrow_end = (int(mid_x + perp_dx), int(mid_y + perp_dy))
            cv2.arrowedLine(frame, (int(mid_x), int(mid_y)), arrow_end, color, 2, tipLength=0.3)
            
            # Add "IN" label
            cv2.putText(frame, "IN", arrow_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def draw_count_overlay(
    frame: np.ndarray,
    count_in: int,
    count_out: int,
    position: str = 'top-right',
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Draw count overlay on frame.
    
    Args:
        frame: Input frame
        count_in: IN count
        count_out: OUT count
        position: Position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        bg_color: Background color
        text_color: Text color
    
    Returns:
        Frame with count overlay
    """
    h, w = frame.shape[:2]
    
    # Prepare text
    in_text = f"IN: {count_in}"
    out_text = f"OUT: {count_out}"
    net_text = f"NET: {count_in - count_out}"
    
    # Calculate text sizes
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    (in_w, in_h), _ = cv2.getTextSize(in_text, font, font_scale, font_thickness)
    (out_w, out_h), _ = cv2.getTextSize(out_text, font, font_scale, font_thickness)
    (net_w, net_h), _ = cv2.getTextSize(net_text, font, font_scale, font_thickness)
    
    max_w = max(in_w, out_w, net_w)
    total_h = in_h + out_h + net_h + 60
    
    # Determine position
    margin = 20
    if position == 'top-left':
        x, y = margin, margin
    elif position == 'top-right':
        x, y = w - max_w - margin * 2, margin
    elif position == 'bottom-left':
        x, y = margin, h - total_h - margin
    else:  # bottom-right
        x, y = w - max_w - margin * 2, h - total_h - margin
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + max_w + margin * 2, y + total_h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + max_w + margin * 2, y + total_h), (255, 255, 255), 2)
    
    # Draw text
    offset_y = y + in_h + margin
    cv2.putText(frame, in_text, (x + margin, offset_y), font, font_scale, (0, 255, 0), font_thickness)
    
    offset_y += out_h + margin
    cv2.putText(frame, out_text, (x + margin, offset_y), font, font_scale, (0, 0, 255), font_thickness)
    
    offset_y += net_h + margin
    cv2.putText(frame, net_text, (x + margin, offset_y), font, font_scale, text_color, font_thickness)
    
    return frame


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input frame
        fps: Current FPS
        position: Text position (x, y)
        color: Text color
    
    Returns:
        Frame with FPS overlay
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return frame


def create_side_by_side(
    frame1: np.ndarray,
    frame2: np.ndarray,
    labels: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        labels: Optional (label1, label2) to display
    
    Returns:
        Combined frame
    """
    # Resize frames to same height
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    target_h = min(h1, h2)
    frame1 = cv2.resize(frame1, (int(w1 * target_h / h1), target_h))
    frame2 = cv2.resize(frame2, (int(w2 * target_h / h2), target_h))
    
    # Concatenate horizontally
    combined = np.hstack([frame1, frame2])
    
    # Add labels if provided
    if labels:
        cv2.putText(combined, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, labels[1], (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return combined
