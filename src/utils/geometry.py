"""
Geometric Utilities for Line-Crossing Detection
Provides functions for line-segment intersection and point-to-line calculations.
"""

import numpy as np
from typing import Tuple, Optional


def line_intersection(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point of two line segments.
    
    Args:
        line1: First line segment ((x1, y1), (x2, y2))
        line2: Second line segment ((x3, y3), (x4, y4))
    
    Returns:
        Intersection point (x, y) if segments intersect, None otherwise
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Lines are parallel if denominator is zero
    if abs(denom) < 1e-10:
        return None
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None


def point_to_line_distance(
    point: Tuple[float, float],
    line: Tuple[Tuple[float, float], Tuple[float, float]]
) -> float:
    """
    Calculate the perpendicular distance from a point to a line.
    
    Args:
        point: Point coordinates (x, y)
        line: Line segment ((x1, y1), (x2, y2))
    
    Returns:
        Perpendicular distance from point to line
    """
    (x0, y0) = point
    (x1, y1), (x2, y2) = line
    
    # Calculate distance using cross product formula
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if denominator < 1e-10:
        # Line segment is actually a point
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    return numerator / denominator


def point_side_of_line(
    point: Tuple[float, float],
    line: Tuple[Tuple[float, float], Tuple[float, float]]
) -> int:
    """
    Determine which side of a line a point is on.
    
    Args:
        point: Point coordinates (x, y)
        line: Line segment ((x1, y1), (x2, y2))
    
    Returns:
        1 if point is on the right side (or above for horizontal line)
        -1 if point is on the left side (or below for horizontal line)
        0 if point is on the line
    """
    (x0, y0) = point
    (x1, y1), (x2, y2) = line
    
    # Calculate cross product
    cross_product = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
    
    if abs(cross_product) < 1e-10:
        return 0
    elif cross_product > 0:
        return 1
    else:
        return -1


def trajectory_crosses_line(
    trajectory: list,
    line: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[bool, Optional[str]]:
    """
    Check if a trajectory (sequence of points) crosses a line.
    
    Args:
        trajectory: List of (x, y) points representing object trajectory
        line: Counting line ((x1, y1), (x2, y2))
    
    Returns:
        (crossed, direction) where:
            crossed: True if trajectory crossed the line
            direction: 'IN' or 'OUT' based on crossing direction, None if not crossed
    """
    if len(trajectory) < 2:
        return False, None
    
    # Check consecutive segments of trajectory for intersection
    for i in range(len(trajectory) - 1):
        segment = (trajectory[i], trajectory[i + 1])
        intersection = line_intersection(segment, line)
        
        if intersection is not None:
            # Determine direction based on which side points are on
            side_before = point_side_of_line(trajectory[i], line)
            side_after = point_side_of_line(trajectory[i + 1], line)
            
            if side_before != side_after:
                # Define IN as moving from negative to positive side
                direction = 'IN' if side_after > side_before else 'OUT'
                return True, direction
    
    return False, None


def calculate_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2) or (x, y, w, h)
    
    Returns:
        Center point (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    
    # Assume format is either (x1, y1, x2, y2) or (x, y, w, h)
    # If x2 < x1, it's likely (x, y, w, h) format
    if x2 < x1:
        # (x, y, w, h) format
        cx = x1 + x2 / 2
        cy = y1 + y2 / 2
    else:
        # (x1, y1, x2, y2) format
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
    
    return (cx, cy)


def bbox_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate the bottom-center point of a bounding box.
    Useful for people tracking as feet are typically the reference point.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Bottom-center point (cx, y2)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    return (cx, y2)


def angle_between_vectors(
    v1: Tuple[float, float],
    v2: Tuple[float, float]
) -> float:
    """
    Calculate the angle between two vectors in degrees.
    
    Args:
        v1: First vector (dx1, dy1)
        v2: Second vector (dx2, dy2)
    
    Returns:
        Angle in degrees [0, 180]
    """
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0
    
    # Calculate angle using arccos
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg
