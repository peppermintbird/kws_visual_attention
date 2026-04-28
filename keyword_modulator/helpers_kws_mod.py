"""
Helper utilities for keyword spotting modulation.

Functions:
  - compute_quadrant_saliencies: extract per-quadrant saliency scores
  - compute_attention_bias: generate spatial bias from command
  - visualize_quadrants: overlay quadrant boundaries on saliency
"""

import numpy as np
import cv2


def compute_quadrant_saliencies(saliency_map: np.ndarray) -> dict:
    """
    Compute mean and max saliency for each quadrant.
    
    Args:
        saliency_map: np.ndarray (H, W), saliency values
        
    Returns:
        dict: {quad_id: {"mean": float, "max": float, "sum": float}, ...}
    """
    H, W = saliency_map.shape[:2]
    y_split = H // 2
    x_split = W // 2
    
    quadrant_bounds = {
        0: (0, y_split, 0, x_split),        # TL
        1: (0, y_split, x_split, W),        # TR
        2: (y_split, H, 0, x_split),        # BL
        3: (y_split, H, x_split, W),        # BR
    }
    
    results = {}
    for quad_id, (y_start, y_end, x_start, x_end) in quadrant_bounds.items():
        quad_saliency = saliency_map[y_start:y_end, x_start:x_end]
        results[quad_id] = {
            "mean": np.mean(quad_saliency),
            "max": np.max(quad_saliency),
            "sum": np.sum(quad_saliency),
        }
    
    return results


def extract_peak_quadrant(salmax_coords: tuple, H: int, W: int) -> int:
    """
    Determine which quadrant contains the peak attention.
    
    Args:
        salmax_coords: (row, col), peak location
        H: height of saliency map
        W: width of saliency map
        
    Returns:
        int: quadrant ID (0=TL, 1=TR, 2=BL, 3=BR)
    """
    peak_y = int(salmax_coords[0])
    peak_x = int(salmax_coords[1])
    
    # Clamp to valid range
    peak_y = np.clip(peak_y, 0, H - 1)
    peak_x = np.clip(peak_x, 0, W - 1)
    
    y_split = H // 2
    x_split = W // 2
    
    col = 1 if peak_x >= x_split else 0
    row = 1 if peak_y >= y_split else 0
    quad = row * 2 + col
    
    return quad


def visualize_quadrants(
    saliency_map: np.ndarray,
    peak_location: tuple = None,
    color_map: str = "jet",
) -> np.ndarray:
    """
    Create visualization with quadrant boundaries overlaid.
    
    Args:
        saliency_map: np.ndarray (H, W), saliency to visualize
        peak_location: (row, col) optional peak to mark
        color_map: cv2 colormap name ("jet", "viridis", etc.)
        
    Returns:
        np.ndarray: (H, W, 3) RGB visualization
    """
    H, W = saliency_map.shape[:2]
    y_split = H // 2
    x_split = W // 2
    
    # Convert to uint8 and apply color map
    sal_uint8 = np.clip(saliency_map, 0, 255).astype(np.uint8)
    if color_map == "jet":
        sal_colored = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
    else:
        sal_colored = cv2.cvtColor(cv2.applyColorMap(sal_uint8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    
    # Draw quadrant boundaries (white lines, thickness=2)
    cv2.line(sal_colored, (x_split, 0), (x_split, H), (255, 255, 255), 2)
    cv2.line(sal_colored, (0, y_split), (W, y_split), (255, 255, 255), 2)
    
    # Mark quadrant IDs
    quad_labels = [(20, 25, "0"), (W-20, 25, "1"), (20, H-5, "2"), (W-20, H-5, "3")]
    for x, y, label in quad_labels:
        cv2.putText(sal_colored, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Mark peak if provided
    if peak_location is not None:
        peak_y = int(peak_location[0])
        peak_x = int(peak_location[1])
        peak_y = np.clip(peak_y, 0, H - 1)
        peak_x = np.clip(peak_x, 0, W - 1)
        cv2.circle(sal_colored, (peak_x, peak_y), 10, (0, 255, 255), 3)  # Cyan circle
    
    return sal_colored


def get_direction_vector(direction: str) -> tuple:
    """
    Get 2D movement vector for a direction keyword.
    
    Args:
        direction: str, one of {"left", "right", "up", "down"}
        
    Returns:
        tuple: (dy, dx) displacement vector
    """
    direction_map = {
        "left": (0, -1),
        "right": (0, 1),
        "up": (-1, 0),
        "down": (1, 0),
    }
    return direction_map.get(direction, (0, 0))
