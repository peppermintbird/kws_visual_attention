"""
Example usage of KWSModulator in the full visual attention pipeline.

This demonstrates:
1. Initializing the KWSModulator
2. Processing KWS output (class_id, confidence)
3. Applying spatial bias to saliency maps
4. Handling edge cases and visualization
"""

import numpy as np
from kws_mod_new import KWSModulator, QUADRANT_LAYOUT, KEYWORD_TO_DIRECTION
from helpers_kws_mod import (
    compute_quadrant_saliencies,
    extract_peak_quadrant,
    visualize_quadrants,
)


def example_basic_workflow():
    """
    Basic workflow: initialize, push command, apply bias.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Workflow")
    print("=" * 70)
    
    # Initialize modulator
    modulator = KWSModulator(threshold=0.5, alpha=1.6, verbose=True)
    
    # Simulate visual attention output: saliency map with peak in top-left quad
    H, W = 256, 256
    saliency_map = np.random.randint(20, 100, (H, W), dtype=np.uint8)
    saliency_map[20:60, 20:60] = 200  # Brightest region in top-left
    salmax_coords = (40, 40)  # Peak at (40, 40) = top-left quadrant
    
    print(f"\nInitial saliency: shape={saliency_map.shape}, max={saliency_map.max()}")
    print(f"Peak location: {salmax_coords}")
    
    # Simulate KWS output: "right" command with high confidence
    class_id = 1  # "right"
    confidence = 0.85
    
    print(f"\nReceived KWS output: class_id={class_id} ('{KEYWORD_TO_DIRECTION[class_id]}'), confidence={confidence}")
    
    # Push command (will pass threshold gate)
    accepted = modulator.push(class_id, confidence)
    print(f"Command accepted: {accepted}")
    
    # Apply bias to saliency map
    saliency_boosted = modulator.apply(saliency_map, salmax_coords)
    
    print(f"\nBoosted saliency: shape={saliency_boosted.shape}, max={saliency_boosted.max()}")
    
    # Analyze per-quadrant saliencies
    quad_stats_before = compute_quadrant_saliencies(saliency_map)
    quad_stats_after = compute_quadrant_saliencies(saliency_boosted)
    
    print("\nPer-quadrant saliency before:")
    for quad_id, stats in quad_stats_before.items():
        print(f"  Quad {quad_id} ({QUADRANT_LAYOUT[quad_id]:12s}): mean={stats['mean']:.1f}, max={stats['max']:.1f}")
    
    print("\nPer-quadrant saliency after:")
    for quad_id, stats in quad_stats_after.items():
        print(f"  Quad {quad_id} ({QUADRANT_LAYOUT[quad_id]:12s}): mean={stats['mean']:.1f}, max={stats['max']:.1f}")


def example_threshold_rejection():
    """
    Demonstrate threshold gate: low confidence is rejected.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Threshold Gate (Low Confidence → Rejected)")
    print("=" * 70)
    
    modulator = KWSModulator(threshold=0.7, alpha=1.6, verbose=True)
    
    H, W = 256, 256
    saliency_map = np.ones((H, W), dtype=np.uint8) * 100
    saliency_map[128:150, 128:150] = 200  # Peak in bottom-right
    salmax_coords = (140, 140)
    
    print(f"\nModulator threshold = {modulator.threshold}")
    
    # Low confidence command
    class_id = 0  # "left"
    confidence = 0.45
    print(f"\nAttempting command: class_id={class_id} ('{KEYWORD_TO_DIRECTION[class_id]}'), confidence={confidence}")
    
    accepted = modulator.push(class_id, confidence)
    print(f"Accepted: {accepted}")
    
    # Apply (will return unchanged map since command was rejected)
    saliency_unchanged = modulator.apply(saliency_map, salmax_coords)
    print(f"\nSaliency map changed: {not np.array_equal(saliency_map, saliency_unchanged)}")


def example_boundary_edge_case():
    """
    Demonstrate edge case: trying to move outside quadrant boundary.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Boundary Edge Case (Cannot Move Outside Boundary)")
    print("=" * 70)
    
    modulator = KWSModulator(threshold=0.5, alpha=1.6, verbose=True)
    
    H, W = 256, 256
    saliency_map = np.ones((H, W), dtype=np.uint8) * 100
    saliency_map[10:40, 10:40] = 220  # Peak in top-left quadrant (quad 0)
    salmax_coords = (25, 25)
    
    print(f"\nPeak in top-left quadrant (0)")
    
    # Try to move "up" from top-left (no quadrant above)
    class_id = 2  # "up"
    confidence = 0.8
    print(f"\nAttempting '{KEYWORD_TO_DIRECTION[class_id]}' from top-left...")
    
    modulator.push(class_id, confidence)
    saliency_result = modulator.apply(saliency_map, salmax_coords)
    
    print(f"\nSaliency changed: {not np.array_equal(saliency_map, saliency_result)}")
    print("(No change because 'up' is not available from top-left)")


def example_sequential_commands():
    """
    Demonstrate sequential commands: multiple push/apply cycles.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Sequential Commands (Multiple Frames)")
    print("=" * 70)
    
    modulator = KWSModulator(threshold=0.6, alpha=1.5, verbose=False)
    
    H, W = 256, 256
    saliency_map = np.ones((H, W), dtype=np.uint8) * 100
    
    # Simulate several frames with different commands
    commands = [
        (1, 0.8, (128, 128), "right from center"),
        (0, 0.75, (128, 200), "left from right-center"),
        (2, 0.85, (40, 40), "up from top-left"),
        (3, 0.5, (100, 100), "down (below threshold, rejected)"),
    ]
    
    for i, (class_id, confidence, peak, description) in enumerate(commands):
        print(f"\nFrame {i+1}: {description}")
        print(f"  Command: class_id={class_id} ('{KEYWORD_TO_DIRECTION[class_id]}'), confidence={confidence:.2f}")
        print(f"  Peak: {peak}")
        
        # Simulate some saliency variation per frame
        saliency_map[peak[0]-20:peak[0]+20, peak[1]-20:peak[1]+20] = 200
        
        accepted = modulator.push(class_id, confidence)
        saliency_boosted = modulator.apply(saliency_map, peak)
        
        print(f"  Accepted: {accepted}, Boosted: {saliency_boosted.max() > saliency_map.max()}")
        
        # Reset for next frame
        saliency_map[:] = 100


def example_visualization():
    """
    Create visualization with quadrant overlay.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Visualization with Quadrants")
    print("=" * 70)
    
    H, W = 256, 256
    saliency_map = np.random.randint(50, 150, (H, W), dtype=np.uint8)
    saliency_map[20:80, 20:80] = 220  # Bright in TL
    saliency_map[100:160, 200:256] = 210  # Bright in BR
    
    peak = (50, 50)
    
    # Create visualization
    vis = visualize_quadrants(saliency_map, peak_location=peak, color_map="jet")
    
    print(f"Visualization shape: {vis.shape}")
    print(f"Peak marked at: {peak}")
    print("Quadrant boundaries overlaid with white lines")
    print("Quadrant IDs (0,1,2,3) labeled at corners")
    
    # Show which quadrant contains the peak
    peak_quad = extract_peak_quadrant(peak, H, W)
    print(f"Peak is in quadrant {peak_quad} ({QUADRANT_LAYOUT[peak_quad]})")


if __name__ == "__main__":
    example_basic_workflow()
    example_threshold_rejection()
    example_boundary_edge_case()
    example_sequential_commands()
    example_visualization()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
