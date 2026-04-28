"""
Integration guide: How to use KWSModulator in your full pipeline.

This shows how to connect:
1. KWS (Keyword Spotter) GCNN outputs
2. Visual Attention SNN outputs
3. KWSModulator for spatial bias
4. Final motor command generation
"""

# ============================================================================
# PIPELINE STRUCTURE
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────┐
│                           FULL PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DVS Events                    Microphone Audio                         │
│       │                              │                                  │
│       ▼                              ▼                                  │
│   [Visual Attention SNN]         [KWS GCNN (FPGA)]                     │
│       │                              │                                  │
│       ├─ saliency_map              ├─ class_id (0-3)                   │
│       └─ salmax_coords (y,x)       └─ confidence ∈ [0,1]               │
│                                                                         │
│                          ▼▼▼▼▼▼▼▼▼▼▼▼                                   │
│                     [KWS MODULATOR]  ← YOU ARE HERE                    │
│                                                                         │
│       ┌──────────────────────────────────────┐                         │
│       │ 1. Threshold gate:                   │                         │
│       │    confidence >= T → accept          │                         │
│       │                                      │                         │
│       │ 2. Shift attention first:            │                         │
│       │    current_quad = extract(salmax)    │                         │
│       │    target_quad = NEIGHBOR[curr][cmd] │                         │
│       │                                      │                         │
│       │ 3. Apply spatial bias:               │                         │
│       │    saliency[target] *= (conf * α)   │                         │
│       │                                      │                         │
│       │ 4. Normalize: Σ saliency → 255      │                         │
│       └──────────────────────────────────────┘                         │
│                          ▼                                              │
│                  [biased_saliency]                                      │
│                          │                                              │
│       ┌──────────────────┴──────────────────┐                         │
│       │  Find new peak (argmax)             │                         │
│       │  → peak_location (y_new, x_new)     │                         │
│       └──────────────────┬──────────────────┘                         │
│                          ▼                                              │
│                  [Motor Controller]                                     │
│                  pan/tilt command → iCub eyes                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# EXAMPLE: INTEGRATION WITH YOUR EXISTING CODE
# ============================================================================

import numpy as np
from visual_attention.helpers_visual_att import run_attention, initialise_attention
from keyword_spotter.kws_spotter import KWSSpotter  # pseudo-import
from keyword_modulator.kws_mod_new import KWSModulator
from keyword_modulator.helpers_kws_mod import extract_peak_quadrant

def example_full_pipeline():
    """
    Complete pipeline integration.
    """
    
    # ====================================================================
    # INITIALIZATION (once at startup)
    # ====================================================================
    print("Initializing pipeline components...")
    
    # 1. Visual Attention config and model
    device = torch.device("cpu")
    config = Config()
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
    
    # 2. KWS Modulator
    kws_modulator = KWSModulator(
        threshold=0.6,          # Commands must have confidence >= 60%
        alpha=1.6,              # Boost factor
        normalize=True,
        verbose=True,
    )
    
    # 3. Motor controller (your existing code)
    # motor = iCubMotor(...)
    
    # ====================================================================
    # MAIN LOOP (runs continuously)
    # ====================================================================
    for frame_idx in range(total_frames):
        # Step 1: Process DVS events
        # - Accumulate events in temporal window
        window = accumulate_dvs_events(...)  # your code
        
        # Step 2: Run visual attention
        # - Produces saliency map and peak coordinates
        saliency_map, salmax_coords = run_attention(
            window, net_attention, device, resolution,
            config.ATTENTION_PARAMS['num_pyr']
        )
        
        # Step 3: Process audio and run KWS
        # - GCNN outputs class_id and confidence
        audio_frame = get_audio_frame(...)  # your microphone code
        class_id, confidence_scores = kws_spotter(audio_frame)
        peak_confidence = confidence_scores[class_id]
        
        # Step 4: Push KWS output to modulator
        # - Modulator stores pending command if confidence >= threshold
        modulator_accepted = kws_modulator.push(class_id, peak_confidence)
        
        # Step 5: Apply spatial bias to saliency
        # - Modulator applies boost to target quadrant
        saliency_biased = kws_modulator.apply(saliency_map, salmax_coords)
        
        # Step 6: Find new peak after modulation
        peak_y_biased, peak_x_biased = np.unravel_index(
            np.argmax(saliency_biased),
            saliency_biased.shape
        )
        peak_biased = (peak_y_biased, peak_x_biased)
        
        # Step 7: Scale to original resolution (if downsampled)
        peak_x_original = int(peak_x_biased * DOWNSAMPLE_FACTOR)
        peak_y_original = int(peak_y_biased * DOWNSAMPLE_FACTOR)
        
        # Step 8: Send motor command
        # - Pan/tilt to point at new peak location
        motor.pan_tilt(peak_x_original, peak_y_original)
        
        # Optional: Visualization/logging
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}:")
            print(f"  KWS: class={class_id}, conf={peak_confidence:.3f}, accepted={modulator_accepted}")
            print(f"  Saliency peak: {salmax_coords} → {peak_biased}")
            print(f"  Motor command: pan={peak_x_original}, tilt={peak_y_original}")


# ============================================================================
# KEY INTEGRATION POINTS
# ============================================================================

# 1. KWS OUTPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────
# Your GCNN produces:
#    output  : Tensor [1, T]           confidence over time
#    cls     : Tensor [1, num_cls, T]  class scores over time
#
# Extract peak (as per README):
#    t_max       = argmax(output[0])
#    confidence  = output[0, t_max]     ← use this
#    class_id    = argmax(cls[0, :, t_max])  ← use this
#
# Then call:
#    kws_modulator.push(class_id, confidence)


# 2. VISUAL ATTENTION OUTPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────
# run_attention() returns:
#    saliency_map    : np.ndarray (H, W)  downsampled resolution
#    salmax_coords   : (row, col)         peak location in downsampled space
#
# Pass directly to modulator:
#    saliency_biased = kws_modulator.apply(saliency_map, salmax_coords)


# 3. MODULATOR API
# ─────────────────────────────────────────────────────────────────────────
# Two-step interface:
#
#   (a) Push new command:
#       accepted = modulator.push(class_id, confidence)
#       → Stores command IF confidence >= threshold
#       → Returns True/False indicating acceptance
#
#   (b) Apply to saliency (every frame):
#       saliency_biased = modulator.apply(saliency_map, salmax_coords)
#       → Returns modified saliency if command is pending
#       → Returns unchanged copy if no pending command
#       → Command is consumed (one-shot)


# 4. QUADRANT SYSTEM
# ─────────────────────────────────────────────────────────────────────────
# Grid layout (quadrant IDs):
#
#    +-------+-------+
#    |  0    |  1    |  0 = top-left
#    |  TL   |  TR   |  1 = top-right
#    +-------+-------+  2 = bottom-left
#    |  2    |  3    |  3 = bottom-right
#    |  BL   |  BR   |
#    +-------+-------+
#
# Neighbor transitions (from current quadrant, what quad reaches a direction):
#
#    From TL(0):  left→None,  right→1,  up→None, down→2
#    From TR(1):  left→0,     right→None, up→None, down→3
#    From BL(2):  left→None,  right→3,  up→0,   down→None
#    From BR(3):  left→2,     right→None, up→1,   down→None
#
# Note: If you're at an edge and command points outside, it's rejected
# (returns unchanged saliency, no boost applied)


# 5. PARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────
# threshold (default 0.5):
#    - Lower = more lenient (more commands accepted)
#    - Higher = more strict (only high-confidence commands)
#    - Recommendation: 0.6-0.75 for structured words, adjust per class
#
# alpha (default 1.6):
#    - Factor by which to multiply confidence for boost
#    - B = confidence × alpha
#    - Higher = stronger spatial bias toward target
#    - Recommendation: 1.5-2.0, tune based on balance between KWS and saliency
#
# normalize (default True):
#    - Whether to rescale saliency after boost
#    - Keep True for consistent peak detection


# 6. DEBUGGING & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────
# Enable verbose mode to see detailed logs:
#
#    modulator = KWSModulator(verbose=True)
#
# Outputs like:
#    [KWS] ✓ Command 'right' accepted: confidence 0.850, bias B=1.360
#    [KWS] Current attention at quad 0 (top_left), peak=(40, 40)
#    [KWS] Target quad: 1 (top_right)
#    [KWS] Saliency boost applied: 200.0 → 254.0
#
# Use helper functions for analysis:
#
#    from helpers_kws_mod import (
#        compute_quadrant_saliencies,
#        extract_peak_quadrant,
#        visualize_quadrants,
#    )
#
#    quad_stats = compute_quadrant_saliencies(saliency_map)
#    peak_quad = extract_peak_quadrant(salmax_coords, H, W)
#    vis = visualize_quadrants(saliency_map, peak_location=peak)


print(__doc__)
