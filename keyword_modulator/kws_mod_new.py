"""
Keyword spotting (KWS) modulation with attention bias

Workflow:
    1. run_attention() -> saliency_map, salmax_coods (ok)
    2. KWS spotter fires -> class_id, confidence_score (needs sorting out the code)
    3. push(class_id, confidence_score) -> threshold + bias computation (writing code)
    4. apply(saliency_map, salmax_coords) -> spatial bias applied to saliency map (next step)
        - shift attention first: determine currecnt quadrant from salmax_coords | where i am
        - neighbor lookup: find the target quadrant based on the keyword and current postion | where i want to go
        - apply the bias: boost target quadrant pixels by factor B | how much i want to go there
        - normalization: --- 


Inputs:
  - class_id: int, keyword class (1=left, 2=right, 3=up, 4=down)
  - confidence_score: float [0, 1], network confidence score
  - saliency_map: np.ndarray (H, W), from run_attention()
  - salmax_coords: (row, col), peak attention location (downsampled)

Outputs:
  - saliency_map: np.ndarray (H, W), boosted saliency with spatial bias applied
"""

import numpy as np


# QUADRANT LAYOUT AND NEIGHBOR LOOKUP

QUADRANT_LAYOUT = {
    1: "top_left",
    2: "top_right",
    3: "bottom_left",
    4: "bottom_right",
}

NEIGHBOR_GRID = {
    1: {"left": None, "right": 2, "up": None, "down": 3},     # Top-left
    2: {"left": 1,    "right": None, "up": None, "down": 4},  # Top-down
    3: {"left": None, "right": 4, "up": 1, "down": None},     # Bottom-up
    4: {"left": 3,    "right": None, "up": 2, "down": None},  # Bottom-right
}

KEYWORD_TO_DIRECTION = {
    1: "left",
    2: "right",
    3: "up",
    4: "down",
}

# KWS MODULATOR CLASS

class KWSModulator:
    """
    explanation
    """

    def __init__(
        self,
        threshold: float = 0.5, # arbritary for now
        alpha: float = 1.6,     # arbritary for now 
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            threshold: minimum confidence required to accept command [0, 1]
            alpha: boost multiplier, B = confidence * alpha | e.g. conf = 0.8 -> 0.8 * 1.6 = 1.28 -> 28% boost
            normalize: whether to normalize saliency after boost
            verbose: print debug info
        """
        self.threshold = threshold
        self.alpha = alpha
        self.normalize_saliency = normalize
        self.verbose = verbose
        
        # Persistent state for pending commands
        self._pending_command = None  # {"class_id": int, "B": float}

    def push(self, class_id: int, confidence: float) -> bool:
        """
        exp
        """
        # Below threshold - reject
        if confidence < self.threshold:
            if self.verbose:
                print(f"KWS command {class_id} rejected: confidence {confidence:.3f} < threshold {self.threshold}")
            return False

        # Above threshold - accept and compute bias
        B = confidence * self.alpha
        self._pending_command = {"class_id": class_id, "confidence": confidence, "B": B}
        
        if self.verbose:
            direction = KEYWORD_TO_DIRECTION.get(class_id, "unknown")
            print(f"KWS command '{direction}' ({class_id}) accepted: confidence {confidence:.3f}, bias B={B:.3f}")
        
        return True

    def apply():
