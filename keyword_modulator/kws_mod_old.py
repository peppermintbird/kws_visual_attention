"""
Takes keyword spotter output and returns boosted saliency map.

Input (from keyword spotter):
-keyword     
-confidence  

Input (from existing run_attention() call):
- saliency_map   
- salmax_coords (downsampled)

Output:
- saliency_map (target quadrant boosted, rest intact)

Grid:
    top-left(0)    | top-right(1)
    -------------- + ------------
    bottom-left(2) | bottom-right(3)
"""

import numpy as np


NEIGHBOR = {
    0: {"left": None, "right": 1,    "up": None, "down": 2   },  # TL
    1: {"left": 0,    "right": None, "up": None, "down": 3   },  # TR
    2: {"left": None, "right": 3,    "up": 0,    "down": None},  # BL
    3: {"left": 2,    "right": None, "up": 1,    "down": None},  # BR
}


class KWSModulator: 

    def __init__(self, threshold: float = 0.5, alpha: float = 1.6):
        self.T        = threshold   # confidence must exceed this
        self.alpha    = alpha       # B = confidence * alpha
        self._pending = None        # set by push(), consumed by apply()

    # call when spotter fires 

    def push(self, keyword: str | None, confidence: float):
        """
        Threshold:
        - confidence < T  ->  reject, nothing stored
        - confidence >= T ->  W = confidence

        Bias;
        - B = W × alpha
        - stored as pending until next apply()
        """
        if keyword is None:
            return

        # Threshold
        if confidence < self.T:
            return                          # below threshold - ignore

        # Bias
        B = confidence * self.alpha
        self._pending = {"keyword": keyword, "B": B}

    # call every frame, right after run_attention() 

    def apply(self,
              saliency_map:  np.ndarray,
              salmax_coords: tuple) -> np.ndarray:
        """
        - neighbor lookup: Derive current_quad from salmax_coords, look up neighbor
        - apply bias: multiply target quadrant pixels by B, clip to [0, 255]

        If no keyword is pending, returns saliency_map unchanged (think of normalizing)

        Args
            saliency_map    np.ndarray (H, W)    from run_attention()
            salmax_coords   (row, col)           from run_attention()

        Returns
            saliency_map    np.ndarray (H, W)    boosted or unchanged
        """
        if self._pending is None:
            return saliency_map

        keyword = self._pending["keyword"]
        B       = self._pending["B"]
        self._pending = None            # one-shot — consume

        H, W = saliency_map.shape[:2]

        # current quadrant from live saliency peak
        peak_x = int(salmax_coords[1])
        peak_y = int(salmax_coords[0])
        col    = 1 if peak_x >= W / 2 else 0
        row    = 1 if peak_y >= H / 2 else 0
        current_quad = row * 2 + col

        # neighbor lookup
        target = NEIGHBOR[current_quad].get(keyword, None)
        if target is None:
            return saliency_map         # at edge — ignore

        # bias target quadrant
        col_t = target % 2
        row_t = target // 2
        x0    = col_t * (W // 2);  x1 = W if col_t == 1 else W // 2
        y0    = row_t * (H // 2);  y1 = H if row_t == 1 else H // 2

        result = saliency_map.copy().astype(np.float32)
        result[y0:y1, x0:x1] = np.clip(result[y0:y1, x0:x1] * B, 0, 255)
        return result.astype(saliency_map.dtype)