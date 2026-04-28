import numpy as np
from visual_attention.helpers_visual_att import initialise_attention, run_attention
import torch
import cv2
import sys
 
# config
class Config:
    ATTENTION_PARAMS = {
        'size_krn': 16,
        'r0': 7,
        'rho': 0.015,
        'theta': np.pi * 3 / 2,
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
        'thick': 12,
        'fltr_resize_perc': [2, 2],
        'offsetpxs': 0,
        'offset': (0, 0),
        'num_pyr': 6,
        'tau_mem': 0.3,
        'stride': 1,
        'out_ch': 1
    }
 
# kws mod parameters
THRESHOLD    = 0.7    # confidence threshold
BOOST_WEIGHT = 2.0    # saliency multiplier for target region
 
# hardcoded command for testing (replace with real KWS output later)
HARDCODED_WORD       = "right"
HARDCODED_CONFIDENCE = 0.85
 
 
def get_quadrant(x, y, mid_x, mid_y):
    h = "left"   if x < mid_x else "right"
    v = "top"    if y < mid_y else "bottom"
    return v, h
 
 
def kws_modulate(saliency_map, salmax_coords, word, confidence):
    """
    saliency_map   : 2D numpy array (H, W), values 0–255
    salmax_coords  : (row, col) = (y, x) from np.unravel_index
    word           : "left" | "right" | "up" | "down"
    confidence     : float
 
    Returns:
        boosted_map  : 2D numpy array, modulated saliency
        new_coords   : (row, col) of new peak
        accepted     : bool
    """
    H, W   = saliency_map.shape
    mid_x  = W // 2
    mid_y  = H // 2
 
    # salmax_coords is (row, col) -> (y, x)
    y, x   = salmax_coords
    v_quad, h_quad = get_quadrant(x, y, mid_x, mid_y)
 
    # threshold check
    if confidence < THRESHOLD:
        return saliency_map.copy(), salmax_coords, False
 
    weight  = confidence
    boosted = saliency_map.copy().astype(float)
 
    if word == "left":
        if h_quad == "left":
            return saliency_map.copy(), salmax_coords, False
        boosted[:, :mid_x] *= (1 + weight * BOOST_WEIGHT)
 
    elif word == "right":
        if h_quad == "right":
            return saliency_map.copy(), salmax_coords, False
        boosted[:, mid_x:] *= (1 + weight * BOOST_WEIGHT)
 
    elif word == "up":
        if v_quad == "top":
            return saliency_map.copy(), salmax_coords, False
        boosted[:mid_y, :] *= (1 + weight * BOOST_WEIGHT)
 
    elif word == "down":
        if v_quad == "bottom":
            return saliency_map.copy(), salmax_coords, False
        boosted[mid_y:, :] *= (1 + weight * BOOST_WEIGHT)
 
    else:
        return saliency_map.copy(), salmax_coords, False
 
    # clip and find new peak
    boosted    = np.clip(boosted, 0, 255)
    new_coords = np.unravel_index(np.argmax(boosted), boosted.shape)
    return boosted, new_coords, True
 
 
def normalise_for_display(arr):
    """Normalise float array to uint8 0–255."""
    arr = arr.astype(float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255
    return arr.astype(np.uint8)
 
 
# setup vision
 
device = torch.device("cpu")
print(f"Using device: {device}")
 
config = Config()
 
data = np.load('shapes_jitter_5_events.npy')
 
if data.dtype.names:
    x = data['x'].astype(int)
    y = data['y'].astype(int)
    p = data['p']
    t = data['t']
else:
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    p = data[:, 2]
    t = data[:, 3]
 
if t.max() < 100:
    t = t * 1e3
elif t.max() > 1e6:
    t = t / 1e3
 
max_x_orig = x.max() + 1
max_y_orig = y.max() + 1
 
DOWNSAMPLE_FACTOR = 4
max_x  = max_x_orig // DOWNSAMPLE_FACTOR
max_y  = max_y_orig // DOWNSAMPLE_FACTOR
 
x_down = (x // DOWNSAMPLE_FACTOR).clip(0, max_x - 1)
y_down = (y // DOWNSAMPLE_FACTOR).clip(0, max_y - 1)
 
resolution = (max_y, max_x)
 
print(f"Original resolution : {max_x_orig} x {max_y_orig}")
print(f"Processing resolution: {max_x} x {max_y} (downsampled {DOWNSAMPLE_FACTOR}x)")
print(f"Hardcoded command   : '{HARDCODED_WORD}'  (confidence={HARDCODED_CONFIDENCE})")
 
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
 
window_period = 100
time   = t.min() + window_period
window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
 
# video writer: side by side (2 × original width) 
 
output_filename = f'kws_modulation_{HARDCODED_WORD}.mp4'
fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
fps             = 10
# side by side: two panels at original resolution
frame_size      = (max_x_orig * 2, max_y_orig)
 
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
if not video_writer.isOpened():
    print("ERROR: Could not open video writer!")
    sys.exit(1)
 
# main loop
 
print("\nProcessing events...")
frame_count = 0
 
try:
    for xi, yi, ti in zip(x_down, y_down, t):
        if ti <= time:
            window[0][yi][xi] = 255
        else:
            # initial saliency 
            saliency_map, salmax_coords = run_attention(
                window, net_attention, device, resolution,
                config.ATTENTION_PARAMS['num_pyr']
            )
 
            # KWS modulation 
            boosted_map, new_coords, accepted = kws_modulate(
                saliency_map, salmax_coords,
                HARDCODED_WORD, HARDCODED_CONFIDENCE
            )
 
            # upscale both maps to original resolution (might be a problem later)
            def upscale(arr):
                return cv2.resize(
                    normalise_for_display(arr),
                    (max_x_orig, max_y_orig),
                    interpolation=cv2.INTER_LINEAR
                )
 
            orig_up    = upscale(saliency_map)
            boosted_up = upscale(boosted_map)
 
            orig_colored    = cv2.applyColorMap(orig_up,    cv2.COLORMAP_JET)
            boosted_colored = cv2.applyColorMap(boosted_up, cv2.COLORMAP_JET)
 
            # draw peaks 
            # original peak (white)
            orig_peak_x = int(salmax_coords[1] * DOWNSAMPLE_FACTOR)
            orig_peak_y = int(salmax_coords[0] * DOWNSAMPLE_FACTOR)
            cv2.circle(orig_colored, (orig_peak_x, orig_peak_y), 10, (255, 255, 255), 4)
 
            # new peak after modulation (white)
            new_peak_x = int(new_coords[1] * DOWNSAMPLE_FACTOR)
            new_peak_y = int(new_coords[0] * DOWNSAMPLE_FACTOR)
            cv2.circle(boosted_colored, (new_peak_x, new_peak_y), 10, (255, 255, 255), 4)
 
            # labels
            label_orig    = "Original saliency"
            label_boosted = f"After '{HARDCODED_WORD}' command"
            status        = "Accepted" if accepted else "Rejected (already there)"
 
            cv2.putText(orig_colored,    label_orig,    (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(boosted_colored, label_boosted, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(boosted_colored, status,        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if accepted else (0, 0, 255), 2)
 
            # figure side by side and write to video
            combined = np.hstack((orig_colored, boosted_colored))
            video_writer.write(combined)
            frame_count += 1
 
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: "
                      f"original peak=({orig_peak_x},{orig_peak_y})  "
                      f"→  new peak=({new_peak_x},{new_peak_y})  "
                      f"accepted={accepted}")
 
            time  += window_period
            window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
 
finally:
    video_writer.release()
    print(f"\nVideo saved : '{output_filename}'")
    print(f"Total frames: {frame_count}")
 