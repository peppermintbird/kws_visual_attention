import numpy as np
from helpers_visual_att import initialise_attention, run_attention
import torch
import cv2
import sys

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

device = torch.device("cpu")
print(f"Using device: {device}")

config = Config()

# Load data
data = np.load('animation_shapes_jitter_ev.npy')

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

# Time conversion
if t.max() < 100:
    t = t * 1e3
elif t.max() > 1e6:
    t = t / 1e3

max_x_orig = x.max() + 1
max_y_orig = y.max() + 1

# ============================================
# DOWNSAMPLING 
# ============================================
DOWNSAMPLE_FACTOR = 4  # Try 2, 4, or 8

# New resolution
max_x = max_x_orig // DOWNSAMPLE_FACTOR
max_y = max_y_orig // DOWNSAMPLE_FACTOR

# Scale coordinates down
x_down = (x // DOWNSAMPLE_FACTOR).clip(0, max_x - 1)
y_down = (y // DOWNSAMPLE_FACTOR).clip(0, max_y - 1)

resolution = (max_y, max_x)

print(f"\nOriginal resolution: {max_x_orig} x {max_y_orig}")
print(f"Processing resolution: {max_x} x {max_y} (downsampled {DOWNSAMPLE_FACTOR}x)")
print(f"Objects are now {DOWNSAMPLE_FACTOR}x smaller")
print(f"VM filter size with R0={config.ATTENTION_PARAMS['r0']}: ~{2*3*config.ATTENTION_PARAMS['r0']+1}px")

net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

window_period = 100
time = t.min() + window_period
window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

# Video at ORIGINAL resolution for nice viewing
output_filename = f'saliency_downsampled_{DOWNSAMPLE_FACTOR}x.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
frame_size = (max_x_orig, max_y_orig)

video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

if not video_writer.isOpened():
    print("ERROR: Could not open video writer!")
    sys.exit(1)

print("\nProcessing events...")
frame_count = 0

try:
    for xi, yi, ti in zip(x_down, y_down, t):
        if ti <= time:
            window[0][yi][xi] = 255
        else:
            # Run attention on DOWNSAMPLED resolution
            saliency_map, salmax_coords = run_attention(
                window, net_attention, device, resolution,
                config.ATTENTION_PARAMS['num_pyr']
            )
            
            # Upscale saliency to original resolution for viewing
            saliency_upscaled = cv2.resize(
                saliency_map.astype(np.uint8),
                (max_x_orig, max_y_orig),
                interpolation=cv2.INTER_LINEAR
            )
            
            saliency_colored = cv2.applyColorMap(saliency_upscaled, cv2.COLORMAP_JET)
            
            # Scale peak coordinates back to original
            peak_x = int(salmax_coords[1] * DOWNSAMPLE_FACTOR)
            peak_y = int(salmax_coords[0] * DOWNSAMPLE_FACTOR)
            
            cv2.circle(saliency_colored, (peak_x, peak_y), 10, (255, 255, 255), 4)
            cv2.putText(saliency_colored, f'Downsampled {DOWNSAMPLE_FACTOR}x', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_writer.write(saliency_colored)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: peak at ({peak_x}, {peak_y})")
            
            time += window_period
            window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

finally:
    video_writer.release()
    print(f"\nVideo saved: '{output_filename}'")
    print(f"Total frames: {frame_count}")