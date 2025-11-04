#!/usr/bin/env python3
# shebang line for linux / mac

import numpy as np
import cv2
from functools import partial
from scipy.optimize import least_squares
from copy import deepcopy


def changeImageColor(image_in, s, b, mask=None):
    # Convert to float
    image_in_float = image_in.astype(float)
    image_out_float = image_in_float * s + b
    image_out_float = np.clip(image_out_float, 0, 255)
    image_out = image_out_float.astype(np.uint8)

    # Keep original outside mask
    if mask is not None:
        mask_neg = np.logical_not(mask)
        image_out[mask_neg] = image_in[mask_neg]

    return image_out


def objectiveFunction(params, shared_mem):
    s, b = params
    q_image = shared_mem['q_image']
    t_image = shared_mem['t_image']
    q_mask = shared_mem['q_mask']
    mosaic_image = shared_mem['mosaic_image']

    # Apply transformation
    q_image_changed = changeImageColor(q_image, s, b, mask=q_mask)
    shared_mem['q_image'] = q_image_changed

    # --- Compute error ---
    # Get absolute difference between images
    diff_image = cv2.absdiff(t_image, q_image_changed)

    # Use the mask to only consider valid pixels
    masked_diff = diff_image[q_mask]

    # Mean absolute difference (scalar)
    error = np.mean(masked_diff)

    # --- Visualization ---
    # Blend for visualization
    mosaic_image[q_mask] = 0.5 * t_image[q_mask] + 0.5 * q_image_changed[q_mask]

    win_name_q = 'query image'
    win_name_m = 'mosaic image'

    cv2.namedWindow(win_name_q, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_name_m, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name_q, q_image_changed)
    cv2.imshow(win_name_m, mosaic_image)
    cv2.waitKey(1)

    print(f"s = {s:.3f}, b = {b:.3f}, error = {error:.3f}")

    # least_squares() expects a residual vector, not a scalar, so expand it
    return np.array([error])




# ------------------------------
# Example usage
# ------------------------------
q_image = cv2.imread('q_image_transformed.png')
t_image = cv2.imread('t_image.png')
q_mask = cv2.imread('q_mask.png', cv2.IMREAD_GRAYSCALE)
q_mask = q_mask > 128  # boolean mask

mosaic_image = deepcopy(t_image)

shared_mem = {
    'q_image': q_image,
    't_image': t_image,
    'q_mask': q_mask,
    'mosaic_image': mosaic_image
}

initial_params = [1.0, 0.0]

# Run optimization
result = least_squares(
    partial(objectiveFunction, shared_mem=shared_mem),
    initial_params,
    bounds=([0, -255], [5, 255]),
    diff_step=0.1,
    verbose=2
)

print("\n✅ Optimization finished.")
print(f"Best parameters: s = {result.x[0]:.3f}, b = {result.x[1]:.3f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
