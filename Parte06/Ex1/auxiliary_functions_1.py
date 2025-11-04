#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import random
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse

# import matplotlib
# matplotlib.use('Agg')


def changeImageColor(image_in, s, b, mask=None):

    # image_in is uint8 -> [0 (black), 255 (white)]
    # [0, 255] * 1.0 -> float image

    # Convert the image to float type
    image_in_float = image_in.astype(float)  # 0 = black, 1=white

    image_out_float = image_in_float * s + b  # apply model

    image_out_float[image_out_float > 255] = 255  # saturate to 255
    image_out_float[image_out_float < 0] = 0  # undersaturate to 0

    # b = 0 255

    # convert back to uint8
    image_out = (image_out_float).astype(np.uint8)

    # Copy back the original values of image_in in the regions where the mask is zero
    # if mask is not None:

    #     mask_negated = np.logical_not(mask)
    #     image_out[mask_negated] = image_in[mask_negated]

    return image_out


def computeMosaic(t_image, q_image, mask):

    mosaic_image = deepcopy(q_image)  # the outer part is alreay ok, jsut need to change the middel
    mosaic_image[mask] = t_image[mask]
    # mosaic_image[q_mask] = q_image_transformed[q_mask]

    # Convert the mosaic back to unsigned integer 8 bits (uint8)
    mosaic_image = mosaic_image.astype(np.uint8)
    return mosaic_image


# Define the objective function
def objectiveFunction(params, shared_mem):
    # minimuzed version = objectiveFunction(params), shared_mem was already given

    # Extract the parameters
    # params = [s , b]
    s = params[0]
    b = params[1]
    print('s = ' + str(s))
    print('b = ' + str(b))
    q_image = shared_mem['q_image']
    t_image = shared_mem['t_image']
    q_mask = shared_mem['q_mask']
    q_mask = np.logical_not(q_mask)

    
    # Applying the image model
    t_image_changed = changeImageColor(t_image, s=s, b=b, mask=q_mask)

    # Compute the error
    diff_image = cv2.absdiff(q_image, t_image_changed)

    # Calculate the average of the absolute differences
    # The mean is calculated across all elements (pixels and channels) in the diff_image.
    error = np.mean(diff_image[np.logical_not(q_mask)])

    # TODO recompute the mosaic and show it
    mosaic_image = computeMosaic(t_image_changed, q_image, q_mask)

    # Draw the new line
    win_name = 'target image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, t_image_changed)  # type: ignore


    win_name = 'mosaic_target_changed'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, mosaic_image)  # type: ignore

    cv2.waitKey(500)

    print('error = ' + str(error))
    return error  # can be a scalar or a list
