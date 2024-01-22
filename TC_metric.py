import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
import matplotlib.pyplot as plt
import numpy as np

def TCmetric(depth1, depth2, image1, image2, occlusion_threshold, threshold):
        #compute the optical flow
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #the depth image is in gray scale with one channel and dimentions (256,192), 
    #so we add a dimension to allow it to be broadcasted or used in operations that require an additional dimension.
    depth1 = np.expand_dims(depth1, axis=-1)
    depth2 = np.expand_dims(depth2, axis=-1)

    #applying the optical flow on the 1st predicted image 
    height, width, _ = depth1.shape
    
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    dx, dy = flow[..., 0], flow[..., 1]
    map_x = x_coords + dx
    map_y = y_coords + dy
    warped_depth1 = cv2.remap(depth1, map_x, map_y, cv2.INTER_LINEAR)
    
    #calculating the occlusion matrix
    warped_image1 = cv2.remap(image1, map_x, map_y, cv2.INTER_LINEAR)
    #3-channel image, where each pixel represents the absolute difference between the corresponding pixel values in image2 and warped_image1
    color_diff = np.abs(image2 - warped_image1)
    non_occlusion_mask = (color_diff < occlusion_threshold).astype(np.uint8)
    
    #warped_depth2 is with two dimenstions, we added a third one so we can divide it by depth2 that is with 3d.
    warped_depth1 = np.expand_dims(warped_depth1, axis=-1)
    # Compute the ratio of depth change
    ratio_depth_change = np.abs(depth2 / warped_depth1)
    num_valid_pixels = np.sum(non_occlusion_mask)
    
    # Computing the TC metric
    temporal_consistency = np.sum(non_occlusion_mask *(np.maximum(ratio_depth_change, 1/ratio_depth_change) < threshold)) / num_valid_pixels
    return temporal_consistency