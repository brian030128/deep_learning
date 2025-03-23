import torch 

def dice_score(pred_mask, gt_mask):
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    
    # Calculate intersection of both positive and negative classes
    # For positive class (1s)
    pos_intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
    # For negative class (0s)
    neg_pred = 1 - pred_mask
    neg_gt = 1 - gt_mask
    neg_intersection = (neg_pred * neg_gt).sum(dim=(1, 2))
    
    # Total intersection is sum of both classes
    total_intersection = pos_intersection + neg_intersection
    
    # Total number of pixels in each image
    total_pixels = pred_mask.shape[1] * pred_mask.shape[2]
    
    # Calculate dice considering the entire image
    dice = (2 * total_intersection) / (2 * total_pixels)
    
    return dice

def output_2_mask(output):
    return torch.argmax(output, axis=1)

import numpy as np
import matplotlib.pyplot as plt

colors = np.array([[128, 0, 128], [255, 255, 0]]) 

def display_mask(mask):
    print("wwwww")
    print(mask.shape)
    tensor_np = mask.cpu().numpy()

    # Create an RGB image
    rgb_image = np.zeros((*tensor_np.shape, 3), dtype=np.uint8)
    for i in range(tensor_np.shape[0]):
        for j in range(tensor_np.shape[1]):
            rgb_image[i, j] = colors[int(tensor_np[i, j])]

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.axis('off')  # Hide axis
    plt.title('Binary Tensor Visualization')
    plt.show()