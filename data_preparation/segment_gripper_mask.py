import os
import sys
import shutil

# Change working directory to Segment-and-Track-Anything
segment_track_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Segment-and-Track-Anything'))
os.chdir(segment_track_dir)
sys.path.append('.')

import cv2
import torch
import numpy as np
import argparse
from SegTracker import SegTracker
from model_args import sam_args, aot_args, segtracker_args
import gc

# Update checkpoint paths to be robust
sam_args["sam_checkpoint"] = os.path.join(segment_track_dir, "ckpt/sam_vit_b_01ec64.pth")
aot_args["model_path"] = os.path.join(segment_track_dir, "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth")

# Adjust SAM parameters for more thorough segmentation
sam_args["generator_args"] = {
    'points_per_side': 64,  # Increased from 32 for more detailed segmentation
    'pred_iou_thresh': 0.5,  # Lowered from 0.65 to be more lenient
    'stability_score_thresh': 0.5,  # Lowered from 0.75 to be more lenient
    'crop_n_layers': 4,  # Increased from 3 for more detailed analysis
    'crop_n_points_downscale_factor': 1,  # Decreased from 2 for finer detail
    'min_mask_region_area': 50,  # Decreased from 100 to catch smaller parts
}

def get_mask_center(mask):
    """Get the center point of the mask"""
    # Convert to uint8 for OpenCV operations
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    
    return None

def keep_largest_component(mask):
    """Keep only the largest connected component in the mask"""
    # Convert to uint8 for OpenCV operations
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels > 1:  # If we have any components (besides background)
        # Find the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Create a new mask with only the largest component
        mask_uint8 = np.zeros_like(mask_uint8)
        mask_uint8[labels == largest_label] = 255
    
    return mask_uint8 > 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment gripper in first frame')
    parser.add_argument('--object', type=str, required=True, help='Name of the object to track')
    args = parser.parse_args()

    # Get the mmint-research directory path robustly
    mmint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"mmint-research directory: {mmint_dir}")
    
    # Set up input and output paths
    input_path = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "rgb", "0000.png")
    output_mask_path = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "gripper_mask.png")
    print(f"Input path: {input_path}")
    print(f"Output mask path: {output_mask_path}")
    
    # Read first frame
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Could not read image at {input_path}")
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # For first frame, use two box prompts for top-left and bottom-left corners
    h, w = frame.shape[:2]
    
    # Create boxes for top-left and bottom-left corners
    boxes = [
        # Top-left box: from (0,0) to (w/2.5, 2h/5)
        [[0, 0], [int(w / 2.5), (2 * h) // 5]],
        # Bottom-left box: from (0,h-2h/5) to (w/2.5, h)
        [[0, h - (2 * h) // 5], [int(w / 2.5), h]],
    ]
    
    # Draw boxes and search points on a copy of the frame for visualization
    vis_frame = frame.copy()
    for box in boxes:
        # Draw rectangle
        cv2.rectangle(vis_frame, 
                     (box[0][0], box[0][1]), 
                     (box[1][0], box[1][1]), 
                     (0, 255, 0), 2)  # Green color, thickness 2
        
        # Draw search points (grid of points within the box)
        step = 20  # Distance between points
        for x in range(box[0][0] + step, box[1][0], step):
            for y in range(box[0][1] + step, box[1][1], step):
                cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)  # Red dots
    
    # Save visualization
    vis_path = os.path.join(os.path.dirname(output_mask_path), "gripper_box_visualization.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
    print(f"Saved box visualization to {vis_path}")
    
    # Initialize SegTracker
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    
    # Initialize combined mask
    combined_mask = None
    
    # Process each box
    for box in boxes:
        # Get mask using box prompt
        pred_mask, _ = segtracker.seg_acc_bbox(frame, box)
        torch.cuda.empty_cache()
        gc.collect()
        
        if pred_mask is not None:
            # Keep only the largest component
            pred_mask = keep_largest_component(pred_mask)
            
            # Combine with previous mask if it exists
            if combined_mask is None:
                combined_mask = pred_mask
            else:
                combined_mask = np.logical_or(combined_mask, pred_mask)
    
    if combined_mask is not None:
        # Create initial mask where detected objects are white (255) and background is black (0)
        initial_mask = np.zeros_like(combined_mask, dtype=np.uint8)
        initial_mask[combined_mask] = 255
        
        # Add border around the detected area
        kernel = np.ones((5,5), np.uint8)
        initial_mask = cv2.dilate(initial_mask, kernel, iterations=5)
        
        # Invert the mask so the area to avoid is black (0) and background is white (255)
        final_mask = cv2.bitwise_not(initial_mask)
        
        # Print pixel sum
        print(f"Pixels in mask: {np.sum(final_mask > 0)}")
        
        # Save binary mask
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        cv2.imwrite(output_mask_path, final_mask)
        print(f"Saved mask to {output_mask_path}")
        
    else:
        print("No suitable masks found in first frame")
    
    # Clean up
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
