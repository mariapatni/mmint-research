import os
import sys

# Change working directory to Segment-and-Track-Anything
segment_track_dir = os.path.join(os.path.dirname(__file__), 'Segment-and-Track-Anything')
os.chdir(segment_track_dir)
sys.path.append('.')

import cv2
import torch
import numpy as np
import shutil
import argparse
from SegTracker import SegTracker
from model_args import sam_args, aot_args, segtracker_args
from PIL import Image
from aot_tracker import _palette
import gc

# Update checkpoint paths to be relative to Segment-and-Track-Anything directory
sam_args["sam_checkpoint"] = os.path.join(segment_track_dir, "ckpt/sam_vit_b_01ec64.pth")
aot_args["model_path"] = os.path.join(segment_track_dir, "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth")

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

def refine_mask(mask):
    """Refine the mask by keeping only the largest component and filling holes"""
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
    
    # Fill holes more conservatively
    kernel = np.ones((5,5), np.uint8)  # Reduced kernel size
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)  # Single iteration
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)  # Single iteration
    
    # Additional check to prevent mask growth
    if np.sum(mask_eroded) > np.sum(mask_uint8) * 1.5:  # If mask grew by more than 50%
        return mask_uint8 > 0  # Return original mask
    
    return mask_eroded > 0

def save_visualization(image, mask, output_dir, frame_num, debug_info=None):
    """Save the visualization of the mask overlaid on the frame with optional debug info"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save the RGB overlay
    vis_path = os.path.join(output_dir, f"vis_{frame_num:04d}.png")
    display = image.copy()
    
    # Create a colored overlay
    overlay = np.zeros_like(display)
    
    # Get unique object IDs (excluding background 0)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]
    
    # Assign different colors to each object
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 0],    # Dark Red
        [0, 128, 0],    # Dark Green
        [0, 0, 128],    # Dark Blue
    ]
    
    # Color each object differently
    for i, obj_id in enumerate(obj_ids):
        color = colors[i % len(colors)]
        overlay[mask == obj_id] = color
        
        # Add debug info if provided
        if debug_info and obj_id in debug_info:
            center = get_mask_center((mask == obj_id).astype(np.uint8))
            if center:
                cv2.circle(display, center, 5, color, -1)
                cv2.putText(display, f"ID:{obj_id}", (center[0]+10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add frame center marker
    h, w = image.shape[:2]
    cv2.circle(display, (w//2, h//2), 5, [255, 255, 255], -1)
    
    # Blend the overlay with the original image
    display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
    
    # Save the RGB overlay
    cv2.imwrite(vis_path, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

def is_mask_near_edge(mask, threshold=50):
    """Check if the mask is near the edge of the image"""
    h, w = mask.shape
    # Find the bounding box of the mask
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return False
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    # Check if any part of the mask is near the edges
    return (min_x < threshold or max_x > w - threshold or 
            min_y < threshold or max_y > h - threshold)

def is_grayish(color, threshold=30):
    """Check if a color is grayish by comparing RGB channels"""
    r, g, b = color
    return abs(r - g) < threshold and abs(r - b) < threshold and abs(g - b) < threshold

def get_mask_color_score(frame, mask):
    """Calculate a score based on how non-gray the masked region is"""
    # Get the average color of the masked region
    masked_region = frame[mask]
    if len(masked_region) == 0:
        return 0.0
    
    avg_color = np.mean(masked_region, axis=0)
    
    # Calculate colorfulness score
    # Higher score for more colorful regions, lower for grayish regions
    r, g, b = avg_color
    color_variance = np.var([r, g, b])
    gray_score = 1.0 - (color_variance / 255.0)  # Normalize to [0,1]
    
    # Additional check for grayish colors
    if is_grayish(avg_color):
        gray_score *= 0.5  # Penalize grayish colors
    
    return 1.0 - gray_score  # Convert to colorfulness score

def is_mask_touching_boundary(mask, threshold=5):
    """Check if mask touches image boundary"""
    h, w = mask.shape
    # Check top and bottom edges
    if np.any(mask[:threshold, :]) or np.any(mask[-threshold:, :]):
        return True
    # Check left and right edges
    if np.any(mask[:, :threshold]) or np.any(mask[:, -threshold:]):
        return True
    return False

def filter_mask_by_criteria(mask, frame_shape, min_size=1000, max_size=6000, center_weight=0.9):
    """Filter mask based on multiple criteria:
    - Size constraints
    - Distance from center
    - Aspect ratio
    - Boundary check
    """
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Get all unique object IDs
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]  # Remove background
    
    if len(obj_ids) == 0:
        return mask
    
    best_obj_id = None
    best_score = -float('inf')
    
    for obj_id in obj_ids:
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8)
        
        # Skip if mask touches boundary
        if is_mask_touching_boundary(obj_mask):
            continue
        
        # Calculate area
        area = np.sum(obj_mask)
        if area < min_size or area > max_size:
            continue
        
        # Get center and contour
        center, contour = get_mask_center(obj_mask)
        if center is None:
            continue
            
        # Calculate distance from image center
        dist_from_center = np.sqrt((center[0] - center_x)**2 + (center[1] - center_y)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        center_score = 1 - (dist_from_center / max_dist)
        
        # Calculate aspect ratio
        if contour is not None:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-6)
            aspect_score = 1 / (1 + abs(aspect_ratio - 1))  # Prefer more square objects
        else:
            aspect_score = 0
        
        # Calculate size score (prefer medium-sized objects)
        size_score = 1 - abs(area - (max_size + min_size)/2) / ((max_size - min_size)/2)
        
        # Combined score with stronger center weighting
        score = (center_score * center_weight + 
                aspect_score * (1 - center_weight) * 0.5 + 
                size_score * (1 - center_weight) * 0.5)
        
        if score > best_score:
            best_score = score
            best_obj_id = obj_id
    
    # Create new mask with only the best object
    new_mask = np.zeros_like(mask)
    if best_obj_id is not None:
        new_mask[mask == best_obj_id] = 1
    
    return new_mask

def select_center_object(mask, frame_shape):
    """Select the object closest to the center of the frame with size constraints"""
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Get unique object IDs (excluding background 0)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]
    
    if len(obj_ids) == 0:
        return mask
    
    best_obj_id = None
    best_score = -float('inf')
    
    # Size constraints
    min_size = 1000  # Minimum number of pixels
    max_size = 10000  # Maximum number of pixels
    
    for obj_id in obj_ids:
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8)
        
        # Calculate area
        area = np.sum(obj_mask)
        if area < min_size or area > max_size:
            continue
        
        # Get center of this object
        center = get_mask_center(obj_mask)
        if center is None:
            continue
        
        # Calculate distance from frame center
        dist = np.sqrt((center[0] - center_x)**2 + (center[1] - center_y)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        center_score = 1 - (dist / max_dist)  # Higher score for objects closer to center
        
        # Calculate aspect ratio
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contours[0])
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-6)
            aspect_score = 1 / (1 + abs(aspect_ratio - 1))  # Prefer more square objects
        else:
            aspect_score = 0
        
        # Combined score (heavily weighted towards center position)
        score = center_score * 0.8 + aspect_score * 0.2
        
        if score > best_score:
            best_score = score
            best_obj_id = obj_id
    
    # Create new mask with only the best object
    new_mask = np.zeros_like(mask)
    if best_obj_id is not None:
        new_mask[mask == best_obj_id] = 1
    
    return new_mask

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
    parser = argparse.ArgumentParser(description='Segment and track object in video frames')
    parser.add_argument('--object', type=str, required=True, help='Name of the object to track')
    args = parser.parse_args()

    # Get the mmint-research directory path (only go up one level)
    mmint_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"mmint-research directory: {mmint_dir}")
    
    # Set up input and output directories
    input_dir = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "rgb")
    masks_dir = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "masks")
    vis_dir = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "vis")
    print(f"Input directory: {input_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Visualization directory: {vis_dir}")
    
    # Create output directories if they don't exist
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Clear existing output directory contents
    for dir_path in [masks_dir, vis_dir]:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
    
    # Initialize SegTracker
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    
    # Get all frames
    rgb_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if not rgb_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    # Process each frame
    with torch.cuda.amp.autocast():
        for frame_num, rgb_file in enumerate(rgb_files):
            # Read frame
            frame = cv2.imread(os.path.join(input_dir, rgb_file))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_num == 0:
                # For first frame, use box prompt
                h, w = frame.shape[:2]
                
                # Create a box around 25% from the left edge, centered vertically
                box_size = min(h, w) // 3
                center_y = h // 2
                x0 = int(w * 0.25) - box_size // 2  # 25% from left edge
                y0 = center_y - box_size // 2
                x1 = int(w * 0.25) + box_size // 2  # 25% from left edge
                y1 = center_y + box_size // 2
                
                # Get mask using box prompt
                pred_mask, _ = segtracker.seg_acc_bbox(frame, [[x0, y0], [x1, y1]])
                torch.cuda.empty_cache()
                gc.collect()
                
                if pred_mask is not None:
                    # Save debug visualization of all detected objects
                    save_visualization(frame, pred_mask, vis_dir, frame_num, debug_info={})
                    
                    # Select the object closest to center
                    pred_mask = select_center_object(pred_mask, frame.shape)
                    
                    # Keep only the largest component
                    pred_mask = keep_largest_component(pred_mask)
                    
                    # Print pixel sum
                    print(f"Frame {frame_num + 1}: Pixels = {np.sum(pred_mask)}")
                    
                    # Save mask visualization
                    save_visualization(frame, pred_mask, vis_dir, frame_num)
                    
                    # Save binary mask
                    mask_path = os.path.join(masks_dir, f"mask_{frame_num:04d}.png")
                    cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
                    
                    # Initialize tracking with the selected mask
                    segtracker.add_reference(frame, pred_mask)
                else:
                    print("No suitable mask found in first frame")
                    continue
            else:
                # Track the object
                pred_mask = segtracker.track(frame, update_memory=True)
                
                # Keep only the largest component
                pred_mask = keep_largest_component(pred_mask)
                
                # Save binary mask
                mask_path = os.path.join(masks_dir, f"mask_{frame_num:04d}.png")
                cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save visualization
            save_visualization(frame, pred_mask, vis_dir, frame_num)
            
            print(f"Processed frame {frame_num + 1}, obj_num {segtracker.get_obj_num()}", end='\r')
    
    print('\nFinished')
    
    # Clean up
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main() 