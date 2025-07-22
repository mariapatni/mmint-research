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
from PIL import Image
from aot_tracker import _palette
import gc

# Global variables for interactive bounding box selection
drawing = False
start_point = None
end_point = None
bbox_coords = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for interactive bounding box selection"""
    global drawing, start_point, end_point, bbox_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Ensure coordinates are in correct order (top-left to bottom-right)
        x0, y0 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x1, y1 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        bbox_coords = (x0, y0, x1, y1)
        print(f"Bounding box selected: ({x0}, {y0}) to ({x1}, {y1})")

def draw_bbox(image, start_point, end_point):
    """Draw the bounding box on the image"""
    if start_point and end_point:
        x0, y0 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x1, y1 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(image, "Drag to select bounding box", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, "Press 'Enter' to confirm, 'R' to reset", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def interactive_bbox_selection(frame):
    """Interactive bounding box selection using GUI"""
    global drawing, start_point, end_point, bbox_coords
    
    # Reset global variables
    drawing = False
    start_point = None
    end_point = None
    bbox_coords = None
    
    # Create window and set mouse callback
    window_name = "Select Bounding Box - Drag to select, Enter to confirm, R to reset"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize window to fit screen
    h, w = frame.shape[:2]
    screen_width = 1920  # Adjust based on your screen
    screen_height = 1080
    scale = min(screen_width / w, screen_height / h) * 0.8
    new_w, new_h = int(w * scale), int(h * scale)
    cv2.resizeWindow(window_name, new_w, new_h)
    
    print("Interactive bounding box selection:")
    print("- Click and drag to draw bounding box")
    print("- Press 'Enter' to confirm selection")
    print("- Press 'R' to reset selection")
    print("- Press 'Q' to quit")
    
    while True:
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw current bounding box
        draw_bbox(display_frame, start_point, end_point)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('r') or key == ord('R'):
            # Reset selection
            drawing = False
            start_point = None
            end_point = None
            bbox_coords = None
            print("Selection reset")
        elif key == 13:  # Enter key
            if bbox_coords:
                cv2.destroyAllWindows()
                return bbox_coords
            else:
                print("Please select a bounding box first")
    
    cv2.destroyAllWindows()
    return None

# Update checkpoint paths to be robust
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

def save_visualization(image, mask, output_dir, frame_num, debug_info=None, prompt_box=None):
    """Save the visualization of the mask overlaid on the frame with optional debug info"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save the RGB overlay
    vis_path = os.path.join(output_dir, f"{frame_num:04d}.png")
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
    
    # Draw prompt box if provided
    if prompt_box is not None:
        x0, y0, x1, y1 = prompt_box
        cv2.rectangle(display, (x0, y0), (x1, y1), [0, 255, 255], 2)  # Cyan box
        cv2.putText(display, "Prompt Box", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255], 2)
    
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
    
    # Ensure we have exactly 3 color channels
    if len(avg_color) != 3:
        print(f"Warning: Expected 3 color channels, got {len(avg_color)}")
        return 0.0
    
    # Calculate colorfulness score
    # Higher score for more colorful regions, lower for grayish regions
    r, g, b = avg_color
    color_variance = np.var([r, g, b])
    gray_score = 1.0 - (color_variance / 255.0)  # Normalize to [0,1]
    
    # Additional check for grayish colors
    if is_grayish(avg_color):
        gray_score *= 0.5  # Penalize grayish colors
    
    return 1.0 - gray_score  # Convert to colorfulness score

def detect_grid_pattern(mask, frame_shape, min_grid_size=20):
    """Detect if a mask contains a grid pattern"""
    h, w = frame_shape[:2]
    
    # Convert mask to binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Check if the mask has many small connected components (grid-like)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    # If there are many small components, it might be a grid
    if num_labels > 10:  # More than 10 components
        # Check if most components are small
        small_components = 0
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_grid_size * min_grid_size:
                small_components += 1
        
        # If more than 70% are small components, likely a grid
        if small_components / (num_labels - 1) > 0.7:
            return True
    
    return False

def get_mask_compactness(mask):
    """Calculate compactness score (area / perimeter^2) - higher is more compact"""
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return 0.0
    
    # Compactness = area / perimeter^2 (higher is more compact)
    compactness = area / (perimeter * perimeter)
    
    return compactness

def get_foreground_score(mask, frame_shape):
    """Calculate foreground score based on position and size - objects closer to camera appear larger"""
    h, w = frame_shape[:2]
    
    # Convert mask to binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Find the bounding box of the mask
    y_indices, x_indices = np.where(mask_binary > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0.0
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    # Calculate relative size (larger objects are likely closer)
    mask_area = np.sum(mask_binary)
    frame_area = h * w
    relative_size = mask_area / frame_area
    
    # Calculate position score (objects in lower part of frame are often closer)
    # In most camera setups, objects closer to camera appear lower in the frame
    center_y = (min_y + max_y) / 2
    position_score = 1.0 - (center_y / h)  # Higher score for objects lower in frame
    
    # Calculate edge proximity (objects touching edges are often background)
    edge_distance = min(min_x, min_y, w - max_x, h - max_y)
    edge_score = min(edge_distance / 100.0, 1.0)  # Normalize to [0,1]
    
    # Combined foreground score
    foreground_score = (relative_size * 0.4 + 
                       position_score * 0.4 + 
                       edge_score * 0.2)
    
    return foreground_score

def detect_background_mask(mask, frame_shape):
    """Detect if a mask is likely background (covers too much of the frame)"""
    h, w = frame_shape[:2]
    
    # Convert mask to binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Calculate coverage
    mask_area = np.sum(mask_binary)
    frame_area = h * w
    coverage = mask_area / frame_area
    
    # If mask covers more than 70% of frame, likely background
    if coverage > 0.7:
        return True
    
    # Check if mask touches all edges (typical of background)
    y_indices, x_indices = np.where(mask_binary > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return False
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    # If mask touches all four edges, likely background
    touches_left = min_x < 10
    touches_right = max_x > w - 10
    touches_top = min_y < 10
    touches_bottom = max_y > h - 10
    
    if touches_left and touches_right and touches_top and touches_bottom:
        return True
    
    return False

def get_mask_inverse_score(mask, frame_shape):
    """Calculate score for the inverse of the mask (to find the actual object)"""
    h, w = frame_shape[:2]
    
    # Create inverse mask
    inverse_mask = np.ones((h, w), dtype=np.uint8)
    inverse_mask[mask > 0] = 0
    
    # Calculate foreground score for inverse
    return get_foreground_score(inverse_mask, frame_shape)

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

def select_center_object(mask, frame_shape, frame=None):
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
    
    # More lenient size constraints
    min_size = 500  # Reduced minimum
    max_size = 50000  # Increased maximum
    
    for obj_id in obj_ids:
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8)
        
        # Calculate area
        area = np.sum(obj_mask)
        if area < min_size or area > max_size:
            continue
        
        # Skip if this looks like a grid pattern
        if detect_grid_pattern(obj_mask, frame_shape):
            print(f"Object {obj_id} detected as grid pattern, skipping")
            continue
        
        # Check if this is likely background (covers too much or touches all edges)
        if detect_background_mask(obj_mask, frame_shape):
            print(f"Object {obj_id} detected as background, skipping")
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
        
        # Calculate compactness (prefer more compact objects, not grid-like)
        compactness_score = get_mask_compactness(obj_mask)
        
        # Calculate foreground score (prefer objects closer to camera)
        foreground_score = get_foreground_score(obj_mask, frame_shape)
        
        # Calculate color score if frame is provided
        color_score = 0.0
        if frame is not None:
            color_score = get_mask_color_score(frame, obj_mask)
        
        # Combined score with heavy foreground weighting
        score = (foreground_score * 0.6 +  # Heavily weight foreground objects
                center_score * 0.2 + 
                compactness_score * 1000 +  # Scale up compactness
                color_score * 0.2)
        
        print(f"Object {obj_id}: foreground={foreground_score:.3f}, center={center_score:.3f}, compactness={compactness_score:.6f}, color={color_score:.3f}, total={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_obj_id = obj_id
    
    # If no object meets the criteria, try without grid detection
    if best_obj_id is None:
        print("No object met criteria, trying without grid detection...")
        for obj_id in obj_ids:
            obj_mask = (mask == obj_id).astype(np.uint8)
            area = np.sum(obj_mask)
            if area < min_size or area > max_size:
                continue
            
            center = get_mask_center(obj_mask)
            if center is None:
                continue
            
            dist = np.sqrt((center[0] - center_x)**2 + (center[1] - center_y)**2)
            max_dist = np.sqrt(w**2 + h**2) / 2
            center_score = 1 - (dist / max_dist)
            
            compactness_score = get_mask_compactness(obj_mask)
            foreground_score = get_foreground_score(obj_mask, frame_shape)
            
            score = foreground_score * 0.7 + center_score * 0.3 + compactness_score * 1000
            
            print(f"Object {obj_id} (no grid filter): foreground={foreground_score:.3f}, center={center_score:.3f}, compactness={compactness_score:.6f}, total={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_obj_id = obj_id
    
    # If still no object, just return the largest object
    if best_obj_id is None:
        print("No object met criteria, selecting largest object")
        largest_area = 0
        for obj_id in obj_ids:
            obj_mask = (mask == obj_id).astype(np.uint8)
            area = np.sum(obj_mask)
            if area > largest_area:
                largest_area = area
                best_obj_id = obj_id
    
    # Create new mask with only the best object
    new_mask = np.zeros_like(mask)
    if best_obj_id is not None:
        new_mask[mask == best_obj_id] = 1
        print(f"Selected object {best_obj_id} with {np.sum(new_mask)} pixels")
    else:
        print("No object selected, returning original mask")
        return mask
    
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
    parser.add_argument('--input_dir', type=str, required=True, help='Custom input directory path (overrides default object_sdf_data path)')
    parser.add_argument('--output_dir', type=str, help='Custom output directory path (overrides default object_sdf_data path)')
    args = parser.parse_args()

    input_dir = args.input_dir

    output_base_dir = input_dir
    masks_dir = input_dir.replace("rgb", "masks")
    vis_dir = input_dir.replace("rgb", "vis")
    
    
    if args.output_dir:
        # Use custom output directory
        masks_dir = os.path.join(args.output_dir, "masks")
        vis_dir = os.path.join(args.output_dir, "masks_vis")
        print(f"Using custom output directory: {args.output_dir}")
    
    
    print(f"Input directory: {input_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Visualization directory: {vis_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        print("Please check the path or use --input_dir to specify a custom path")
        return
    
    # After defining masks_dir and vis_dir
    for dir_path in [masks_dir, vis_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # Clear existing output directory contents
    for dir_path in [masks_dir, vis_dir]:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
    
    # Initialize SegTracker
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    
    # Get all frames - try both rgb and color subdirectories
    rgb_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not rgb_files:
        # Try looking for a 'color' subdirectory
        color_dir = os.path.join(input_dir, 'color')
        if os.path.exists(color_dir):
            input_dir = color_dir
            rgb_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Found images in color subdirectory: {input_dir}")
        else:
            # Try looking for a 'rgb' subdirectory
            rgb_subdir = os.path.join(input_dir, 'rgb')
            if os.path.exists(rgb_subdir):
                input_dir = rgb_subdir
                rgb_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Found images in rgb subdirectory: {input_dir}")
    
    if not rgb_files:
        print(f"No PNG, JPG, or JPEG files found in {input_dir}")
        print("Please check that your images are in the correct directory")
        return
    
    print(f"Found {len(rgb_files)} image files")
    
    # Process each frame
    with torch.cuda.amp.autocast():
        for frame_num, rgb_file in enumerate(rgb_files):
            # Read frame
            frame = cv2.imread(os.path.join(input_dir, rgb_file))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_num == 0:
                # For first frame, use interactive bounding box selection
                h, w = frame.shape[:2]
                
                print(f"Frame shape: {frame.shape}")
                print("Opening interactive bounding box selection...")
                
                # Convert frame to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Get bounding box from user interaction
                bbox_coords = interactive_bbox_selection(frame_bgr)
                
                if bbox_coords is None:
                    print("No bounding box selected, exiting...")
                    return
                
                x0, y0, x1, y1 = bbox_coords
                print(f"Selected bounding box: ({x0}, {y0}) to ({x1}, {y1})")
                print(f"Box dimensions: {x1-x0} x {y1-y0}")
                
                # Get mask using box prompt
                pred_mask, _ = segtracker.seg_acc_bbox(frame, [[x0, y0], [x1, y1]])
                torch.cuda.empty_cache()
                gc.collect()
                
                if pred_mask is not None:
                    print(f"Initial segmentation found {len(np.unique(pred_mask))-1} objects")
                    print(f"Initial mask pixel sum: {np.sum(pred_mask)}")
                    
                    # Save debug visualization of all detected objects with prompt box
                    save_visualization(frame, pred_mask, vis_dir, frame_num, debug_info={}, prompt_box=(x0, y0, x1, y1))
                    
                    # Select the object closest to center
                    # Ensure frame is in RGB format for color analysis
                    frame_rgb = frame.copy()
                    if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] > 3:
                        # If more than 3 channels, take only the first 3
                        frame_rgb = frame_rgb[:, :, :3]
                    pred_mask = select_center_object(pred_mask, frame.shape, frame_rgb)
                    print(f"After center selection: {np.sum(pred_mask)} pixels")
                    
                    # Keep only the largest component
                    pred_mask = keep_largest_component(pred_mask)
                    print(f"After largest component: {np.sum(pred_mask)} pixels")
                    
                    # If we got a reasonable mask, use it
                    if np.sum(pred_mask) > 500:  # At least 500 pixels
                        print(f"Frame {frame_num + 1}: Final Pixels = {np.sum(pred_mask)}")
                        
                        # Save mask visualization
                        save_visualization(frame, pred_mask, vis_dir, frame_num, prompt_box=(x0, y0, x1, y1))
                        
                        # Save binary mask with correct naming format
                        mask_path = os.path.join(masks_dir, rgb_file)
                        cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
                        
                        # Initialize tracking with the selected mask
                        segtracker.add_reference(frame, pred_mask)
                    else:
                        print("Mask too small, trying larger box...")
                        # Try with a larger box (75% of frame dimensions)
                        center_x, center_y = w // 2, h // 2
                        box_width = int(w * 0.75)  # 75% of frame width
                        box_height = int(h * 0.75)  # 75% of frame height
                        x0 = center_x - box_width // 2
                        y0 = center_y - box_height // 2
                        x1 = center_x + box_width // 2
                        y1 = center_y + box_height // 2
                        
                        print(f"Trying larger box: ({x0}, {y0}) to ({x1}, {y1})")
                        pred_mask, _ = segtracker.seg_acc_bbox(frame, [[x0, y0], [x1, y1]])
                        
                        if pred_mask is not None:
                            # Ensure frame is in RGB format for color analysis
                            frame_rgb = frame.copy()
                            if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] > 3:
                                # If more than 3 channels, take only the first 3
                                frame_rgb = frame_rgb[:, :, :3]
                            pred_mask = select_center_object(pred_mask, frame.shape, frame_rgb)
                            pred_mask = keep_largest_component(pred_mask)
                            
                            if np.sum(pred_mask) > 500:
                                print(f"Larger box worked: {np.sum(pred_mask)} pixels")
                                save_visualization(frame, pred_mask, vis_dir, frame_num, prompt_box=(x0, y0, x1, y1))
                                mask_path = os.path.join(masks_dir, rgb_file)
                                cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
                                segtracker.add_reference(frame, pred_mask)
                            else:
                                print("Still no good mask found")
                                continue
                        else:
                            print("No mask found with larger box")
                            continue
                else:
                    print("No suitable mask found in first frame")
                    continue
            else:
                # Track the object
                try:
                    pred_mask = segtracker.track(frame, update_memory=True)
                    
                    if pred_mask is not None and np.sum(pred_mask) > 0:
                        # Keep only the largest component
                        pred_mask = keep_largest_component(pred_mask)
                        
                        # Save binary mask with correct naming format
                        mask_path = os.path.join(masks_dir, rgb_file)
                        cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
                        
                        # Save visualization
                        save_visualization(frame, pred_mask, vis_dir, frame_num)
                        
                        print(f"Processed frame {frame_num + 1}, obj_num {segtracker.get_obj_num()}", end='\r')
                    else:
                        print(f"Frame {frame_num + 1}: No valid mask from tracking")
                        # Save empty mask with correct naming format
                        mask_path = os.path.join(masks_dir, rgb_file)
                        cv2.imwrite(mask_path, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8))
                        
                except Exception as e:
                    print(f"Error tracking frame {frame_num + 1}: {e}")
                    # Save empty mask with correct naming format
                    mask_path = os.path.join(masks_dir, rgb_file)
                    cv2.imwrite(mask_path, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8))
                    continue
            
            torch.cuda.empty_cache()
            gc.collect()
    
    print('\nFinished')
    
    # Clean up
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main() 