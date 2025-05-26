import os
import cv2
import torch
import numpy as np
from PIL import Image
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker

def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette([0, 0, 0, 255, 255, 255])  # Black and white palette
    save_mask.save(os.path.join(output_dir, file_name))

def process_image_sequence(rgb_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of RGB images
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Initialize SegTracker
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    
    # Process images
    for frame_idx, rgb_file in enumerate(rgb_files):
        # Read image
        frame = cv2.imread(os.path.join(rgb_dir, rgb_file))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            # First frame: segment the object
            pred_mask = segtracker.seg(frame)
            segtracker.add_reference(frame, pred_mask)
        else:
            # Track the object in subsequent frames
            pred_mask = segtracker.track(frame, update_memory=True)

        # Save the mask
        save_prediction(pred_mask, output_dir, f"{frame_idx:04d}.png")
        
        print(f"Processed frame {frame_idx}", end='\r')

    print("\nFinished processing image sequence")

if __name__ == "__main__":
    # Set paths for lego object data
    base_dir = "/home/maria/mmint-research"
    rgb_dir = os.path.join(base_dir, "data/object_sdf_data/lego/rgb")
    output_dir = os.path.join(base_dir, "data/object_sdf_data/lego/masks")
    
    process_image_sequence(rgb_dir, output_dir) 