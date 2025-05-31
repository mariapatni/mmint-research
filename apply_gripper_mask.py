import os
import cv2
import numpy as np
import argparse
import shutil

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create gripper masks for all RGB frames')
    parser.add_argument('--object', type=str, required=True, help='Name of the object to process')
    args = parser.parse_args()

    # Get the mmint-research directory path
    mmint_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"mmint-research directory: {mmint_dir}")
    
    # Set up input and output paths
    rgb_dir = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "rgb")
    mask_path = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "gripper_mask.png")
    g_mask_dir = os.path.join(mmint_dir, "data/object_sdf_data", args.object, "g_mask")
    
    # Create output directory
    os.makedirs(g_mask_dir, exist_ok=True)
    
    # Read the gripper mask
    gripper_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gripper_mask is None:
        print(f"Could not read mask at {mask_path}")
        return
    
    # Get all RGB frames
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    if not rgb_files:
        print(f"No PNG files found in {rgb_dir}")
        return
    
    print(f"Processing {len(rgb_files)} frames...")
    
    # Process each frame
    for rgb_file in rgb_files:
        # Save mask with same name as RGB file
        output_path = os.path.join(g_mask_dir, rgb_file)
        cv2.imwrite(output_path, gripper_mask)
        print(f"Created mask for {rgb_file}")
    
    print(f"Finished creating {len(rgb_files)} masks")
    print(f"Masks saved to {g_mask_dir}")

if __name__ == "__main__":
    main() 