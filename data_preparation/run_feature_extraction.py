#!/usr/bin/env python3

import os
import subprocess
import argparse
import sys
import shutil
import sqlite3
import numpy as np
import glob
import configparser

def create_colmap_project_ini(database_path, image_path, project_path, camera_mask_path=None):
    """Create a COLMAP project.ini file with all settings."""
    config = configparser.ConfigParser()
    
    # Database settings
    config['Database'] = {
        'database_path': database_path
    }
    
    # Image reader settings
    config['ImageReader'] = {
        'image_path': image_path,
        'camera_model': 'SIMPLE_RADIAL',
        'single_camera': '0',
        'single_camera_per_folder': '0',
        'single_camera_per_image': '0',
        'existing_camera_id': '-1',
        'default_focal_length_factor': '1.2'
    }
    
    if camera_mask_path:
        config['ImageReader']['camera_mask_path'] = camera_mask_path
    
    # SIFT extraction settings (using defaults from COLMAP 3.12.0)
    config['SiftExtraction'] = {
        'num_threads': '-1',
        'use_gpu': '1',
        'gpu_index': '-1',
        'max_image_size': '3200',
        'max_num_features': '8192',
        'first_octave': '-1',
        'num_octaves': '4',
        'octave_resolution': '3',
        'peak_threshold': '0.0066666666666666671',
        'edge_threshold': '10',
        'estimate_affine_shape': '0',
        'max_num_orientations': '2',
        'upright': '0',
        'domain_size_pooling': '0',
        'dsp_min_scale': '0.16666666666666666',
        'dsp_max_scale': '3',
        'dsp_num_scales': '10'
    }
    
    # SIFT matching settings (using defaults)
    config['SiftMatching'] = {
        'num_threads': '-1',
        'use_gpu': '1',
        'gpu_index': '-1',
        'max_ratio': '0.8',
        'max_distance': '0.7',
        'cross_check': '1',
        'max_num_matches': '32768',
        'guided_matching': '0'
    }
    
    # Sequential matching settings
    config['SequentialMatching'] = {
        'num_threads': '-1',
        'overlap': '20',
        'loop_detection': '0'
    }
    
    # Write the config file
    with open(project_path, 'w') as f:
        config.write(f)
    
    print(f"COLMAP project file created: {project_path}")

def sample_images(object_dir, n):
    """Copy every nth image from rgb to rgb_sampled, recreating rgb_sampled each time."""
    rgb_dir = os.path.join(object_dir, "rgb")
    sampled_dir = os.path.join(object_dir, "rgb_sampled")
    # Remove and recreate sampled_dir
    if os.path.exists(sampled_dir):
        shutil.rmtree(sampled_dir)
    os.makedirs(sampled_dir, exist_ok=True)
    # Get sorted list of images
    images = sorted(glob.glob(os.path.join(rgb_dir, '*')))
    for idx, img_path in enumerate(images):
        if idx % n == 0:
            shutil.copy(img_path, sampled_dir)
    print(f"Sampled {len(os.listdir(sampled_dir))} images (every {n}th) to {sampled_dir}")

def run_colmap_feature_extraction(object_id, sample_every=1):
    """Run COLMAP feature extraction and matching on the object's images."""
    # Get the mmint-research directory path robustly
    mmint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Define paths robustly
    object_dir = os.path.join(mmint_dir, "data/object_sdf_data", object_id)
    images_dir = os.path.join(object_dir, "rgb_sampled")
    rgb_dir = os.path.join(object_dir, "rgb")
    database_path = os.path.join(object_dir, "database.db")
    sparse_dir = os.path.join(object_dir, "sparse")
    project_path = os.path.join(object_dir, "project.ini")

    # Robust directory and file checks
    if not os.path.isdir(object_dir):
        print(f"ERROR: Object directory does not exist: {object_dir}")
        sys.exit(1)
    if not os.path.isdir(rgb_dir):
        print(f"ERROR: RGB images directory does not exist: {rgb_dir}")
        sys.exit(1)
    if not os.path.isfile(database_path):
        print(f"WARNING: database.db does not exist yet in {object_dir} (will be created)")

    # Sample images first
    sample_images(object_dir, sample_every)
    
    # Clean up existing database if it exists
    if os.path.exists(database_path):
        os.remove(database_path)
    
    # Clean up existing sparse directory if it exists
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Create COLMAP project.ini file (for reference/documentation)
    camera_mask_path = "/data/object_sdf_data/ob_0000002/gripper_mask.png"
    create_colmap_project_ini(database_path, images_dir, project_path, camera_mask_path)
    
    # Run feature extraction with EXIF intrinsics and all other defaults, matching GUI screenshot
    print("Running feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.camera_mask_path", camera_mask_path
    ])
    
    # Run sequential matcher with only database_path set, all else default
    print("Running sequential matcher...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", database_path
    ])



def main():
    parser = argparse.ArgumentParser(description="Run COLMAP feature extraction and matching.")
    parser.add_argument('--object', required=True, help='Object ID (e.g., ob_0000001)')
    parser.add_argument('--sample_every', type=int, default=1, help='Sample every nth image from rgb (default: 1)')
    args = parser.parse_args()

    DATASET_PATH = "./data/object_sdf_data/ob_0000004"
    
    
    #run_colmap_feature_extraction(args.object, sample_every=args.sample_every)

if __name__ == "__main__":
    main() 