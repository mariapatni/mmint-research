#!/usr/bin/env python3

import os
import subprocess
import argparse
import sys
import shutil
import sqlite3
import numpy as np
import glob

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

def run_colmap_feature_extraction(object_id, sample_every=5):
    """Run COLMAP feature extraction and matching on the object's images."""
    # Define paths
    object_dir = os.path.join("data/object_sdf_data", object_id)
    images_dir = os.path.join(object_dir, "rgb_sampled")
    database_path = os.path.join(object_dir, "database.db")
    sparse_dir = os.path.join(object_dir, "sparse")
    
    # Sample images first
    sample_images(object_dir, sample_every)
    
    # Clean up existing database if it exists
    if os.path.exists(database_path):
        os.remove(database_path)
    
    # Clean up existing sparse directory if it exists
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Run feature extraction with provided intrinsics
    print("Running feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.camera_params", "387.290558,324.675354,240.405365,0",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "8000",
        "--SiftExtraction.peak_threshold", "0.01"
    ])
    
    # Run sequential matcher
    print("Running sequential matcher...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.loop_detection", "1"
    ])
    
    # Run mapper
    print("Running mapper...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir
    ])

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP feature extraction and matching.")
    parser.add_argument('--object', required=True, help='Object ID (e.g., ob_0000001)')
    parser.add_argument('--sample_every', type=int, default=5, help='Sample every nth image from rgb (default: 5)')
    args = parser.parse_args()
    
    run_colmap_feature_extraction(args.object, sample_every=args.sample_every)

if __name__ == "__main__":
    main() 