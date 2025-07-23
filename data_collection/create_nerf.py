#!/usr/bin/env python3
"""
NeRF data collection and processing script using nerfstudio.

This script will:
1. Prepare Record3D data for nerfstudio
2. Train a NeRF model
3. Export the trained model
"""

import subprocess
import sys
import os
import argparse
import time
import json
from pathlib import Path

def get_object_path(object_name):
    """
    Get the path to the object folder
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        str: Path to the object directory, or None if not found
    """
    # Get the parent directory of data_collection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Construct the path to the object's data directory
    object_path = os.path.join(parent_dir, "data", "object_sdf_data", object_name)
    
    if not os.path.exists(object_path):
        print(f"⚠️  No data directory found for object: {object_name}")
        return None
    
    print(f"📁 Object path: {object_path}")
    return object_path

def run_data_preparation(object_name, debug_mode=False):
    """
    Prepare Record3D data for nerfstudio using ns-process-data
    
    Args:
        object_name (str): Name of the object to process
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"📦 Starting data preparation for object: {object_name}")
    print(f"🐍 Using conda environment: nerfstudio")
    
    # Get the object path
    object_path = get_object_path(object_name)
    if not object_path:
        print(f"❌ Could not find object path for: {object_name}")
        return False
    
    # Standard directory structure for each object
    exr_rgbd_path = os.path.join(object_path, "EXR_RGBD")
    lidar_path = os.path.join(object_path, "lidar")
    
    if not os.path.exists(exr_rgbd_path):
        print(f"❌ EXR_RGBD directory not found: {exr_rgbd_path}")
        return False
    
    print(f"📁 Using EXR_RGBD data: {exr_rgbd_path}")
    print(f"📁 Using lidar data: {lidar_path}")
    
    # Create output directory for processed data
    processed_data_path = os.path.join(object_path, "nerf_data")
    os.makedirs(processed_data_path, exist_ok=True)
    
    print(f"📁 Output directory: {processed_data_path}")
    
    # Build the command for ns-process-data with explicit conda activation
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-process-data", "record3d",
        "--data", exr_rgbd_path,
        "--ply-dir", lidar_path,
        "--output-dir", processed_data_path,
        "--max_dataset_size", "600"
    ]
    
    if debug_mode:
        print(f"🐛 Debug mode enabled - will show more verbose output")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the data preparation script
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from data_collection directory
            check=True,
            stdout=None,  # Print to terminal
            stderr=None,  # Print to terminal
            text=True
        )
        
        print(f"✅ Data preparation completed successfully")
        print(f"📁 Processed data saved to: {processed_data_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Data preparation interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during data preparation: {e}")
        return False

def run_training(object_name, debug_mode=False):
    """
    Train a NeRF model using nerfacto
    
    Args:
        object_name (str): Name of the object to train
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎯 Starting NeRF training for object: {object_name}")
    print(f"🐍 Using conda environment: nerfstudio")
    
    # Get the object path
    object_path = get_object_path(object_name)
    if not object_path:
        print(f"❌ Could not find object path for: {object_name}")
        return False
    
    # Get the processed data path (output from previous step)
    processed_data_path = os.path.join(object_path, "nerf_data")
    
    if not os.path.exists(processed_data_path):
        print(f"❌ Processed data directory not found: {processed_data_path}")
        print(f"❌ Please run data preparation first with --prepare")
        return False
    
    # Create output directory for training results
    
    
    print(f"📁 Using processed data: {processed_data_path}")
    print("To view the nerfstudio training, run the following commands:")
    print("From mac: ssh -L 7007:localhost:7007 maria@141.212.84.141")
    print("Type into web browser: http://localhost:7007")

    

    # Build the command for ns-train
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-train", "nerfacto",
        "--data", processed_data_path,
        "--pipeline.model.predict-normals", "True",
        "--viewer.quit_on_train_completion", "True",
        "--output-dir", object_path
    ]
    
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the training script with real-time output
        working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Running from directory: {working_dir}")
        
        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='')  # print as it comes, avoid extra newlines
        
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ NeRF training completed successfully")
            return True
        else:
            print(f"❌ NeRF training failed with return code: {process.returncode}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ NeRF training failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  NeRF training interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during NeRF training: {e}")
        return False

def run_mesh_export(object_name, debug_mode=False):
    """
    Export mesh using Poisson surface reconstruction
    
    Args:
        object_name (str): Name of the object to export
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"📦 Starting mesh export for object: {object_name}")
    print(f"🐍 Using conda environment: nerfstudio")
    
    # Get the object path
    object_path = get_object_path(object_name)
    if not object_path:
        print(f"❌ Could not find object path for: {object_name}")
        return False
    
    # Create output directory for mesh
    mesh_output_path = os.path.join(object_path, "mesh_export")
    os.makedirs(mesh_output_path, exist_ok=True)
    
    print(f"📁 Mesh output directory: {mesh_output_path}")
    
    # Find the latest config file from {object_path}/nerf_data/nerfacto
    outputs_dir = os.path.join(object_path, "nerf_data", "nerfacto")
    
    if not os.path.exists(outputs_dir):
        print(f"❌ Training outputs directory not found: {outputs_dir}")
        print(f"❌ Please run training first with --train")
        return False
    
    # Find the latest timestamp directory
    timestamp_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    if not timestamp_dirs:
        print(f"❌ No training runs found in: {outputs_dir}")
        return False
    
    # Sort by timestamp (newest first) and get the latest
    latest_timestamp = sorted(timestamp_dirs, reverse=True)[0]
    config_path = os.path.join(outputs_dir, latest_timestamp, "config.yml")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"📁 Using config: {config_path}")

    # Build the command for ns-export
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export", "poisson",
        "--load-config", config_path,
        "--output-dir", mesh_output_path,
        "--target-num-faces", "30000",
        "--num-pixels-per-side", "1024",
        "--num-points", "500000",
        "--remove-outliers", "True",
        "--normal-method", "open3d",
        "--obb_center", "0.0000000000", "0.0000000000", "0.0000000000",
        "--obb_rotation", "0.0000000000", "0.0000000000", "0.0000000000",
        "--obb_scale", "0.5", "0.5", "1"
    ]
    
    if debug_mode:
        print(f"🐛 Debug mode enabled - will show more verbose output")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the mesh export script
        working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Running from directory: {working_dir}")
        
        result = subprocess.run(
            cmd,
            cwd=working_dir,  # Run from parent directory
            check=True,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        
        print(f"✅ Mesh export completed successfully")
        print(f"📁 Mesh saved to: {mesh_output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Mesh export failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Mesh export interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during mesh export: {e}")
        return False

def run_overlay(object_name, debug_mode=False):
    """
    Run the point cloud overlay visualization
    
    Args:
        object_name (str): Name of the object to overlay
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎨 Starting overlay visualization for object: {object_name}")
    print(f"🐍 Using conda environment: open3d")
    
    # Build the command to run pc_overlay.py in the open3d environment
    cmd = [
        "conda", "run", "-n", "open3d",
        "python", "pc_overlay.py", object_name
    ]
    
    if debug_mode:
        print(f"🐛 Debug mode enabled - will show more verbose output")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the overlay script in open3d environment
        working_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Running from directory: {working_dir}")
        
        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='')  # print as it comes, avoid extra newlines
        
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ Overlay visualization completed successfully")
            return True
        else:
            print(f"❌ Overlay visualization failed with return code: {process.returncode}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Overlay visualization failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Overlay visualization interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during overlay visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'object', 
        type=str, 
        help='Name of the object to process',
        metavar='OBJECT'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Prepare Record3D data for nerfstudio'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train NeRF model using nerfacto'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export mesh using Poisson surface reconstruction'
    )
    
    parser.add_argument(
        '--overlay',
        action='store_true',
        help='Run point cloud overlay visualization'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting NeRF data collection and processing pipeline")
    print(f"📦 Object: {args.object}")
    print(f"🐛 Debug mode: {args.debug}")
    print(f"📦 Prepare: {args.prepare}")
    print(f"🎯 Train: {args.train}")
    print(f"📦 Export: {args.export}")
    print(f"🎨 Overlay: {args.overlay}")
    print("-" * 50)
    
    # Step 1: Data Preparation
    if args.prepare:
        print("\n📦 STEP 1: Data Preparation")
        print("=" * 30)
        
        success = run_data_preparation(args.object, args.debug)
        if not success:
            print("❌ Data preparation failed. Exiting.")
            sys.exit(1)
        
        print("✅ Data preparation completed successfully!")
    
    # Step 2: Training
    if args.train:
        print("\n🎯 STEP 2: NeRF Training")
        print("=" * 30)
        
        success = run_training(args.object, args.debug)
        if not success:
            print("❌ NeRF training failed. Exiting.")
            sys.exit(1)
        
        print("✅ NeRF training completed successfully!")
    
    # Step 3: Mesh Export
    if args.export:
        print("\n📦 STEP 3: Mesh Export")
        print("=" * 30)
        
        success = run_mesh_export(args.object, args.debug)
        if not success:
            print("❌ Mesh export failed. Exiting.")
            sys.exit(1)
        
        print("✅ Mesh export completed successfully!")
    
    # Step 4: Overlay Visualization
    if args.overlay:
        print("\n🎨 STEP 4: Point Cloud Overlay")
        print("=" * 30)
        
        success = run_overlay(args.object, args.debug)
        if not success:
            print("❌ Overlay visualization failed. Exiting.")
            sys.exit(1)
        
        print("✅ Overlay visualization completed successfully!")
    
    # If no specific steps were requested, run data preparation
    if not args.prepare and not args.train and not args.export and not args.overlay:
        print("\n🚀 Running data preparation (no specific steps requested)")
        print("=" * 50)
        
        success = run_data_preparation(args.object, args.debug)
        if not success:
            print("❌ Data preparation failed. Exiting.")
            sys.exit(1)
        
        print("✅ Data preparation completed successfully!")
    
    print("\n🎉 Pipeline completed!")
    
    # Get the object path for final output
    object_path = get_object_path(args.object)
    if object_path:
        processed_data_path = os.path.join(object_path, "nerf_data")
        print(f"📂 Processed data available in: {processed_data_path}")

if __name__ == "__main__":
    main() 