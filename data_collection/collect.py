#!/usr/bin/env python3
"""
Combined data collection, segmentation, and pose tracking script.

This script will:
1. Run data_collection_v2.py for the specified object
2. Segment object masks (to be implemented)
3. Apply pose tracking via FoundationPose++ (to be implemented)
"""

import subprocess
import sys
import os
import argparse
import time
import json
from pathlib import Path

def run_data_collection(object_name, debug_mode=False):
    """
    Run data_collection_v2.py for the specified object in 'pose' environment
    
    Args:
        object_name (str): Name of the object to collect data for
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎬 Starting data collection for object: {object_name}")
    print(f"🐍 Using conda environment: pose")
    
    # Build the command with conda environment activation
    cmd = [
        "conda", "run", "-n", "pose",
        "python", "data_collection_v2.py",
        "--object", object_name
    ]
    
    if debug_mode:
        cmd.append("--debug")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the data collection script in pose environment
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from data_collection directory
            check=True,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        
        print(f"✅ Data collection completed successfully for {object_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data collection failed for {object_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Data collection interrupted for {object_name}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during data collection for {object_name}: {e}")
        return False

def get_latest_trial_path(object_name):
    """
    Get the path to the latest trial for the given object
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        str: Path to the latest trial directory, or None if not found
    """
    # Get the parent directory of data_collection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Construct the path to the object's data directory
    object_dir = os.path.join(parent_dir, "data", "collected_data", object_name)
    
    if not os.path.exists(object_dir):
        print(f"⚠️  No data directory found for object: {object_name}")
        return None
    
    # Find the latest trial
    trial_dirs = [d for d in os.listdir(object_dir) 
                  if os.path.isdir(os.path.join(object_dir, d)) 
                  and d.startswith('trial')]
    
    if not trial_dirs:
        print(f"⚠️  No trial directories found for object: {object_name}")
        return None
    
    # Sort trials and get the latest
    trial_dirs.sort(key=lambda x: int(x.replace('trial', '')))
    latest_trial = trial_dirs[-1]
    
    trial_path = os.path.join(object_dir, latest_trial)
    print(f"📁 Latest trial path: {trial_path}")
    
    return trial_path

def run_mask_segmentation(trial_path, debug_mode=False):
    """
    Run segment_obj_masks.py to create object masks in 'tracking' environment
    
    Args:
        trial_path (str): Path to the trial directory containing RGB/depth data
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎭 Starting mask segmentation for trial: {trial_path}")
    print(f"🐍 Using conda environment: tracking")
    
    # Find the gripper camera directory (D405 camera)
    gripper_camera_path = os.path.join(trial_path, "gripper_camera")
    if not os.path.exists(gripper_camera_path):
        print(f"❌ Gripper camera directory not found: {gripper_camera_path}")
        return False
    
    # Check if RGB and depth directories exist
    rgb_path = os.path.join(gripper_camera_path, "rgb")
    depth_path = os.path.join(gripper_camera_path, "depth")
    
    if not os.path.exists(rgb_path):
        print(f"❌ RGB directory not found: {rgb_path}")
        return False
    
    if not os.path.exists(depth_path):
        print(f"❌ Depth directory not found: {depth_path}")
        return False
    
    # Build the command for segment_obj_masks.py in tracking environment
    # Note: segment_obj_masks.py expects --input_dir (not --rgb_dir)
    cmd = [
        "conda", "run", "-n", "tracking",
        "python", "./segment_obj_masks.py",
        "--input_dir", rgb_path,
        "--output_dir", gripper_camera_path
    ]
    
    if debug_mode:
        cmd.append("--debug")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the segmentation script in tracking environment
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from data_collection directory
            check=True,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        
        print(f"✅ Mask segmentation completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Mask segmentation failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Mask segmentation interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during mask segmentation: {e}")
        return False

def run_pose_tracking(trial_path, debug_mode=False):
    """
    Run FoundationPose++ pose tracking in 'pose' environment
    
    Args:
        trial_path (str): Path to the trial directory containing RGB/depth/mask data
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎯 Starting pose tracking for trial: {trial_path}")
    print(f"🐍 Using conda environment: pose")
    
    # Find the gripper camera directory (D405 camera)
    gripper_camera_path = os.path.join(trial_path, "gripper_camera")
    if not os.path.exists(gripper_camera_path):
        print(f"❌ Gripper camera directory not found: {gripper_camera_path}")
        return False
    
    # Check if required directories exist
    rgb_path = os.path.join(gripper_camera_path, "rgb")
    depth_path = os.path.join(gripper_camera_path, "depth")
    masks_path = os.path.join(gripper_camera_path, "masks")
    cam_K_path = os.path.join(gripper_camera_path, "cam_K.txt")
    
    if not os.path.exists(rgb_path):
        print(f"❌ RGB directory not found: {rgb_path}")
        return False
    
    if not os.path.exists(depth_path):
        print(f"❌ Depth directory not found: {depth_path}")
        return False
    
    if not os.path.exists(masks_path):
        print(f"❌ Masks directory not found: {masks_path}")
        return False
    
    if not os.path.exists(cam_K_path):
        print(f"❌ Camera intrinsics file not found: {cam_K_path}")
        return False
    
    # Read camera intrinsics
    try:
        with open(cam_K_path, 'r') as f:
            lines = f.readlines()
            if len(lines) != 3:
                print(f"❌ Invalid camera intrinsics file format: {cam_K_path}")
                return False
            
            # Parse the 3x3 camera matrix
            cam_K = []
            for line in lines:
                row = [float(x) for x in line.strip().split()]
                cam_K.append(row)
            
            cam_K_json = json.dumps(cam_K)
            print(f"📷 Camera intrinsics loaded: {cam_K}")
            
    except Exception as e:
        print(f"❌ Failed to read camera intrinsics: {e}")
        return False
    
    # Find the first mask file for initialization
    mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.png')])
    if not mask_files:
        print(f"❌ No mask files found in: {masks_path}")
        return False
    
    init_mask_path = os.path.join(masks_path, mask_files[0])
    print(f"🎭 Using initial mask: {init_mask_path}")
    
    # Create output directories for pose tracking
    pose_output_dir = gripper_camera_path
    pose_vis_dir = os.path.join(gripper_camera_path, "poses_vis")
    
    for dir_path in [pose_output_dir, pose_vis_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Find the object mesh file
    object_name = os.path.basename(os.path.dirname(trial_path))
    
    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(trial_path)))  # /home/maria/mmint-research/data
    mesh_path = os.path.join(data_dir, "object_sdf_data", object_name, "mesh_export", "mesh_scaled.obj")
    
    if not os.path.exists(mesh_path):
        print(f"❌ Mesh file not found: {mesh_path}")
        print(f"💡 Expected path: {mesh_path}")
        return False
    
    print(f"📦 Using mesh: {mesh_path}")
    
    # Build the command for FoundationPose++ pose tracking
    cmd = [
        "conda", "run", "-n", "pose",
        "python", "src/obj_pose_track.py",
        "--rgb_seq_path", rgb_path,
        "--depth_seq_path", depth_path,
        "--mesh_path", mesh_path,
        "--init_mask_path", init_mask_path,
        "--pose_output_path", os.path.join(pose_output_dir, "poses.npy"),
        "--mask_visualization_path", "None",
        "--bbox_visualization_path", "None",
        "--pose_visualization_path", pose_vis_dir,
        "--cam_K", cam_K_json,
        "--est_refine_iter", "10",
        "--track_refine_iter", "4",
        "--apply_scale", "1.0",
        "--activate_2d_tracker",
        "--activate_kalman_filter",
    ]
    
    if debug_mode:
        print(f"🐛 Debug mode enabled - will show more verbose output")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        # Run the pose tracking script in pose environment
        # Change to FoundationPose-plus-plus directory
        foundationpose_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FoundationPose-plus-plus")
        
        result = subprocess.run(
            cmd,
            cwd=foundationpose_dir,  # Run from FoundationPose-plus-plus directory
            check=True,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        
        print(f"✅ Pose tracking completed successfully")
        print(f"📁 Pose results saved to: {pose_output_dir}")
        print(f"🎨 Visualizations saved to: {pose_vis_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pose tracking failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Pose tracking interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during pose tracking: {e}")
        return False

def run_pose_visualization(trial_path, debug_mode=False):
    """
    Run visualize_poses.py to play back pose tracking results
    
    Args:
        trial_path (str): Path to the trial directory containing pose visualization data
        debug_mode (bool): Whether to run in debug mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎬 Starting pose visualization for trial: {trial_path}")
    
    # Find the pose visualization directory
    gripper_camera_path = os.path.join(trial_path, "gripper_camera")
    pose_vis_dir = os.path.join(gripper_camera_path, "poses_vis")
    
    if not os.path.exists(pose_vis_dir):
        print(f"❌ Pose visualization directory not found: {pose_vis_dir}")
        print("💡 Make sure pose tracking has been run first")
        return False
    
    # Check if there are any visualization frames
    vis_files = [f for f in os.listdir(pose_vis_dir) if f.endswith('.png')]
    if not vis_files:
        print(f"❌ No visualization frames found in: {pose_vis_dir}")
        return False
    
    print(f"📁 Found {len(vis_files)} visualization frames")
    
    # Build the command for visualize_poses.py
    cmd = [
        "python", "./visualize_poses.py",
        "--frames_dir", pose_vis_dir,
        "--fps", "30",
        "--window_name", f"Pose Visualization - {os.path.basename(trial_path)}"
    ]
    
    if debug_mode:
        print(f"🐛 Debug mode enabled")
    
    print(f"📋 Running command: {' '.join(cmd)}")
    print("🎮 Controls: Space=Pause, Arrows=Step, R=Reset, Q=Quit")
    
    try:
        # Run the visualization script
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from data_collection directory
            check=True,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        
        print(f"✅ Pose visualization completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pose visualization failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Pose visualization interrupted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during pose visualization: {e}")
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
        help='Enable debug mode for data collection'
    )
    
    parser.add_argument(
        '--collect',
        action='store_true',
        help='Run data collection step'
    )
    
    parser.add_argument(
        '--segment',
        action='store_true',
        help='Run mask segmentation step'
    )
    
    parser.add_argument(
        '--track',
        action='store_true',
        help='Run pose tracking step'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Play back pose visualization'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting combined data collection and processing pipeline")
    print(f"📦 Object: {args.object}")
    print(f"🐛 Debug mode: {args.debug}")
    print(f"📹 Collect: {args.collect}")
    print(f"🎭 Segment: {args.segment}")
    print(f"🎯 Track: {args.track}")
    print(f"🎬 Visualize: {args.visualize}")
    print("-" * 50)
    
    # Get the latest trial path (needed for most operations)
    trial_path = None
    if args.segment or args.track or args.visualize:
        trial_path = get_latest_trial_path(args.object)
        if not trial_path:
            print("❌ Could not find trial path. Exiting.")
            sys.exit(1)
        print(f"📁 Using latest trial: {trial_path}")
    
    # Step 1: Data Collection
    if args.collect:
        print("\n📹 STEP 1: Data Collection")
        print("=" * 30)
        
        success = run_data_collection(args.object, args.debug)
        if not success:
            print("❌ Data collection failed. Exiting.")
            sys.exit(1)
        
        print("✅ Data collection completed successfully!")
        
        # Update trial path after collection
        trial_path = get_latest_trial_path(args.object)
        if not trial_path:
            print("❌ Could not find trial path after collection. Exiting.")
            sys.exit(1)
        print(f"📁 New trial available at: {trial_path}")
    
    # Step 2: Mask Segmentation
    if args.segment:
        if not trial_path:
            print("❌ No trial path available for segmentation. Run --collect first or ensure trial exists.")
            sys.exit(1)
            
        print("\n🎭 STEP 2: Object Segmentation")
        print("=" * 30)
        
        success = run_mask_segmentation(trial_path, args.debug)
        if not success:
            print("❌ Mask segmentation failed. Exiting.")
            sys.exit(1)
        
        print("✅ Mask segmentation completed successfully!")
    
    # Step 3: Pose Tracking
    if args.track:
        if not trial_path:
            print("❌ No trial path available for tracking. Run --collect first or ensure trial exists.")
            sys.exit(1)
            
        print("\n🎯 STEP 3: Pose Tracking")
        print("=" * 30)
        
        success = run_pose_tracking(trial_path, args.debug)
        if not success:
            print("❌ Pose tracking failed. Exiting.")
            sys.exit(1)
        
        print("✅ Pose tracking completed successfully!")
    
    # Step 4: Pose Visualization
    if args.visualize:
        if not trial_path:
            print("❌ No trial path available for visualization. Run --collect first or ensure trial exists.")
            sys.exit(1)
            
        print("\n🎬 STEP 4: Pose Visualization")
        print("=" * 30)
        
        success = run_pose_visualization(trial_path, args.debug)
        if not success:
            print("❌ Pose visualization failed.")
        else:
            print("✅ Pose visualization completed successfully!")
    
    # If no specific steps were requested, run the full pipeline
    if not any([args.collect, args.segment, args.track, args.visualize]):
        print("\n🚀 Running full pipeline (no specific steps requested)")
        print("=" * 50)
        
        # Step 1: Data Collection
        print("\n📹 STEP 1: Data Collection")
        print("=" * 30)
        
        success = run_data_collection(args.object, args.debug)
        if not success:
            print("❌ Data collection failed. Exiting.")
            sys.exit(1)
        
        print("✅ Data collection completed successfully!")
        
        # Get the trial path
        trial_path = get_latest_trial_path(args.object)
        if not trial_path:
            print("❌ Could not find trial path. Exiting.")
            sys.exit(1)
        
        print(f"📁 Data available at: {trial_path}")
        
        # Step 2: Mask Segmentation
        print("\n🎭 STEP 2: Object Segmentation")
        print("=" * 30)
        
        success = run_mask_segmentation(trial_path, args.debug)
        if not success:
            print("❌ Mask segmentation failed. Exiting.")
            sys.exit(1)
        
        print("✅ Mask segmentation completed successfully!")
        
        # Step 3: Pose Tracking
        print("\n🎯 STEP 3: Pose Tracking")
        print("=" * 30)
        
        success = run_pose_tracking(trial_path, args.debug)
        if not success:
            print("❌ Pose tracking failed. Exiting.")
            sys.exit(1)
        
        print("✅ Pose tracking completed successfully!")
    
    print("\n🎉 Pipeline completed!")
    if trial_path:
        print(f"📂 Results available in: {trial_path}")

if __name__ == "__main__":
    main()
