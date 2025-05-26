import cv2
import numpy as np
import threading
import time
from queue import Queue
import os
import argparse
import pyrealsense2 as rs
import shutil  # Add shutil for directory removal

def debug_print(message, debug_mode=False):
    """Print debug messages only when debug mode is enabled"""
    if debug_mode:
        print(message)

def cleanup_directory(directory):
    """Remove directory and all its contents if it exists"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleaned up existing directory: {directory}")

def create_output_directories(object_name):
    """Create directories for saving frames"""
    # Get the parent directory of data_collection_scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create object_sdf_data directory
    sdf_dir = os.path.join(data_dir, "object_sdf_data")
    os.makedirs(sdf_dir, exist_ok=True)
    
    # Create object directory and clean it if it exists
    object_dir = os.path.join(sdf_dir, object_name)
    cleanup_directory(object_dir)
    os.makedirs(object_dir, exist_ok=True)
    
    # Create subdirectories for RGB, depth, and masks
    rgb_dir = os.path.join(object_dir, "rgb")
    depth_dir = os.path.join(object_dir, "depth")
    masks_dir = os.path.join(object_dir, "masks")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    return object_dir

def setup_realsense():
    """Setup RealSense camera (prefer D405, fallback to D435)"""
    ctx = rs.context()
    devices = list(ctx.query_devices())
    
    # Look for D405 first, then D435
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if serial == "218622278343":  # D405
            return setup_camera(serial)
        elif serial == "137322077775":  # D435
            return setup_camera(serial)
    
    return None

def setup_camera(serial):
    """Setup individual RealSense camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start streaming
    pipeline.start(config)
    
    # Get camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    
    # Save camera intrinsics
    K = np.array([
        [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
        [0, depth_intrinsics.fy, depth_intrinsics.ppy],
        [0, 0, 1]
    ])
    
    return pipeline, K

def save_frame(object_dir, frame_count, color_frame, depth_frame, debug_mode=False):
    """Save RGB and depth frames"""
    # Save RGB frame
    rgb_path = os.path.join(object_dir, "rgb", f"{frame_count:06d}.png")
    cv2.imwrite(rgb_path, color_frame)
    debug_print(f"Saved RGB frame to: {rgb_path}", debug_mode)
    
    # Save depth frame (as uint16)
    depth_path = os.path.join(object_dir, "depth", f"{frame_count:06d}.png")
    cv2.imwrite(depth_path, depth_frame)
    debug_print(f"Saved depth frame to: {depth_path}", debug_mode)

def save_camera_intrinsics(object_dir, K):
    """Save camera intrinsic matrix"""
    intrinsics_path = os.path.join(object_dir, "cam_K.txt")
    np.savetxt(intrinsics_path, K, fmt='%.6f')
    print(f"Saved camera intrinsics to: {intrinsics_path}")

def create_combined_view(color_image, depth_image):
    """Create a combined view of color and depth images side by side"""
    # Normalize depth image for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Resize images to same height if they're different
    h1, w1 = color_image.shape[:2]
    h2, w2 = depth_colormap.shape[:2]
    
    if h1 != h2:
        scale = h1 / h2
        depth_colormap = cv2.resize(depth_colormap, (int(w2 * scale), h1))
    
    # Combine images horizontally
    combined = np.hstack((color_image, depth_colormap))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Color", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Depth", (w1 + 10, 30), font, 1, (255, 255, 255), 2)
    
    return combined

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect data from gripper camera')
    parser.add_argument('--object', type=str, required=True,
                      help='Name of the object being recorded')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with verbose logging')
    args = parser.parse_args()
    
    # Setup RealSense camera
    camera_setup = setup_realsense()
    if camera_setup is None:
        print("No compatible RealSense camera found!")
        return
    
    pipeline, K = camera_setup
    
    # Create output directories
    object_dir = create_output_directories(args.object)
    print(f"\nSaving frames to: {object_dir}")
    
    # Save camera intrinsics
    save_camera_intrinsics(object_dir, K)
    
    print("\nStreaming started. Press 'q' to quit")
    
    # Frame counter
    frame_count = 0
    
    # Create window
    cv2.namedWindow('Camera Streams', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Streams', 1280, 480)  # Width for two 640x480 images
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Create combined view
            combined_view = create_combined_view(color_image, depth_image)
            
            # Display the combined view
            cv2.imshow('Camera Streams', combined_view)
            
            # Save frames
            save_frame(object_dir, frame_count, color_image, depth_image, args.debug)
            
            # Increment frame counter
            frame_count += 1
            
            # Print status every 50 frames
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received")
                break
            
            time.sleep(0.01)  # Small sleep to prevent high CPU usage
            
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main() 