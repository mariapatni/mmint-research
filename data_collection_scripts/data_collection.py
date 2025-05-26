import cv2
import numpy as np
import threading
import time
from queue import Queue
import os
from datetime import datetime
from camera_classes import setup_digits, setup_realsense
import argparse
import open3d as o3d  # Add open3d for point cloud processing

def debug_print(message, debug_mode=False):
    """Print debug messages only when debug mode is enabled"""
    if debug_mode:
        print(message)

def get_next_trial_number(object_dir):
    """Get the next trial number based on existing directories"""
    if not os.path.exists(object_dir):
        return 1
        
    # Get all trial directories
    existing_trials = [d for d in os.listdir(object_dir) 
                      if os.path.isdir(os.path.join(object_dir, d)) 
                      and d.startswith('trial')]
    
    if not existing_trials:
        return 1
        
    # Extract trial numbers and find the maximum
    trial_numbers = [int(trial.replace('trial', '')) for trial in existing_trials]
    return max(trial_numbers) + 1

def create_output_directories(object_name):
    """Create directories for saving frames"""
    # Get the parent directory of data_collection_scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create collected_data directory under data
    collected_data_dir = os.path.join(data_dir, "collected_data")
    os.makedirs(collected_data_dir, exist_ok=True)
    
    # Create object_name directory under collected_data
    object_dir = os.path.join(collected_data_dir, object_name)
    os.makedirs(object_dir, exist_ok=True)
    
    # Get next trial number
    trial_number = get_next_trial_number(object_dir)
    trial_dir = os.path.join(object_dir, f"trial{trial_number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Create camera directories
    # Create directories for RealSense cameras
    realsense_mapping = {
        "250122077836": "side_camera_1",  # Left camera
        "137322077775": "gripper_camera",  # D435 camera
        "218622278343": "gripper_camera",  # D405 camera - will override D435 if both connected
        "332322072522": "side_camera_2"   # Right camera
    }
    
    for camera_name in realsense_mapping.values():
        camera_dir = os.path.join(trial_dir, camera_name)
        os.makedirs(camera_dir, exist_ok=True)
        os.makedirs(os.path.join(camera_dir, "color"), exist_ok=True)
        os.makedirs(os.path.join(camera_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(camera_dir, "point_cloud"), exist_ok=True)  # Add point cloud directory
    
    # Create directories for DIGIT cameras
    digit_mapping = {
        "D20722": "left_digit",
        "D21202": "right_digit"
    }
    
    for camera_name in digit_mapping.values():
        camera_dir = os.path.join(trial_dir, camera_name)
        os.makedirs(camera_dir, exist_ok=True)
    
    return object_dir, trial_number

def save_frame(output_dir, device_type, serial, frame, frame_count, trial_number, debug_mode=False):
    """Save a frame to the appropriate directory"""
    # Convert device_type to lowercase for case-insensitive comparison
    device_type = device_type.lower()
    
    if device_type == 'digit':
        # Map DIGIT serial numbers to directory names
        digit_mapping = {
            "D20722": "left_digit",
            "D21202": "right_digit"
        }
        
        if serial not in digit_mapping:
            debug_print(f"Warning: Unknown DIGIT serial number {serial}", debug_mode)
            return
        
        # Save DIGIT frame
        filename = f"frame_{frame_count:06d}.png"
        save_path = os.path.join(output_dir, f"trial{trial_number}", digit_mapping[serial], filename)
        debug_print(f"Saving DIGIT frame to: {save_path}", debug_mode)
        cv2.imwrite(save_path, frame)
        
    elif device_type == 'realsense':
        # Map RealSense serial numbers to directory names
        realsense_mapping = {
            "250122077836": "side_camera_1",  # Left camera
            "137322077775": "gripper_camera",  # D435 camera
            "218622278343": "gripper_camera",  # D405 camera
            "332322072522": "side_camera_2"   # Right camera
        }
        
        if serial not in realsense_mapping:
            debug_print(f"Warning: Unknown RealSense serial number {serial}", debug_mode)
            return
            
        # Save RealSense frames (both color and depth)
        base_dir = os.path.join(output_dir, f"trial{trial_number}", realsense_mapping[serial])
        
        # Save color frame
        color_filename = f"frame_{frame_count:06d}.png"
        color_path = os.path.join(base_dir, "color", color_filename)
        debug_print(f"Saving RealSense color frame to: {color_path}", debug_mode)
        cv2.imwrite(color_path, frame['color'])
        
        # Save depth frame
        depth_filename = f"frame_{frame_count:06d}.png"
        depth_path = os.path.join(base_dir, "depth", depth_filename)
        debug_print(f"Saving RealSense depth frame to: {depth_path}", debug_mode)
        cv2.imwrite(depth_path, frame['depth'])
    else:
        debug_print(f"Warning: Unknown device type {device_type}", debug_mode)
        return

def create_combined_grid(digit_frames, realsense_frames, debug_mode=False):
    """Create a combined grid layout with DIGIT and RealSense frames"""
    if not digit_frames and not realsense_frames:
        debug_print("No frames available for grid creation", debug_mode)
        return None
        
    # Get frame dimensions
    digit_frame = next(iter(digit_frames.values())) if digit_frames else None
    rs_frame = next(iter(realsense_frames.values())) if realsense_frames else None
    
    # If we have no frames, return None
    if digit_frame is None and rs_frame is None:
        debug_print("No valid frames found", debug_mode)
        return None
        
    # Get dimensions
    digit_h, digit_w = digit_frame.shape[:2] if digit_frame is not None else (0, 0)
    rs_h, rs_w = rs_frame['color'].shape[:2] if rs_frame is not None else (0, 0)
    
    debug_print(f"DIGIT dimensions: {digit_h}x{digit_w}", debug_mode)
    debug_print(f"RealSense dimensions: {rs_h}x{rs_w}", debug_mode)
    
    # Calculate total dimensions
    # Width: max of (2 DIGIT frames side by side) or (3 RealSense frames)
    total_width = max(digit_w * 2, rs_w * 3)
    # Height: DIGIT frames + RealSense frames (2 rows)
    total_height = digit_h + rs_h * 2
    
    debug_print(f"Creating combined grid with dimensions: {total_width}x{total_height}", debug_mode)
    
    # Create empty grid with white background
    grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place DIGIT frames at the top
    if digit_frames:
        digit_frames_list = sorted(digit_frames.items())
        if len(digit_frames_list) >= 1:
            # First DIGIT (left)
            left_x = (total_width // 2 - digit_w) // 2
            grid[0:digit_h, left_x:left_x+digit_w] = digit_frames_list[0][1]
            debug_print(f"Placed first DIGIT frame from {digit_frames_list[0][0]}", debug_mode)
        if len(digit_frames_list) >= 2:
            # Second DIGIT (right)
            right_x = total_width - (total_width // 2 - digit_w) // 2 - digit_w
            grid[0:digit_h, right_x:right_x+digit_w] = digit_frames_list[1][1]
            debug_print(f"Placed second DIGIT frame from {digit_frames_list[1][0]}", debug_mode)
    
    # Place RealSense frames below DIGIT frames
    if realsense_frames:
        start_y = digit_h  # Start after DIGIT frames
        
        # Define the order of cameras
        # Priority order for which camera gets which position when multiple are present
        camera_positions = {
            0: ["218622278343", "250122077836"],  # Left position: prefer D405, then 250122077836
            1: ["250122077836", "137322077775"],  # Center position: prefer 250122077836, then 137322077775
            2: ["137322077775", "332322072522"]   # Right position: prefer 137322077775, then 332322072522
        }
        
        # Track which positions are filled
        filled_positions = set()
        
        # First, check for the D405 camera and place it with highest priority
        if "218622278343" in realsense_frames:
            # Place D405 in position 0 (left)
            j = 0
            filled_positions.add(j)
            frame_dict = realsense_frames["218622278343"]
            # Place color frame in top row
            grid[start_y:start_y+rs_h, j*rs_w:(j+1)*rs_w] = frame_dict['color']
            # Place depth frame in bottom row
            grid[start_y+rs_h:start_y+rs_h*2, j*rs_w:(j+1)*rs_w] = frame_dict['depth']
            debug_print(f"Placed D405 RealSense frame at position {j}", debug_mode)
            
        # Then place other cameras based on priority
        for j, camera_list in camera_positions.items():
            if j in filled_positions:
                continue  # Skip already filled positions
                
            for serial in camera_list:
                if serial in realsense_frames and serial != "218622278343":  # Skip D405 as it's already placed
                    frame_dict = realsense_frames[serial]
                    # Place color frame in top row
                    grid[start_y:start_y+rs_h, j*rs_w:(j+1)*rs_w] = frame_dict['color']
                    # Place depth frame in bottom row
                    grid[start_y+rs_h:start_y+rs_h*2, j*rs_w:(j+1)*rs_w] = frame_dict['depth']
                    debug_print(f"Placed RealSense frame from {serial} at position {j}", debug_mode)
                    filled_positions.add(j)
                    break  # Move to the next position
    
    return grid

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect data from DIGIT and RealSense cameras')
    parser.add_argument('--object', type=str, required=True,
                      help='Name of the object being recorded')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with verbose logging')
    args = parser.parse_args()
    
    # Setup all cameras
    connected_digits = setup_digits()
    connected_realsense = setup_realsense()
    
    if not connected_digits and not connected_realsense:
        print("No cameras could be connected!")
        return
    
    print(f"Connected DIGIT cameras: {[cam.serial for cam in connected_digits]}")
    print(f"Connected RealSense cameras: {[cam.serial for cam in connected_realsense]}")
    
    # Create output directories
    output_dir, trial_number = create_output_directories(args.object)
    print(f"\nSaving frames to: {output_dir}/trial{trial_number}")
    
    print("\nStreaming started. Press 'q' to quit")
    
    # Create a stop event to signal all threads
    stop_event = threading.Event()
    
    # Create queue for frames
    frame_queue = Queue()
    
    # Start streaming for all cameras
    for camera in connected_digits + connected_realsense:
        camera.start_streaming(frame_queue, stop_event)
    
    # Create single window for all streams
    window_name = 'All Camera Streams'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)
    
    # Dictionaries to store latest frames from each camera
    latest_realsense_frames = {}
    latest_digit_frames = {}
    
    # Frame counter
    frame_count = 0
    
    # Status update frequency (frames)
    status_update_frequency = 50
    
    try:
        # Main display loop
        while not stop_event.is_set():
            # Get frames from queue
            try:
                while not frame_queue.empty():
                    device_type, serial, frame = frame_queue.get_nowait()
                    debug_print(f"Received frame from {device_type} camera {serial}", args.debug)
                    debug_print(f"Frame type: {type(frame)}", args.debug)
                    
                    if device_type.lower() == 'realsense':
                        if isinstance(frame, dict) and 'color' in frame and 'depth' in frame:
                            latest_realsense_frames[serial] = frame
                            debug_print(f"Stored RealSense frame from {serial} with color shape {frame['color'].shape}", args.debug)
                        else:
                            debug_print(f"Invalid RealSense frame format: {frame.keys() if isinstance(frame, dict) else 'Not a dict'}", args.debug)
                    elif device_type.lower() == 'digit':
                        if isinstance(frame, np.ndarray):
                            latest_digit_frames[serial] = frame
                            debug_print(f"Stored DIGIT frame from {serial} with shape {frame.shape}", args.debug)
                        else:
                            debug_print(f"Invalid DIGIT frame type: {type(frame)}", args.debug)
            except Exception as e:
                print(f"Error getting frames from queue: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()  # Print full stack trace
            
            # Print number of frames available
            debug_print(f"Number of RealSense cameras with frames: {len(latest_realsense_frames)}", args.debug)
            debug_print(f"Number of DIGIT cameras with frames: {len(latest_digit_frames)}", args.debug)
            
            # Create and display combined grid
            grid = create_combined_grid(latest_digit_frames, latest_realsense_frames, args.debug)
            if grid is not None:
                debug_print(f"Displaying combined grid with dimensions: {grid.shape}", args.debug)
                cv2.imshow(window_name, grid)
            else:
                debug_print("No grid to display - no frames available", args.debug)
            
            # Save frames
            debug_print(f"Number of DIGIT frames to save: {len(latest_digit_frames)}", args.debug)
            debug_print(f"Number of RealSense frames to save: {len(latest_realsense_frames)}", args.debug)
            
            # Save DIGIT frames
            for serial, frame in latest_digit_frames.items():
                debug_print(f"Attempting to save DIGIT frame from camera {serial}", args.debug)
                save_frame(output_dir, 'digit', serial, frame, frame_count, trial_number, args.debug)
            
            # Save RealSense frames
            for serial, frame in latest_realsense_frames.items():
                debug_print(f"Attempting to save RealSense frame from camera {serial}", args.debug)
                save_frame(output_dir, 'realsense', serial, frame, frame_count, trial_number, args.debug)
            
            # Increment frame counter and show status periodically
            frame_count += 1
            if frame_count % status_update_frequency == 0:
                print(f"Processed {frame_count} frames")
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received in main thread")
                stop_event.set()
                break
            
            time.sleep(0.01)  # Small sleep to prevent high CPU usage
            
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        
        # Stop all cameras in reverse order
        for camera in reversed(connected_digits + connected_realsense):
            try:
                camera.stop()
            except Exception as e:
                print(f"Error stopping camera {camera.serial}: {e}")

if __name__ == "__main__":
    main()
 