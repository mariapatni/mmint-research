import cv2
import numpy as np
import threading
import time
from queue import Queue
import sys
import os

# Add parent directory to path to import camera_classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_classes import setup_realsense

def create_grid_layout(frames, num_cameras):
    """Create a grid layout of RealSense frames"""
    if not frames:
        return None
        
    # For 3 cameras, we want two rows (color and depth)
    rows, cols = 2, 3
    
    # Get frame dimensions from the first frame's color image
    h, w = frames[0][1]['color'].shape[:2]
    
    # Create empty grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # Define the order of cameras (left, center, right)
    camera_order = {
        "137322077775": 0,  # Left
        "218622278343": 0,  # Left (D405)
        "250122077836": 1,  # Center
        "332322072522": 2   # Right
    }
    
    # Place frames in grid according to predefined order
    for serial, frame_dict in frames:
        if serial in camera_order:
            j = camera_order[serial]
            # Place color frame in top row
            grid[0:h, j*w:(j+1)*w] = frame_dict['color']
            # Place depth frame in bottom row
            grid[h:h*2, j*w:(j+1)*w] = frame_dict['depth']
    
    return grid

def main():
    # Setup RealSense cameras
    connected_cameras = setup_realsense()
    
    if not connected_cameras:
        print("No RealSense cameras could be connected!")
        return
    
    print(f"Connected RealSense cameras: {[cam.serial for cam in connected_cameras]}")
    print("\nStreaming started. Press 'q' to quit")
    
    # Create a stop event to signal all threads
    stop_event = threading.Event()
    
    # Create queue for frames
    frame_queue = Queue()
    
    # Start streaming for all cameras
    for camera in connected_cameras:
        camera.start_streaming(frame_queue, stop_event)
    
    # Create window for all streams
    cv2.namedWindow('All RealSense Streams', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('All RealSense Streams', 1920, 1080)  # Set initial window size
    
    # Dictionary to store latest frames from each camera
    latest_frames = {}
    
    try:
        # Main display loop
        while not stop_event.is_set():
            # Get frames from queue
            try:
                while not frame_queue.empty():
                    device_type, serial, frame = frame_queue.get_nowait()
                    print(f"Received frame from RealSense camera {serial}")  # Debug print
                    latest_frames[serial] = frame
                    print(f"Stored frames for RealSense camera {serial}")  # Debug print
            except Exception as e:
                print(f"Error getting frames from queue: {e}")
            
            # Print number of frames available
            print(f"Number of RealSense cameras with frames: {len(latest_frames)}")
            
            # Print frame dimensions if available
            if latest_frames:
                first_rs = next(iter(latest_frames.values()))
                if isinstance(first_rs, dict) and 'color' in first_rs and 'depth' in first_rs:
                    print(f"RealSense color frame dimensions: {first_rs['color'].shape}")
                    print(f"RealSense depth frame dimensions: {first_rs['depth'].shape}")
                else:
                    print("Invalid frame format - expected dictionary with 'color' and 'depth' keys")
            
            # Create and display grid if we have frames
            if latest_frames:
                # Sort frames by serial number for consistent ordering
                sorted_frames = sorted(latest_frames.items())
                print("latest frames length: ", len(sorted_frames))
                grid = create_grid_layout(sorted_frames, len(connected_cameras))
                if grid is not None:
                    print(f"Displaying grid with dimensions: {grid.shape}")  # Debug print
                    cv2.imshow('All RealSense Streams', grid)
                else:
                    print("No grid to display - invalid frame format")  # Debug print
            
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
        for camera in reversed(connected_cameras):
            try:
                camera.stop()
            except Exception as e:
                print(f"Error stopping camera {camera.serial}: {e}")

if __name__ == "__main__":
    main()
