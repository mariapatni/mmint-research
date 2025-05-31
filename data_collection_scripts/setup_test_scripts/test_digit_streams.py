import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import threading
import time
from queue import Queue
from camera_classes import setup_digits

def create_grid_layout(frames, num_devices):
    """Create a grid layout of DIGIT frames"""
    if not frames:
        return None
        
    # Calculate grid dimensions
    rows = (num_devices + 1) // 2
    cols = 2
    
    # Get frame dimensions from the first frame
    first_frame = next(iter(frames))[1]
    h, w = first_frame.shape[:2]
    
    # Create empty grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # Place frames in grid
    for idx, (serial, frame) in enumerate(frames):
        i, j = idx // cols, idx % cols
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = frame
    
    return grid

def main():
    # Setup all DIGIT devices
    connected_digits = setup_digits()
    
    if not connected_digits:
        print("No DIGIT devices could be connected!")
        return
    
    print(f"\nSuccessfully connected to {len(connected_digits)} DIGIT devices")
    print("Press 'q' to quit")
    
    # Create a stop event to signal all threads
    stop_event = threading.Event()
    
    # Create a queue for frames from all devices
    frame_queue = Queue()
    
    # Start streaming for all cameras
    for camera in connected_digits:
        camera.start_streaming(frame_queue, stop_event)
    
    # Create single window for all streams
    cv2.namedWindow('DIGIT Streams', cv2.WINDOW_NORMAL)
    # Set initial window size
    rows = (len(connected_digits) + 1) // 2
    cv2.resizeWindow('DIGIT Streams', 640, 240 * rows)
    
    # Dictionary to store latest frames from each device
    latest_frames = {}
    
    try:
        # Main display loop
        while not stop_event.is_set():
            # Get frames from queue
            try:
                while not frame_queue.empty():
                    device_type, serial, frame = frame_queue.get_nowait()
                    latest_frames[serial] = frame
            except Exception as e:
                print(f"Error getting frames from queue: {e}")
            
            # Create and display grid if we have frames
            if latest_frames:
                # Sort frames by serial number
                sorted_frames = sorted(latest_frames.items())
                grid = create_grid_layout(sorted_frames, len(connected_digits))
                if grid is not None:
                    cv2.imshow('DIGIT Streams', grid)
            
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
        for camera in reversed(connected_digits):
            try:
                camera.stop()
            except Exception as e:
                print(f"Error stopping camera {camera.serial}: {e}")

if __name__ == "__main__":
    main() 