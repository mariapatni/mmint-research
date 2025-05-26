import cv2
import numpy as np
import threading
import time
from queue import Queue
from camera_classes import setup_digits, setup_realsense

def create_combined_grid(digit_frames, realsense_frames):
    """Create a combined grid layout with DIGIT and RealSense frames"""
    if not digit_frames and not realsense_frames:
        return None
        
    # Get frame dimensions
    digit_frame = next(iter(digit_frames.values())) if digit_frames else None
    rs_frame = next(iter(realsense_frames.values())) if realsense_frames else None
    
    # If we have no frames, return None
    if digit_frame is None and rs_frame is None:
        return None
        
    # Get dimensions
    digit_h, digit_w = digit_frame.shape[:2] if digit_frame is not None else (0, 0)
    rs_h, rs_w = rs_frame['color'].shape[:2] if rs_frame is not None else (0, 0)
    
    # Calculate total dimensions
    # Width: max of (2 DIGIT frames side by side) or (3 RealSense frames)
    total_width = max(digit_w * 2, rs_w * 3)
    # Height: DIGIT frames + RealSense frames (2 rows)
    total_height = digit_h + rs_h * 2
    
    # Create empty grid with white background
    grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place DIGIT frames at the top
    if digit_frames:
        digit_frames_list = sorted(digit_frames.items())
        if len(digit_frames_list) >= 1:
            # First DIGIT (left)
            left_x = (total_width // 2 - digit_w) // 2
            grid[0:digit_h, left_x:left_x+digit_w] = digit_frames_list[0][1]
        if len(digit_frames_list) >= 2:
            # Second DIGIT (right)
            right_x = total_width - (total_width // 2 - digit_w) // 2 - digit_w
            grid[0:digit_h, right_x:right_x+digit_w] = digit_frames_list[1][1]
    
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
                    filled_positions.add(j)
                    break  # Move to the next position
    
    return grid

def main():
    # Setup all cameras
    connected_digits = setup_digits()
    connected_realsense = setup_realsense()
    
    if not connected_digits and not connected_realsense:
        print("No cameras could be connected!")
        return
    
    print(f"Connected DIGIT cameras: {[cam.serial for cam in connected_digits]}")
    print(f"Connected RealSense cameras: {[cam.serial for cam in connected_realsense]}")
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
    
    try:
        # Main display loop
        while not stop_event.is_set():
            # Get frames from queue
            try:
                while not frame_queue.empty():
                    device_type, serial, frame = frame_queue.get_nowait()
                    
                    if device_type.lower() == 'realsense':
                        color_frame = frame['color']
                        depth_frame = frame['depth']
                        
                        if isinstance(frame, dict) and 'color' in frame and 'depth' in frame:
                            latest_realsense_frames[serial] = frame
                        else:
                            print(f"Invalid RealSense frame format: {frame.keys() if isinstance(frame, dict) else 'Not a dict'}")
                    elif device_type.lower() == 'digit':
                        if isinstance(frame, np.ndarray):
                            latest_digit_frames[serial] = frame
                        else:
                            print(f"Invalid DIGIT frame type: {type(frame)}")
            except Exception as e:
                print(f"Error getting frames from queue: {e}")
                import traceback
                traceback.print_exc()  # Keep this for debugging errors
            
            # Create and display combined grid
            grid = create_combined_grid(latest_digit_frames, latest_realsense_frames)
            if grid is not None:
                cv2.imshow(window_name, grid)
            else:
                print("No grid to display - no frames available")
            
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
