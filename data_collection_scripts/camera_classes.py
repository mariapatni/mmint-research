import cv2
import numpy as np
import threading
import time
from queue import Queue
from digit_interface import DigitHandler
from digit_interface.digit import Digit
import pyrealsense2 as rs
import open3d as o3d

class Camera:
    """Base class for camera devices"""
    def __init__(self, serial):
        self.serial = serial
        self.frame_queue = None
        self.stop_event = None
        self.thread = None
        self.connected = False
        self.streaming = False

    def start_streaming(self, frame_queue, stop_event):
        """Start streaming in a separate thread"""
        if not self.connected:
            print(f"Cannot start streaming for unconnected device {self.serial}")
            return
            
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.streaming = True
        self.thread = threading.Thread(target=self._stream_loop)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.5)  # Small delay between starting threads

    def _stream_loop(self):
        """Stream loop to be implemented by subclasses"""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup to be implemented by subclasses"""
        raise NotImplementedError

    def stop(self):
        """Stop streaming and cleanup"""
        self.streaming = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cleanup()

class DigitCamera(Camera):
    """DIGIT camera implementation"""
    def __init__(self, serial):
        super().__init__(serial)
        self.device = None

    def connect(self):
        """Connect to the DIGIT device"""
        if self.connected:
            return True
            
        try:
            self.device = Digit(self.serial)
            self.device.connect()
            self.device.set_resolution(Digit.STREAMS["QVGA"])
            self.device.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to DIGIT {self.serial}: {e}")
            return False

    def _stream_loop(self):
        """Stream loop for DIGIT camera"""
        if not self.connected or not self.streaming:
            print(f"DIGIT {self.serial} not ready for streaming")
            return
            
        try:
            while not self.stop_event.is_set() and self.streaming:
                try:
                    frame = self.device.get_frame()
                    
                    # Put frame in queue for main thread to display
                    self.frame_queue.put(("DIGIT", self.serial, frame))
                        
                except Exception as e:
                    if self.stop_event.is_set():
                        break
                    print(f"Error getting frame from DIGIT {self.serial}: {e}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"Error in DIGIT {self.serial} stream: {e}")
        finally:
            self.streaming = False

    def cleanup(self):
        """Cleanup DIGIT connection"""
        if not self.connected:
            return
            
        print(f"Stopping DIGIT {self.serial}")
        try:
            if self.device:
                self.device.disconnect()
                self.connected = False
        except Exception as e:
            print(f"Error disconnecting DIGIT {self.serial}: {e}")

class RealSenseCamera(Camera):
    """RealSense camera implementation"""
    _shared_context = None  # Class-level shared context
    _connected_serials = set()  # Track connected serial numbers
    
    def __init__(self, serial):
        super().__init__(serial)
        self.pipeline = None
        self.config = None
        self.device = None
        self.connected = False
        self.streaming = False
        self._frame_timeout = 5000  # 5 seconds timeout
        self._last_frame_time = 0
        self._frame_interval = 1.0/30# 30PS
        self.align = None  # For aligning depth and color frames

    @classmethod
    def get_shared_context(cls):
        """Get or create the shared RealSense context"""
        if cls._shared_context is None:
            cls._shared_context = rs.context()
        return cls._shared_context

    def connect(self):
        """Connect to the RealSense device"""
        if self.connected:
            return True
            
        print(f"CONNECTING TO {self.serial}")
        
        # Check if this serial number is already connected
        if self.serial in self._connected_serials:
            print(f"Warning: RealSense {self.serial} is already connected")
            return False
            
        try:
            # Get shared context
            ctx = self.get_shared_context()
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable device with specific serial
            self.config.enable_device(self.serial)
            
            # Different configuration for D405 vs other cameras
            if self.serial == "218622278343":  # D405 camera
                print(f"Using D405-specific configuration for camera {self.serial}")
                # Try RGB8 format instead of YUYV for D405
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            else:
                # Default configuration for other cameras (D435 etc.)
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get device
            self.device = profile.get_device()
            
            # Create align object
            self.align = rs.align(rs.stream.color)
            
            # Set device options for better performance
            if self.device:
                depth_sensor = self.device.first_depth_sensor()
                if depth_sensor:
                    try:
                        # First disable auto-exposure
                        depth_sensor.set_option(rs.option.enable_auto_exposure, False)
                        # Then set manual exposure
                        depth_sensor.set_option(rs.option.exposure, 8500)
                    except Exception as e:
                        print(f"Warning: Could not set exposure for RealSense {self.serial}: {e}")
                        # If manual setting fails, try enabling auto-exposure
                        try:
                            depth_sensor.set_option(rs.option.enable_auto_exposure, True)
                        except Exception as e:
                            print(f"Warning: Could not enable auto-exposure for RealSense {self.serial}: {e}")
                    
                    try:
                        # Set laser power
                        # D405 doesn't support laser power option
                        if '218622278343' not in self.serial:  # Skip for D405 camera
                            depth_sensor.set_option(rs.option.laser_power, 100)
                    except Exception as e:
                        print(f"Warning: Could not set laser power for RealSense {self.serial}: {e}")
                    
                    try:
                        # Set depth units to millimeters
                        depth_sensor.set_option(rs.option.depth_units, 0.001)
                    except Exception as e:
                        print(f"Warning: Could not set depth units for RealSense {self.serial}: {e}")
            
            self.connected = True
            self._connected_serials.add(self.serial)
            print(f"Successfully started RealSense {self.serial}")
            return True
            
        except Exception as e:
            print(f"Failed to setup RealSense {self.serial}: {e}")
            self.cleanup()
            return False

    def _stream_loop(self):
        """Stream loop for RealSense camera"""
        if not self.connected or not self.streaming:
            print(f"RealSense {self.serial} not ready for streaming")
            return
            
        print(f"Starting stream loop for RealSense {self.serial}")
        try:
            # Create filters with reduced settings for better performance
            depth_filter = rs.spatial_filter()
            depth_filter.set_option(rs.option.filter_magnitude, 1)  # Reduced from 2
            depth_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # Reduced from 0.5
            depth_filter.set_option(rs.option.filter_smooth_delta, 15)  # Reduced from 20
            
            # Create depth to disparity filter
            depth_to_disparity = rs.disparity_transform(True)
            
            # Create disparity to depth filter
            disparity_to_depth = rs.disparity_transform(False)
            
            # Define depth range in millimeters - adjust based on camera model
              # 10cm - safe minimum for all RealSense models
            # The D405 has a shorter range than D435
            if '218622278343' in self.serial:  # D405 camera
                MIN_DEPTH = 50  # D405 works at very close range
                MAX_DEPTH = 600  # Match the MAX_DEPTH of other cameras for consistent visualization
                
                # Special filter settings for D405
                depth_filter.set_option(rs.option.filter_magnitude, 2)
                depth_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
                depth_filter.set_option(rs.option.filter_smooth_delta, 20)
                
                print(f"D405 camera {self.serial} depth range set to {MIN_DEPTH}-{MAX_DEPTH}mm")
            else:
                MIN_DEPTH = 100
                MAX_DEPTH = 600  # 60cm for D435
            
            frame_count = 0
            last_log_time = time.time()
            
            while not self.stop_event.is_set() and self.streaming:
                try:
                    # Calculate time since last frame
                    current_time = time.time()
                    time_since_last = current_time - self._last_frame_time
                    
                    # Skip if we're trying to get frames too quickly
                    if time_since_last < self._frame_interval:
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
                        continue
                    
                    # Log streaming status every 5 seconds
                    if current_time - last_log_time > 5:
                        print(f"RealSense {self.serial} still streaming... (frames: {frame_count})")
                        last_log_time = current_time
                    
                    # Wait for frames with increased timeout
                    frames = self.pipeline.wait_for_frames(timeout_ms=self._frame_timeout)
                    
                    # Align depth frame to color frame
                    aligned_frames = self.align.process(frames)
                    
                    # Get color and depth frames
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        print(f"Warning: Missing frames for RealSense {self.serial}")
                        continue
                    
                    # Apply depth filters
                    filtered_depth = depth_frame
                    filtered_depth = depth_to_disparity.process(filtered_depth)
                    filtered_depth = depth_filter.process(filtered_depth)
                    filtered_depth = disparity_to_depth.process(filtered_depth)
                    
                    # Convert to numpy arrays
                    if self.serial == "218622278343":  # D405 camera
                        try:
                            # Convert RGB8 to BGR format for consistency with other cameras
                            color_img_rgb = np.asanyarray(color_frame.get_data())
                            
                            # Check that we have a proper RGB array
                            if len(color_img_rgb.shape) == 3 and color_img_rgb.shape[2] == 3:
                                # Convert from RGB to BGR (OpenCV's preferred format)
                                color_img = cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR)
                            else:
                                # Fallback if we don't get a proper RGB array
                                print(f"D405 camera: Unexpected RGB format: {color_img_rgb.shape}")
                                color_img = np.zeros((480, 640, 3), dtype=np.uint8)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(color_img, f"D405 Camera {self.serial}", (50, 240), font, 1, (255, 255, 255), 2)
                        except Exception as e:
                            print(f"Error processing RGB frame from D405 {self.serial}: {e}")
                            # Create a blank color image as a fallback
                            color_img = np.zeros((480, 640, 3), dtype=np.uint8)
                            # Add text to the image
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(color_img, f"D405 Camera {self.serial}", (50, 240), font, 1, (255, 255, 255), 2)
                    else:
                        # For other cameras, the format is already BGR8
                        color_img = np.asanyarray(color_frame.get_data())
                        
                    depth_img = np.asanyarray(filtered_depth.get_data())
                    
                    # Print depth stats every 30 frames
                    if frame_count % 30 == 0 and False:  # Disabled depth stats printing
                        # Only consider non-zero depth values
                        non_zero_depths = depth_img[depth_img > 0]
                        if len(non_zero_depths) > 0:
                            min_actual = np.min(non_zero_depths)
                            max_actual = np.max(non_zero_depths)
                            mean_depth = np.mean(non_zero_depths)
                            print(f"Camera {self.serial} - Depth stats (mm): min={min_actual}, max={max_actual}, mean={mean_depth}, range={MIN_DEPTH}-{MAX_DEPTH}")
                    
                    # Normalize and colorize depth image
                    if self.serial == "218622278343":  # D405 camera
                        # Use the simple, standard method that works well in the RealSense examples
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_img, alpha=0.03), 
                            cv2.COLORMAP_JET
                        )
                    else:
                        # Standard approach for D435 cameras
                        # Clamp depth values between MIN_DEPTH and MAX_DEPTH
                        depth_img = np.clip(depth_img, MIN_DEPTH, MAX_DEPTH)
                        
                        # Normalize depth values to 0-255 range
                        depth_normalized = ((depth_img - MIN_DEPTH) * (255.0 / (MAX_DEPTH - MIN_DEPTH))).astype(np.uint8)
                        
                        # Create depth colormap
                        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    
                    # Create frame dictionary
                    frame_dict = {
                        'color': color_img,
                        'depth': depth_colormap
                    }
                    
                    # Update last frame time
                    self._last_frame_time = current_time
                    frame_count += 1
                    
                    # Put frames in queue for main thread to display
                    self.frame_queue.put(("RealSense", self.serial, frame_dict))
                        
                except Exception as e:
                    if self.stop_event.is_set():
                        break
                    print(f"Error processing frames for RealSense {self.serial}: {e}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"Error in RealSense {self.serial} stream: {e}")
        finally:
            print(f"Stream loop ended for RealSense {self.serial}")
            self.streaming = False

    def cleanup(self):
        """Cleanup RealSense connection"""
        if not self.connected:
            return
            
        print(f"Stopping RealSense {self.serial}")
        try:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    print(f"Error stopping pipeline for RealSense {self.serial}: {e}")
                finally:
                    self.pipeline = None
                    
            # Force device reset
            try:
                if self.device:
                    self.device.hardware_reset()
            except Exception as e:
                print(f"Error during hardware reset for RealSense {self.serial}: {e}")
                
        except Exception as e:
            print(f"Error during cleanup for RealSense {self.serial}: {e}")
        finally:
            self.connected = False
            self.config = None
            self.device = None
            self.streaming = False
            self._connected_serials.discard(self.serial)
            # Add a small delay to ensure device is fully released
            time.sleep(1)

class GridLayout:
    """Utility class for creating grid layouts of camera frames"""
    
    @staticmethod
    def create_realsense_layout(frames, num_cameras):
        """Create a grid layout of RealSense frames in left, center, right order"""
        if not frames:
            print("No frames available for grid creation")
            return None
            
        # For 3 cameras, we want two rows (color and depth)
        rows, cols = 2, 3
        
        # Print info about available frames
        print(f"Creating grid layout with {len(frames)} cameras:")
        for serial, frame_dict in frames:
            print(f"  - Camera {serial}: color shape {frame_dict['color'].shape}, depth shape {frame_dict['depth'].shape}")
        
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
                print(f"Placing camera {serial} at position {j}")
                # Place color frame in top row
                grid[0:h, j*w:(j+1)*w] = frame_dict['color']
                # Place depth frame in bottom row
                grid[h:h*2, j*w:(j+1)*w] = frame_dict['depth']
        
        return grid

    @staticmethod
    def create_mixed_layout(digit_frames, realsense_frames):
        """Create a grid layout with DIGIT and RealSense frames in specified arrangement"""
        if not digit_frames and not realsense_frames:
            print("No frames available for grid creation")  # Debug print
            return None, (0, 0)
            
        # Get frame dimensions
        digit_frame = next(iter(digit_frames.values())) if digit_frames else None
        rs_frame = next(iter(realsense_frames.values())) if realsense_frames else None
        
        # If we have no frames, return None
        if digit_frame is None and rs_frame is None:
            print("No valid frames found")  # Debug print
            return None, (0, 0)
            
        # Get dimensions
        digit_h, digit_w = digit_frame.shape[:2] if digit_frame is not None else (0, 0)
        rs_h, rs_w = rs_frame['color'].shape[:2] if rs_frame is not None else (0, 0)
        
        print(f"DIGIT dimensions: {digit_h}x{digit_w}")  # Debug print
        print(f"RealSense dimensions: {rs_h}x{rs_w}")  # Debug print
        
        # If we only have DIGIT frames
        if rs_frame is None:
            total_width = digit_w * 2
            total_height = digit_h
            grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
            
            # Place DIGIT frames
            digit_frames_list = sorted(digit_frames.items())
            if len(digit_frames_list) >= 1:
                grid[0:digit_h, 0:digit_w] = digit_frames_list[0][1]
            if len(digit_frames_list) >= 2:
                grid[0:digit_h, -digit_w:] = digit_frames_list[1][1]
                
            return grid, (total_width, total_height)
            
        # If we only have RealSense frames
        if digit_frame is None:
            total_width = rs_w * 3
            total_height = rs_h * 2
            grid = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            
            # Define the order of cameras (left, center, right)
            camera_order = {
                "137322077775": 0,  # Left
                "218622278343": 0,  # Left (D405)
                "250122077836": 1,  # Center
                "332322072522": 2   # Right
            }
            
            # Place RealSense frames according to predefined order
            for serial, frame_dict in realsense_frames.items():
                if serial in camera_order:
                    j = camera_order[serial]
                    # Place color frame in top row
                    grid[0:rs_h, j*rs_w:(j+1)*rs_w] = frame_dict['color']
                    # Place depth frame in bottom row
                    grid[rs_h:rs_h*2, j*rs_w:(j+1)*rs_w] = frame_dict['depth']
                
            return grid, (total_width, total_height)
        
        # For combined layout:
        # Calculate total dimensions
        # Width: Two DIGIT frames side by side
        total_width = max(digit_w * 2, rs_w * 3)
        # Height: RealSense frames (2 rows) plus DIGIT frames
        total_height = digit_h + rs_h * 2
        
        print(f"Creating combined grid with dimensions: {total_width}x{total_height}")  # Debug print
        
        # Create empty grid with white background
        grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place DIGIT frames at the top
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
        start_y = digit_h  # Start after DIGIT frames
        
        # Define the order of cameras (left, center, right)
        camera_order = {
            "137322077775": 0,  # Left
            "218622278343": 0,  # Left (D405)
            "250122077836": 1,  # Center
            "332322072522": 2   # Right
        }
        
        # Place RealSense frames according to predefined order
        for serial, frame_dict in realsense_frames.items():
            if serial in camera_order:
                j = camera_order[serial]
                # Place color frame in top row
                grid[start_y:start_y+rs_h, j*rs_w:(j+1)*rs_w] = frame_dict['color']
                # Place depth frame in bottom row
                grid[start_y+rs_h:start_y+rs_h*2, j*rs_w:(j+1)*rs_w] = frame_dict['depth']
        
        return grid, (total_width, total_height)

def setup_digits():
    """Setup and connect to all DIGIT devices"""
    digits = DigitHandler.list_digits()
    print("\nDetected DIGIT devices:", digits)
    
    # Keep track of unique serial numbers
    connected_serials = set()
    connected_digits = []
    
    # Sort devices by serial number to ensure consistent ordering
    sorted_digits = sorted(digits, key=lambda x: x['serial'])
    
    for digit_info in sorted_digits:
        serial = digit_info['serial']
        
        # Skip if we've already connected to this serial number
        if serial in connected_serials:
            continue
            
        camera = DigitCamera(serial)
        if camera.connect():
            connected_digits.append(camera)
            connected_serials.add(serial)
            print(f"Successfully connected to DIGIT {serial}")
    
    return connected_digits

def setup_realsense():
    """Setup and connect to all RealSense devices"""
    ctx = rs.context()
    
    # Get all connected devices
    devices = []
    seen_serials = set()
    
    print("\nScanning for RealSense devices...")
    
    # First pass: collect all unique devices
    for d in ctx.devices:
        try:
            serial = d.get_info(rs.camera_info.serial_number)
            name = d.get_info(rs.camera_info.name)
            usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
            print(f"Found device: {name} (Serial: {serial}, USB: {usb_type})")
            
            if serial not in seen_serials:
                devices.append((serial, d))
                seen_serials.add(serial)
        except Exception as e:
            print(f"Error getting device info: {e}")
            continue
    
    # Sort devices by serial number for consistent ordering
    devices.sort(key=lambda x: x[0])
    
    print(f"\nTotal RealSense devices found: {len(devices)}")
    for serial, dev in devices:
        print(f"  - {serial}")
    
    connected_cameras = []
    
    # Second pass: connect to each unique device
    for serial, dev in devices:
        try:
            print(f"\nAttempting to connect to RealSense {serial}...")
            camera = RealSenseCamera(serial)
            if camera.connect():
                connected_cameras.append(camera)
                print(f"Successfully connected to RealSense {serial}")
            else:
                print(f"Failed to connect to RealSense {serial}")
        except Exception as e:
            print(f"Error during connection to RealSense {serial}: {e}")
            continue
    
    print(f"\nSuccessfully connected to {len(connected_cameras)} out of {len(devices)} RealSense cameras")
    return connected_cameras 