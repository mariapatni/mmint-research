## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Print device information
print("Device Name:", device.get_info(rs.camera_info.name))
print("Device Product Line:", device_product_line)
print("Device Serial Number:", device.get_info(rs.camera_info.serial_number))
print("Device Firmware Version:", device.get_info(rs.camera_info.firmware_version))
print("Device USB Type:", device.get_info(rs.camera_info.usb_type_descriptor))

# For D405, RGB is part of the Stereo Module, so we don't need to check for separate RGB sensor
print("D405 detected - RGB capabilities are part of the Stereo Module sensor")

# Check what sensors are available
for s in device.sensors:
    print("Sensor Name:", s.get_info(rs.camera_info.name))
    print("Sensor Serial Number:", s.get_info(rs.camera_info.serial_number))
    print("Sensor Firmware Version:", s.get_info(rs.camera_info.firmware_version))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Give the camera a moment to initialize
print("Camera initialized, waiting for first frames...")
time.sleep(2)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_depth = os.path.join(script_dir, "out/depth")
subfolder_rgb = os.path.join(script_dir, "out/rgb")

# Check if the subfolder exists, and create it if it does not
if not os.path.exists(subfolder_depth):
    os.makedirs(subfolder_depth)
if not os.path.exists(subfolder_rgb):
    os.makedirs(subfolder_rgb)


RecordStream = False
frame_count = 0
max_frames = 200  # Limit to prevent memory issues

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000)  # Increased timeout to 10 seconds
        except Exception as e:
            print(f"Error waiting for frames: {e}")
            print("Retrying...")
            continue
            
        # frames.get_depth_frame() is a 640x360 depth image

      


        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()




        # Get instrinsics from aligned_depth_frame
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Debug: Print raw depth info (only when recording)
        if RecordStream:
            print(f"Raw depth - dtype: {depth_image.dtype}, shape: {depth_image.shape}")
            print(f"Raw depth - min: {depth_image.min()}, max: {depth_image.max()}")
            print(f"Raw depth - sample values: {depth_image[259, 244:249]}")
            print(f"Depth scale: {depth_scale}")
            print(f"Depth in meters - sample: {depth_image[259, 244] * depth_scale:.3f} m")
            print(f"Valid depth pixels: {np.sum((depth_image > 0) & (depth_image < 65535))}")

        # Display the aligned color image
        cv2.namedWindow("RealSense Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("RealSense Camera", color_image)

        key = cv2.waitKey(1)

        # Debug: Print key press info
        if key != -1:
            print(f"Key pressed: {key} (ASCII: {chr(key) if key < 128 else 'non-printable'})")

        # Start saving the frames if space is pressed once until it is pressed again
        if key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True
                frame_count = 0  # Reset frame counter

                with open(os.path.join(script_dir, "out/cam_K.txt"), "w") as f:
                    f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                    f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")

                print("Recording started")
            else:
                RecordStream = False
                print("Recording stopped")

        if RecordStream:
            # Check frame limit
            if frame_count >= max_frames:
                print(f"Reached maximum frame limit ({max_frames}). Stopping recording.")
                RecordStream = False
                continue
                
            # Use sequential frame numbering (0001, 0002, etc.)
            framename = f"{frame_count + 1:04d}"

            # Define the path to the image file within the subfolder
            image_path_depth = os.path.join(subfolder_depth, f"{framename}.png")
            image_path_rgb = os.path.join(subfolder_rgb, f"{framename}.png")

            try:
                # Save aligned depth and RGB frames
                # Convert depth from mm to meters by dividing by 1000

                print("raw depth = ", depth_image[259, 244])
                print("depth scale = ", depth_scale)
                

                depth_m = (depth_image * depth_scale).astype(np.float32)
                depth_mm = (depth_m * 1000.0).astype(np.uint16)

                print("actual depth_m =", depth_image[259, 244] * depth_scale)

                print("stored depth_m = ", depth_m[259, 244])
                print("depth_mm = ", depth_mm[259, 244])


                success_depth = cv2.imwrite(image_path_depth, depth_mm)
                success_rgb = cv2.imwrite(image_path_rgb, color_image)
                
                
                if success_depth and success_rgb:
                    frame_count += 1
                    print(f"Saved frame: {framename}.png ({frame_count}/{max_frames})")
                else:
                    print(f"Failed to save frame: {framename}.png")
                    
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS max
                
            except Exception as e:
                print(f"Error saving frame {framename}.png: {e}")
                continue

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break
finally:
    pipeline.stop()