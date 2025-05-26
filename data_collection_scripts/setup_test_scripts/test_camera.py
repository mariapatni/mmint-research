import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    print("Starting camera...")
    pipeline.start(config)

    # Create align object
    align = rs.align(rs.stream.color)

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap to depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:  # Calculate FPS every 30 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()

            # Display FPS
            cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display images
            cv2.imshow('Color', color_image)
            cv2.imshow('Depth', depth_colormap)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 