#!/usr/bin/env python3
"""
Script to play pose visualization frames as a video
"""

import cv2
import os
import glob
import numpy as np
import argparse

def play_pose_visualization(frames_dir="output1/poses_vis", fps=30, loop=True, window_name="Pose Visualization"):
    """
    Play pose visualization frames as a video
    
    Args:
        frames_dir: Directory containing the visualization frames
        fps: Frames per second for playback
        loop: Whether to loop the video
        window_name: Name of the window
    """
    
    # Get all PNG files in the directory and sort them
    frame_pattern = os.path.join(frames_dir, "*.png")
    frame_files = glob.glob(frame_pattern)
    frame_files.sort()
    
    if not frame_files:
        print(f"❌ No PNG files found in {frames_dir}")
        return
    
    print(f"📁 Found {len(frame_files)} frames in {frames_dir}")
    print(f"🎬 Playing at {fps} FPS")
    print(f"🔄 Loop: {loop}")
    print("🎮 Controls:")
    print("   - Space: Pause/Resume")
    print("   - Left/Right arrows: Step frame by frame")
    print("   - ESC or Q: Quit")
    print("   - R: Reset to beginning")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    current_frame = 0
    paused = False
    frame_delay = int(1000 / fps)  # Delay in milliseconds
    
    try:
        while True:
            if not paused:
                # Load current frame
                frame_path = frame_files[current_frame]
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"❌ Failed to load frame: {frame_path}")
                    current_frame = (current_frame + 1) % len(frame_files)
                    continue
                
                # Add frame info overlay
                frame_info = f"Frame: {current_frame + 1}/{len(frame_files)} - {os.path.basename(frame_path)}"
                cv2.putText(frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add playback status
                status = "PAUSED" if paused else "PLAYING"
                cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space
                paused = not paused
                print(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):  # R
                current_frame = 0
                print("🔄 Reset to beginning")
            elif key == 83:  # Right arrow
                current_frame = (current_frame + 1) % len(frame_files)
                print(f"⏭️  Frame {current_frame + 1}/{len(frame_files)}")
            elif key == 81:  # Left arrow
                current_frame = (current_frame - 1) % len(frame_files)
                print(f"⏮️  Frame {current_frame + 1}/{len(frame_files)}")
            
            # Auto-advance if not paused
            if not paused:
                current_frame = (current_frame + 1) % len(frame_files)
                
                # If we reached the end and not looping, stop
                if current_frame == 0 and not loop:
                    print("🏁 Reached end of sequence")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️  Playback interrupted")
    
    finally:
        cv2.destroyAllWindows()
        print("✅ Playback finished")

def main():
    parser = argparse.ArgumentParser(description="Play pose visualization frames as a video")
    parser.add_argument("--frames_dir", default="output/poses_vis", help="Directory containing visualization frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for playback")
    parser.add_argument("--no_loop", action="store_true", help="Don't loop the video")
    parser.add_argument("--window_name", default="Pose Visualization", help="Name of the window")
    
    args = parser.parse_args()
    
    # Check if frames directory exists
    if not os.path.exists(args.frames_dir):
        print(f"❌ Frames directory not found: {args.frames_dir}")
        print("Please run the pose tracking first to generate visualization frames")
        return
    
    play_pose_visualization(
        frames_dir=args.frames_dir,
        fps=args.fps,
        loop=not args.no_loop,
        window_name=args.window_name
    )

if __name__ == "__main__":
    main() 