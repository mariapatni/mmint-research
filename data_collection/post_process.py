import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
import pyrealsense2 as rs
import open3d as o3d

class RealSenseProcessor:
    """Class to handle post-processing of RealSense camera data"""
    
    def __init__(self, data_dir):
        """
        Initialize the processor with the data directory
        
        Args:
            data_dir (str): Path to the collected data directory or specific trial directory
        """
        self.data_dir = Path(data_dir)
        # Create processed_data inside data_collection_scripts
        self.processed_dir = Path(__file__).parent / "processed_data"
        self.realsense_mapping = {
            "250122077836": "side_camera_1",  # Left camera
            "137322077775": "gripper_camera",  # Center camera
            "332322072522": "side_camera_2"   # Right camera
        }
        
        # Create processed data directory if it doesn't exist
        self.processed_dir.mkdir(exist_ok=True)
        print(f"Created processed data directory at: {self.processed_dir}")
        
        # Initialize RealSense pipeline to get intrinsics
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline to get intrinsics
        self.pipeline.start(self.config)
        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Get depth intrinsics
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Create intrinsic matrix
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.depth_intrinsics.width,
            height=self.depth_intrinsics.height,
            fx=self.depth_intrinsics.fx,
            fy=self.depth_intrinsics.fy,
            cx=self.depth_intrinsics.ppx,
            cy=self.depth_intrinsics.ppy
        )
        
        # Stop pipeline after getting intrinsics
        self.pipeline.stop()
        
    def get_object_name(self, trial_dir):
        """Get the object name from the trial directory path"""
        return trial_dir.parent.name
        
    def get_trial_dirs(self):
        """Get all trial directories in the data directory"""
        # If the data directory is a trial directory, return it
        if self.data_dir.name.startswith('trial'):
            return [self.data_dir]
        # Otherwise, look for trial directories
        return [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('trial')]
    
    def get_camera_dirs(self, trial_dir):
        """Get all RealSense camera directories in a trial"""
        return [d for d in trial_dir.iterdir() 
                if d.is_dir() and d.name in self.realsense_mapping.values()]
    
    def get_frame_files(self, camera_dir):
        """Get all frame files for a camera"""
        color_dir = camera_dir / "color"
        depth_dir = camera_dir / "depth"
        
        # Get all color frames
        color_frames = sorted([f for f in color_dir.glob("*.png")])
        depth_frames = sorted([f for f in depth_dir.glob("*.png")])
        
        return list(zip(color_frames, depth_frames))
    
    def create_point_cloud(self, color_path, depth_path):
        """
        Create a point cloud from color and depth images
        
        Args:
            color_path (Path): Path to color image
            depth_path (Path): Path to depth image
            
        Returns:
            o3d.geometry.PointCloud: Point cloud object
        """
        # Read images
        color_img = cv2.imread(str(color_path))
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        
        # Convert color image to RGB
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0/self.depth_scale,
            depth_trunc=3.0,  # Maximum depth in meters
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsic
        )
        
        return pcd
    
    def setup_processed_dirs(self, trial_dir):
        """
        Create the processed directory structure for a trial
        
        Args:
            trial_dir (Path): Original trial directory
            
        Returns:
            Path: Path to processed trial directory
        """
        # Get object name
        object_name = self.get_object_name(trial_dir)
        
        # Create object directory
        object_dir = self.processed_dir / object_name
        object_dir.mkdir(exist_ok=True)
        
        # Create processed trial directory
        processed_trial_dir = object_dir / trial_dir.name
        processed_trial_dir.mkdir(exist_ok=True)
        
        # Create camera directories
        for camera_name in self.realsense_mapping.values():
            camera_dir = processed_trial_dir / camera_name
            camera_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (camera_dir / "color").mkdir(exist_ok=True)
            (camera_dir / "depth").mkdir(exist_ok=True)
            (camera_dir / "rs_point_cloud").mkdir(exist_ok=True)
        
        return processed_trial_dir
    
    def process_frame(self, color_path, depth_path, processed_trial_dir, camera_name):
        """
        Process a single frame pair (color and depth)
        
        Args:
            color_path (Path): Path to color image
            depth_path (Path): Path to depth image
            processed_trial_dir (Path): Path to processed trial directory
            camera_name (str): Name of the camera
            
        Returns:
            bool: True if processing was successful
        """
        # Get frame number
        frame_number = int(color_path.stem.split('_')[1])
        
        # Create processed paths
        processed_camera_dir = processed_trial_dir / camera_name
        color_out_path = processed_camera_dir / "color" / f"frame_{frame_number:06d}.png"
        depth_out_path = processed_camera_dir / "depth" / f"frame_{frame_number:06d}.png"
        pcd_out_path = processed_camera_dir / "rs_point_cloud" / f"frame_{frame_number:06d}.ply"
        
        try:
            # Copy original images
            shutil.copy2(color_path, color_out_path)
            shutil.copy2(depth_path, depth_out_path)
            
            # Create and save point cloud
            pcd = self.create_point_cloud(color_path, depth_path)
            o3d.io.write_point_cloud(str(pcd_out_path), pcd)
            
            return True
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return False
    
    def process_trial(self, trial_dir):
        """
        Process all frames in a trial
        
        Args:
            trial_dir (Path): Path to trial directory
        """
        print(f"\nProcessing trial: {trial_dir.name}")
        
        # Setup processed directories
        processed_trial_dir = self.setup_processed_dirs(trial_dir)
        
        # Get all camera directories
        camera_dirs = self.get_camera_dirs(trial_dir)
        
        for camera_dir in camera_dirs:
            print(f"\nProcessing camera: {camera_dir.name}")
            
            # Get all frame pairs
            frame_pairs = self.get_frame_files(camera_dir)
            
            # Process each frame pair
            for color_path, depth_path in frame_pairs:
                self.process_frame(color_path, depth_path, processed_trial_dir, camera_dir.name)
    
    def process_all_trials(self):
        """Process all trials in the data directory"""
        trial_dirs = self.get_trial_dirs()
        
        for trial_dir in trial_dirs:
            self.process_trial(trial_dir)

def main():
    parser = argparse.ArgumentParser(description='Post-process collected RealSense data')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the collected data directory or specific trial directory')
    args = parser.parse_args()
    
    # Create processor
    processor = RealSenseProcessor(args.data_dir)
    
    # Process all trials
    processor.process_all_trials()

if __name__ == "__main__":
    main() 