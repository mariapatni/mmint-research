import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import glob
import time
import traceback
import gc
import sys

class PointCloudVisualizer:
    def __init__(self, data_dir, debug=False):
        """
        Initialize the visualizer with the data directory
        
        Args:
            data_dir (str): Path to the data directory (can be processed_data root or specific trial)
            debug (bool): Whether to enable debug output
        """
        self.debug = debug
        try:
            # Remove first base folder and append processed_data
            path_parts = Path(data_dir).parts
            if len(path_parts) > 1:
                self.data_dir = Path("processed_data") / Path(*path_parts[1:])
            else:
                self.data_dir = Path("processed_data")
                
            print(f"Looking for point clouds in: {self.data_dir}")
            
            self.realsense_mapping = {
                "side_camera_1": "Left Camera",
                "gripper_camera": "Center Camera",
                "side_camera_2": "Right Camera"
            }
            
            # Initialize variables
            self.current_frames = {}  # Dictionary to store current frame for each camera
            self.max_frames = 0
            self.current_object = None
            self.current_trial = None
            self.pcd_files = {}  # Dictionary to store point cloud files for each camera
            self.current_pcds = {}  # Dictionary to store current point clouds for each camera
            self.running = True
            self.visualizers = {}  # Dictionary to store visualizers for each camera
            self.key_states = {}  # Dictionary to store key states for each camera
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            print(traceback.format_exc())
            raise
            
    def debug_print(self, message):
        """Print debug message if debug is enabled"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def create_visualizer(self, camera_name):
        """Create a new visualizer window for a specific camera"""
        try:
            if camera_name in self.visualizers and self.visualizers[camera_name] is not None:
                self.cleanup_visualizer(camera_name)
                
            # Create visualization window
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name=f"Point Cloud - {self.realsense_mapping[camera_name]}", width=640, height=480)
            
            # Configure visualizer
            opt = vis.get_render_option()
            opt.point_size = 5.0  # Much larger points
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            
            # Set up the view
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            
            # Initialize key states for this camera
            self.key_states[camera_name] = {
                'n_pressed': False,
                'p_pressed': False,
                'last_key_time': 0,
                'is_updating': False
            }
            
            # Create callback functions that don't capture the visualizer
            def make_key_callback(key, is_press):
                def callback(vis):
                    current_time = time.time()
                    # Only process key press if enough time has passed since last key press and not currently updating
                    if is_press and current_time - self.key_states[camera_name]['last_key_time'] > 0.1 and not self.key_states[camera_name]['is_updating']:
                        if key == ord('N'):
                            self.key_states[camera_name]['is_updating'] = True
                            self.next_frame(vis, camera_name)
                            # Force immediate update
                            vis.poll_events()
                            vis.update_renderer()
                            self.key_states[camera_name]['last_key_time'] = current_time
                            self.key_states[camera_name]['is_updating'] = False
                        elif key == ord('P'):
                            self.key_states[camera_name]['is_updating'] = True
                            self.prev_frame(vis, camera_name)
                            # Force immediate update
                            vis.poll_events()
                            vis.update_renderer()
                            self.key_states[camera_name]['last_key_time'] = current_time
                            self.key_states[camera_name]['is_updating'] = False
                        elif key == ord('R'):
                            self.reset_view(vis, camera_name)
                        elif key == ord('Q'):
                            self.quit(vis, camera_name)
                return callback
            
            # Register the callbacks
            vis.register_key_callback(ord('N'), make_key_callback(ord('N'), True))
            vis.register_key_callback(ord('P'), make_key_callback(ord('P'), True))
            vis.register_key_callback(ord('R'), make_key_callback(ord('R'), True))
            vis.register_key_callback(ord('Q'), make_key_callback(ord('Q'), True))
            
            # Register release callbacks
            vis.register_key_callback(ord('N') + 256, make_key_callback(ord('N'), False))
            vis.register_key_callback(ord('P') + 256, make_key_callback(ord('P'), False))
            
            self.debug_print(f"[DEBUG] registered keys for {camera_name}")
            
            self.visualizers[camera_name] = vis
            
            self.debug_print(f"[DEBUG] exited out of create_visualizer for {camera_name}")
            
        except Exception as e:
            print(f"Error creating visualizer for {camera_name}: {str(e)}")
            print(traceback.format_exc())
            raise
        
    def cleanup_visualizer(self, camera_name):
        """Clean up resources for a specific visualizer"""
        try:
            self.debug_print(f"[DEBUG] Starting cleanup_visualizer for {camera_name}")
            if camera_name in self.visualizers and self.visualizers[camera_name] is not None:
                self.debug_print(f"[DEBUG] Found visualizer for {camera_name}")
                vis = self.visualizers[camera_name]
                try:
                    # First unregister all callbacks
                    self.debug_print(f"[DEBUG] About to unregister callbacks for {camera_name}")
                    vis.register_key_callback(ord('N'), None)
                    vis.register_key_callback(ord('P'), None)
                    vis.register_key_callback(ord('R'), None)
                    vis.register_key_callback(ord('Q'), None)
                    vis.register_key_callback(ord('N') + 256, None)
                    vis.register_key_callback(ord('P') + 256, None)
                    self.debug_print(f"[DEBUG] Unregistered callbacks for {camera_name}")
                    
                    # Clear geometries
                    self.debug_print(f"[DEBUG] About to clear geometries for {camera_name}")
                    vis.clear_geometries()
                    self.debug_print(f"[DEBUG] Cleared geometries for {camera_name}")
                    
                    # Update one last time
                    self.debug_print(f"[DEBUG] About to poll events for {camera_name}")
                    vis.poll_events()
                    self.debug_print(f"[DEBUG] Polled events for {camera_name}")
                    self.debug_print(f"[DEBUG] About to update renderer for {camera_name}")
                    vis.update_renderer()
                    self.debug_print(f"[DEBUG] Updated renderer for {camera_name}")
                    
                    # Destroy window
                    self.debug_print(f"[DEBUG] About to destroy window for {camera_name}")
                    vis.destroy_window()
                    self.debug_print(f"[DEBUG] Destroyed window for {camera_name}")
                    
                except Exception as e:
                    self.debug_print(f"[DEBUG] Error in visualizer cleanup for {camera_name}: {str(e)}")
                
                # Clear references
                self.debug_print(f"[DEBUG] About to clear visualizer reference for {camera_name}")
                self.visualizers[camera_name] = None
                self.debug_print(f"[DEBUG] Cleared visualizer reference for {camera_name}")
                
                if camera_name in self.current_pcds:
                    self.debug_print(f"[DEBUG] About to clear point cloud for {camera_name}")
                    del self.current_pcds[camera_name]
                    self.debug_print(f"[DEBUG] Cleared point cloud for {camera_name}")
                    
            self.debug_print(f"[DEBUG] Completed cleanup_visualizer for {camera_name}")
                    
        except Exception as e:
            self.debug_print(f"[DEBUG] Error in cleanup_visualizer for {camera_name}: {str(e)}")
            print(traceback.format_exc())
        
    def cleanup(self):
        """Clean up all resources"""
        try:
            self.debug_print("Starting cleanup")
            self.running = False
            
            # Just destroy windows and clear references
            for camera_name in list(self.visualizers.keys()):
                if self.visualizers[camera_name] is not None:
                    try:
                        self.visualizers[camera_name].destroy_window()
                    except:
                        pass
                    self.visualizers[camera_name] = None
            
            # Clear all data structures
            self.visualizers.clear()
            self.key_states.clear()
            self.current_frames.clear()
            self.pcd_files.clear()
            self.current_pcds.clear()
            
            self.debug_print("Cleanup complete")
            
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")

    def load_point_cloud_files(self, trial_dir):
        """Load list of point cloud files for all cameras"""
        try:
            # Get all camera directories
            cameras = [d for d in trial_dir.iterdir() 
                      if d.is_dir() and d.name in self.realsense_mapping]
            
            if not cameras:
                print(f"No camera directories found in {trial_dir}")
                return False
                
            # Load point cloud files for each camera
            for camera_dir in cameras:
                pcd_dir = camera_dir / "rs_point_cloud"
                print(f"Looking for point clouds in: {pcd_dir}")
                self.pcd_files[camera_dir.name] = sorted(glob.glob(str(pcd_dir / "*.ply")))
                self.current_frames[camera_dir.name] = 0
                
                if not self.pcd_files[camera_dir.name]:
                    print(f"No point clouds found in {pcd_dir}")
                    return False
            
            # Set max frames based on the camera with the fewest frames
            self.max_frames = min(len(files) for files in self.pcd_files.values())
            
            print(f"Found {self.max_frames} frames for each camera")
            return True
            
        except Exception as e:
            print(f"Error loading point cloud files: {str(e)}")
            print(traceback.format_exc())
            return False
        
    def load_current_point_cloud(self, camera_name):
        """Load the current point cloud for a specific camera"""
        try:
            current_frame = self.current_frames[camera_name]
            if 0 <= current_frame < self.max_frames:
                print(f"Loading frame {current_frame + 1}/{self.max_frames} for {camera_name}")
                
                # Load point cloud
                pcd = o3d.io.read_point_cloud(self.pcd_files[camera_name][current_frame])
                if pcd is None or len(pcd.points) == 0:
                    print(f"Warning: Empty point cloud for {camera_name}")
                    return False
                    
                # Print point cloud info for debugging
                print(f"Point cloud for {camera_name}:")
                print(f"  Number of points: {len(pcd.points)}")
                bbox = pcd.get_axis_aligned_bounding_box()
                print(f"  Bounding box: {bbox}")
                
                # Get center and dimensions
                center = bbox.get_center()
                extent = bbox.get_extent()
                
                # Scale the point cloud to a reasonable size
                scale = 0.5 / max(extent)  # Scale to fit in a 0.5 unit box
                pcd.scale(scale, center=center)
                
                # Ensure point cloud has colors
                if not pcd.has_colors():
                    pcd.paint_uniform_color([1, 0, 0])  # Red color
                    
                # Transform point cloud based on camera position
                if camera_name == "side_camera_1":
                    # Left camera view
                    transform = np.array([
                        [1, 0, 0, -0.5],  # Move left
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                elif camera_name == "side_camera_2":
                    # Right camera view
                    transform = np.array([
                        [1, 0, 0, 0.5],  # Move right
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                else:
                    # Center camera view
                    transform = np.eye(4)
                    
                pcd.transform(transform)
                self.current_pcds[camera_name] = pcd
                
                # Print post-processing info
                print(f"  Points after processing: {len(pcd.points)}")
                print(f"  New bounding box: {pcd.get_axis_aligned_bounding_box()}")
                
                return True
            return False
            
        except Exception as e:
            print(f"Error loading current point cloud for {camera_name}: {str(e)}")
            print(traceback.format_exc())
            return False
        
    def update_visualization(self, camera_name):
        """Update the visualization for a specific camera"""
        try:
            if camera_name not in self.visualizers or self.visualizers[camera_name] is None:
                self.create_visualizer(camera_name)
                
            vis = self.visualizers[camera_name]
            
            # Store the current view parameters before updating
            view_control = vis.get_view_control()
            current_camera_params = view_control.convert_to_pinhole_camera_parameters()
            
            # Clear geometries and update
            vis.clear_geometries()
            
            if self.load_current_point_cloud(camera_name):
                # Add geometry
                vis.add_geometry(self.current_pcds[camera_name])
                
                # Print current frame info
                print(f"Frame {self.current_frames[camera_name] + 1}/{self.max_frames} for {camera_name}")
                
                # Restore the previous view parameters
                view_control.convert_from_pinhole_camera_parameters(current_camera_params)
                
                # Update view
                vis.poll_events()
                vis.update_renderer()
                
        except Exception as e:
            print(f"Error updating visualization for {camera_name}: {str(e)}")
            print(traceback.format_exc())
        
    def next_frame(self, vis, camera_name):
        """Move to next frame for a specific camera"""
        try:
            if self.current_frames[camera_name] < self.max_frames - 1:
                self.current_frames[camera_name] += 1
                self.update_visualization(camera_name)
                
        except Exception as e:
            print(f"Error in next_frame for {camera_name}: {str(e)}")
            print(traceback.format_exc())
        
    def prev_frame(self, vis, camera_name):
        """Move to previous frame for a specific camera"""
        try:
            if self.current_frames[camera_name] > 0:
                self.current_frames[camera_name] -= 1
                self.update_visualization(camera_name)
                
        except Exception as e:
            print(f"Error in prev_frame for {camera_name}: {str(e)}")
            print(traceback.format_exc())
        
    def reset_view(self, vis, camera_name):
        """Reset the view to default for a specific camera"""
        try:
            self.visualizers[camera_name].reset_view_point(True)
            self.visualizers[camera_name].poll_events()
            self.visualizers[camera_name].update_renderer()
            
        except Exception as e:
            print(f"Error in reset_view for {camera_name}: {str(e)}")
            print(traceback.format_exc())
        
    def quit(self, vis, camera_name):
        """Request quit from all visualizations"""
        self.cleanup()
        sys.exit(0)

    def run(self):
        """Run the visualization"""
        try:
            self.debug_print(f"Processing directory: {self.data_dir}")
            
            # If we're in a trial directory, process it directly
            if self.data_dir.name.startswith('trial'):
                self.current_trial = self.data_dir
                self.current_object = self.data_dir.parent.name
            else:
                # Otherwise, look for object directories
                objects = [d for d in self.data_dir.iterdir() if d.is_dir()]
                if not objects:
                    print("No object directories found")
                    return
                    
                # For now, just process the first object's first trial
                self.current_object = objects[0].name
                trials = [d for d in objects[0].iterdir() if d.is_dir() and d.name.startswith('trial')]
                if not trials:
                    print(f"No trials found in {objects[0]}")
                    return
                    
                self.current_trial = trials[0]
            
            print(f"Processing trial: {self.current_trial}")
            print("Controls for each window:")
            print("  N: Next frame")
            print("  P: Previous frame")
            print("  R: Reset view")
            print("  Q: Quit window")
            
            if self.load_point_cloud_files(self.current_trial):
                # Create visualizers for each camera
                for camera_name in self.pcd_files.keys():
                    self.create_visualizer(camera_name)
                    self.update_visualization(camera_name)
                
                # Main visualization loop
                while self.running:
                    try:
                        # Check if all windows are closed
                        if all(v is None for v in self.visualizers.values()):
                            self.running = False
                            break
                            
                        # Update all active visualizers
                        for camera_name, vis in self.visualizers.items():
                            if vis is not None:
                                vis.poll_events()
                                vis.update_renderer()
                        
                        time.sleep(0.001)
                    except Exception as e:
                        print(f"Error in main loop: {str(e)}")
                        self.running = False
                        break
                        
        except Exception as e:
            print(f"Error in run: {str(e)}")
        finally:
            if self.running:
                self.cleanup()
            sys.exit(0)

def main():
    try:
        parser = argparse.ArgumentParser(description='Visualize point cloud sequences')
        parser.add_argument('--data_dir', type=str, required=True,
                          help='Path to the data directory (can be processed_data root or specific trial)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        args = parser.parse_args()
        
        # Create visualizer
        visualizer = PointCloudVisualizer(args.data_dir, debug=args.debug)
        
        # Run visualization
        visualizer.run()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Force garbage collection
        gc.collect()

if __name__ == "__main__":
    main() 