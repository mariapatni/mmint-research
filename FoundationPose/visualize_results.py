import yaml
import cv2
import numpy as np
import os
from datareader import LinemodReader

def load_results(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def draw_pose(img, K, pose, scale=0.1):
    # Define 3D points for coordinate frame
    points = np.float32([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]]).reshape(-1,3)
    
    # Project points to 2D
    points_2d, _ = cv2.projectPoints(points, pose[:3,:3], pose[:3,3], K, None)
    points_2d = points_2d.astype(int)
    
    # Draw coordinate frame
    origin = tuple(points_2d[0].ravel())
    img = cv2.line(img, origin, tuple(points_2d[1].ravel()), (0,0,255), 3)  # X axis in red
    img = cv2.line(img, origin, tuple(points_2d[2].ravel()), (0,255,0), 3)  # Y axis in green
    img = cv2.line(img, origin, tuple(points_2d[3].ravel()), (255,0,0), 3)  # Z axis in blue
    
    return img

def visualize_results(yaml_path, linemod_dir):
    # Load results
    results = load_results(yaml_path)
    
    # For each video sequence
    for video_id in results:
        print(f"\nProcessing video {video_id}")
        
        # Initialize reader for current video sequence
        video_path = f'{linemod_dir}/lm_test_all/test/{video_id:06d}'
        reader = LinemodReader(video_path, split=None)
        
        # For each frame
        for frame_id in results[video_id]:
            # Load RGB image
            frame_idx = int(frame_id)
            rgb = reader.get_color(frame_idx)
            
            # For each object in the frame
            for obj_id in results[video_id][frame_id]:
                # Only process if object ID matches video ID
                if str(obj_id) == str(video_id):
                    pose = np.array(results[video_id][frame_id][obj_id])
                    
                    # Draw pose on image
                    rgb = draw_pose(rgb, reader.K, pose)
            
            # Display image
            cv2.imshow('Pose Estimation Results', rgb)
            key = cv2.waitKey(0)  # Wait for key press
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return

if __name__ == '__main__':
    yaml_path = 'debug/linemod_res.yml'
    linemod_dir = 'lm'
    visualize_results(yaml_path, linemod_dir) 