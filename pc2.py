# THIS IS FOR THE NERF FROM NERFSTUDIO

import open3d as o3d
import os
import numpy as np
from typing import Optional

input_dir = "./data/object_sdf_data/lego"


def load_point_cloud(input_dir: str) -> Optional[o3d.geometry.PointCloud]:
    """Load the point cloud from the input directory."""
    input_path = os.path.join(input_dir, "exports/pcd/point_cloud.ply")
    if not os.path.exists(input_path):
        print(f"Error: Point cloud file not found at {input_path}")
        return None
        
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Loaded point cloud with {len(pcd.points)} points")
    return pcd

def load_cropped_point_cloud(input_dir: str) -> Optional[o3d.geometry.PointCloud]:
    """Load existing cropped point cloud if available."""
    cropped_file_path = os.path.join(input_dir, "cropped_nerf.ply")
    
    if os.path.exists(cropped_file_path):
        print("Loading existing cropped point cloud...")
        pcd = o3d.io.read_point_cloud(cropped_file_path)
        return pcd
    return None

def interactive_crop(pcd: o3d.geometry.PointCloud, input_dir: str) -> Optional[o3d.geometry.PointCloud]:
    """Launch interactive cropping tool for manual point cloud cropping."""
    print("🔧 Instructions:")
    print(" - Use mouse to rotate, zoom, pan")
    print(" - Press 'K' to enable selection mode")
    print(" - Left click to select points of the polygon")
    print(" - Press 'C' to crop, then 'Q' to quit") 
    
    try:
        o3d.visualization.draw_geometries_with_editing(
            [pcd], 
            window_name="Select Crop Area", 
            width=1024, 
            height=768
        )
    
        # Check if cropped file was created
        cropped_file_path = os.path.join(input_dir, "cropped_nerf.ply")
        if os.path.exists(cropped_file_path):
            print("Loading cropped point cloud...")
            return o3d.io.read_point_cloud(cropped_file_path)
        else:
            print("Warning: Cropped file not found. Using original point cloud.")
            return pcd
            
    except Exception as e:
        print(f"Error during interactive cropping: {e}")
        print("Falling back to automatic bounding box crop...")
        return apply_bounding_box_crop(pcd)

def apply_bounding_box_crop(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Apply axis-aligned bounding box crop to the point cloud."""
    if pcd is None:
        print("Error: No point cloud to crop")
        return pcd
        
    bbox = pcd.get_axis_aligned_bounding_box()
    cropped_pcd = pcd.crop(bbox)
    print(f"Applied bounding box crop. Points remaining: {len(cropped_pcd.points)}")
    return cropped_pcd

def apply_dbscan_clustering(pcd: o3d.geometry.PointCloud, eps: float = 0.005, min_points: int = 10) -> Optional[o3d.geometry.PointCloud]:
    """Apply DBSCAN clustering to remove noise and keep largest cluster."""
    if pcd is None or len(pcd.points) == 0:
        print("Error: No points to cluster")
        return None
        
    print(f"Applying DBSCAN clustering with eps={eps}, min_points={min_points}")
    labels = np.array(pcd.cluster_dbscan(
        eps=eps, 
        min_points=min_points, 
        print_progress=False
    ))

    if len(labels) != len(pcd.points):
        raise ValueError("DBSCAN label count does not match point cloud size!")

    # Filter out noise (labels == -1)
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        print("All points labeled as noise by DBSCAN")
        return None
    
    # Find largest cluster
    bincounts = np.bincount(labels[valid_mask])
    print(f"Found {len(bincounts)} clusters after DBSCAN")
    
    if bincounts.size > 0:
        largest_cluster_label = np.argmax(bincounts)
        largest_indices = np.where(labels == largest_cluster_label)[0]
        print(f"Largest cluster has {len(largest_indices)} points")
        
        if largest_indices.size > 0:
            clustered_pcd = pcd.select_by_index(largest_indices)
            print(f"Selected largest cluster. Points remaining: {len(clustered_pcd.points)}")
            return clustered_pcd
        else:
            print("Warning: Largest cluster is empty")
            return None
    else:
        print("No clusters found")
        return None

def visualize_result(pcd: o3d.geometry.PointCloud) -> None:
    """Visualize the final processed point cloud."""
    if pcd is None or len(pcd.points) == 0:
        print("Error: No point cloud to visualize")
        return
        
    try:
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Continuing without visualization...")

def save_final_result(pcd: o3d.geometry.PointCloud, input_dir: str) -> bool:
    """Save the final processed point cloud."""
    if pcd is None or len(pcd.points) == 0:
        print("Error: No point cloud to save")
        return False
        
    final_output_path = os.path.join(input_dir, "final_processed_nerf.ply")
    o3d.io.write_point_cloud(final_output_path, pcd)
    print(f"Final processed point cloud saved to {final_output_path}")
    return True

def main():
    """Main function to run the point cloud processing pipeline."""
    print(f"Processing point cloud from: {input_dir}")
    
    # Step 1: Try to load existing cropped point cloud first
    pcd = load_cropped_point_cloud(input_dir)
    if pcd is None:
        # If no cropped file exists, load original and do interactive cropping
        pcd = load_point_cloud(input_dir)
        if pcd is None:
            print("Point cloud processing failed!")
            return 1
        
        # Step 2: Interactive cropping only if cropped file doesn't exist
        pcd = interactive_crop(pcd, input_dir)
        if pcd is None:
            print("Point cloud processing failed!")
            return 1
    
    # Step 3: Apply bounding box crop
    pcd = apply_bounding_box_crop(pcd)
    
    # Check if we have enough points
    if len(pcd.points) == 0:
        print("Error: No points remaining after cropping!")
        print("Point cloud processing failed!")
        return 1
    
    print(f"Points before clustering: {len(pcd.points)}")
    
    # Step 4: Apply DBSCAN clustering
    pcd = apply_dbscan_clustering(pcd)
    if pcd is None:
        print("Point cloud processing failed!")
        return 1
    
    print(f"Final point count: {len(pcd.points)}")
    
    # Step 5: Visualize result
    visualize_result(pcd)
    
    # Step 6: Save final result
    if not save_final_result(pcd, input_dir):
        print("Point cloud processing failed!")
        return 1
    
    print("Point cloud processing completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())

