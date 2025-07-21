# THIS IS FOR THE MERGING PCD FROM LIDAR .PLYS

import open3d as o3d
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.spatial import KDTree

input_dir = "./data/object_sdf_data/lego/"
output_file = "final_lidar_pcd.ply"

def create_or_load_bbox(input_dir):
    """
    Create bounding box from manual crop and save the cropped point cloud.
    Returns the bounding box object.
    """

    if os.path.exists(input_dir + "cropped_lidar.ply"):
        print("Loading existing cropped point cloud...")
        cropped_pcd = o3d.io.read_point_cloud(input_dir + "cropped_lidar.ply")
        bbox = cropped_pcd.get_axis_aligned_bounding_box()
        print(f"Bounding box: {bbox}")
        return bbox
    
    
    print("Creating new bounding box from manual crop...")
    pcd = o3d.io.read_point_cloud(input_dir + "lidar/0000000.ply")
    
    print("Crop the point cloud manually:")
    print("- Press 'K' to enable selection")
    print("- Left click to select polygon points") 
    print("- Press 'C' to crop")
    print("- Press 'Q' to quit")
    
    o3d.visualization.draw_geometries_with_editing([pcd])

    print("Cropped point cloud:")
    print(input_dir + "cropped_lidar.ply")


    cropped_pcd = o3d.io.read_point_cloud(input_dir + "cropped_lidar.ply")
    o3d.visualization.draw_geometries([cropped_pcd])

    bbox = cropped_pcd.get_axis_aligned_bounding_box()
    print(f"Bounding box: {bbox}")
    return bbox
        
def filter_dense_regions(pcd, radius=0.01, min_neighbors=30):
    """
    Retains only points that have >= min_neighbors within radius.
    """
    points = np.asarray(pcd.points)
    tree = KDTree(points)

    # Count neighbors for each point
    densities = np.array([len(tree.query_ball_point(p, radius)) for p in points])

    # Keep only dense points
    keep_indices = np.where(densities >= min_neighbors)[0]
    print(f"Keeping {len(keep_indices)} of {len(points)} points (dense regions).")

    dense_pcd = pcd.select_by_index(keep_indices)
    return dense_pcd

def process_single_point_cloud(pcd, ply_path):
    """
    Process a single point cloud through the full pipeline.
    """
    print(f"Processing {ply_path} with {len(pcd.points)} points")
    

    pcd = apply_dbscan_clustering(pcd, ply_path)

    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    print(f"  After voxel down sampling: {len(pcd.points)} points")
    
    # Skip outlier removal to avoid memory issues
    print(f"  Skipping outlier removal to avoid memory issues")
    
    return pcd

def apply_dbscan_clustering(pcd, ply_path, eps=0.008, min_points=8):
    """
    Apply DBSCAN clustering to a point cloud and keep the largest cluster.
    
    Args:
        pcd: Input point cloud
        ply_path: Path to the point cloud file (for logging)
        eps: DBSCAN epsilon parameter (default: 0.008)
        min_points: DBSCAN minimum points parameter (default: 8)
    
    Returns:
        Point cloud containing only the largest cluster
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if len(labels) != len(pcd.points):
        raise ValueError("DBSCAN label count does not match point cloud size!")

    valid_mask = labels >= 0
    if np.any(valid_mask):
        bincounts = np.bincount(labels[valid_mask])
        print(f"PC {ply_path} has {len(bincounts)} bincounts after DBSCAN")
        if bincounts.size > 0:
            largest_cluster_label = np.argmax(bincounts)
            largest_indices = np.where(labels == largest_cluster_label)[0]
            print(f"PC {ply_path} has {len(largest_indices)} largest indices after DBSCAN")
            if largest_indices.size > 0:
                pcd = pcd.select_by_index(largest_indices)
            else:
                print("Warning: Largest cluster is empty")
        else:
            print("No clusters found")
    else:
        print("All points labeled as noise by DBSCAN")

    print(f"PC {ply_path} has {len(pcd.points)} points in the largest cluster after DBSCAN")
    
    return pcd

def get_lidar_files(input_dir):
    """
    Get list of lidar files to process (only numeric filenames).
    """
    lidar_files = [f for f in glob.glob(os.path.join(input_dir, "lidar/*.ply")) 
                   if os.path.basename(f).replace('.ply', '').isdigit()]
    return lidar_files

def merge_and_filter_point_clouds(pcds, input_dir):
    """
    Merge all point clouds (simplified to avoid memory issues).
    """
    # Merge all point clouds
    print("Merging point clouds...")
    merged_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        merged_pcd += pcd
        print(f"  Added point cloud {i+1}/{len(pcds)} ({len(pcd.points)} points)")

    print(f"Final merged point cloud has {len(merged_pcd.points)} points")
    
    filtered_pcd = filter_dense_regions(merged_pcd)
    filtered_pcd = apply_dbscan_clustering(filtered_pcd, input_dir)
    
    # Convert to CPU before outlier removal to avoid CUDA issues
    if hasattr(filtered_pcd, 'cpu') and callable(getattr(filtered_pcd, 'cpu', None)):
        filtered_pcd = filtered_pcd.cpu()
    
    # remove_statistical_outlier returns (pointcloud, indices), we only want the pointcloud
    filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return filtered_pcd

def save_and_visualize_point_cloud(merged_pcd, input_dir):
    """
    Save the final merged point cloud and optionally visualize it.
    """
    output_path = input_dir + "final_lidar_pcd.ply"
    
    # Convert to CPU point cloud if it's a CUDA point cloud
    if hasattr(merged_pcd, 'cpu') and callable(getattr(merged_pcd, 'cpu', None)):
        merged_pcd = merged_pcd.cpu()
    
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"✅ Final merged point cloud saved to: {output_path}")
    print(f"✅ Total points: {len(merged_pcd.points)}")
    
    # Try to visualize, but don't fail if no display
    try:
        print("Attempting to display result...")
        o3d.visualization.draw_geometries([merged_pcd], window_name="Final Result")
    except Exception as e:
        print(f"⚠️  Could not display (no display available): {e}")
        print("✅ File saved successfully - you can view it with other tools")

def main():
    """
    Main function to orchestrate the point cloud processing pipeline.
    """
    input_dir = "./data/object_sdf_data/lego/"
    
    # Get or create the bounding box
    bbox = create_or_load_bbox(input_dir)
    if bbox is None:
        print("Bounding box creation failed. Exiting.")
        return
    print(f"Using bbox: min={bbox.min_bound}, max={bbox.max_bound}")
    
    # Get list of lidar files to process
    lidar_files = get_lidar_files(input_dir)
    print(f"Found {len(lidar_files)} lidar files to process")
    
    # Process each point cloud (limit to first 30 files)
    pcds = []
    num_files_to_process = min(30, len(lidar_files))  # Process first 30 files
    print(f"Processing first {num_files_to_process} files out of {len(lidar_files)} total files")
    
    for i, ply_path in enumerate(lidar_files[:num_files_to_process]):
        print(f"Processing {i+1}/{num_files_to_process}: {ply_path}")
        
        # Load point cloud and crop with bounding box
        pcd = o3d.io.read_point_cloud(ply_path)
        cropped_pcd = pcd.crop(bbox)
        print(f"  Cropped from {len(pcd.points)} to {len(cropped_pcd.points)} points")
        
        # Process the cropped point cloud
        processed_pcd = process_single_point_cloud(cropped_pcd, ply_path)
        pcds.append(processed_pcd)
        
        # Clear memory
        del pcd, cropped_pcd
        gc.collect()
    
    print(f"Processed {len(pcds)} point clouds")
    
    # Merge all point clouds
    print("Merging all point clouds...")
    merged_pcd = merge_and_filter_point_clouds(pcds, input_dir)
    
    # Save final result
    save_and_visualize_point_cloud(merged_pcd, input_dir)

if __name__ == "__main__":
    main()







