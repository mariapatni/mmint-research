import open3d as o3d
import glob
import os
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = "./data/object_sdf_data/lego/"
PLY_DIR = INPUT_DIR + "lidar"                  # Path to your .ply files
OUTPUT_FILE = "merged_lidar.ply"       # Output file name
DOWNSAMPLE = True                # Set True to voxel downsample
VOXEL_SIZE = 0.01                # ~5mm (if DOWNSAMPLE = True)



def merge_pointclouds():
    ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "*.ply")))
    if not ply_files:
        print("No .ply files found in", PLY_DIR)
        return

    print(f"Found {len(ply_files)} .ply files. Merging...")

    merged_pcd = o3d.geometry.PointCloud()

    i = 0
    for ply_file in tqdm(ply_files, desc="Merging"):
        pcd = o3d.io.read_point_cloud(ply_file)
        if i % 10 == 0:
            merged_pcd += pcd
        i += 1

    if DOWNSAMPLE:
        print("Downsampling point cloud...")
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    print(f"Saving merged point cloud to {INPUT_DIR + OUTPUT_FILE}...")
    o3d.io.write_point_cloud(INPUT_DIR + OUTPUT_FILE, merged_pcd)
    print("Done.")

    # visualize the merged point cloud
    o3d.visualization.draw_geometries([merged_pcd])


if __name__ == "__main__":
    merge_pointclouds()

    
