import open3d as o3d
import numpy as np



# Load point clouds
nerf_pcd = o3d.io.read_point_cloud("cropped_nerf.ply")
lidar_pcd = o3d.io.read_point_cloud("merged_pcd.ply")


print("Normalizing point clouds...")

aabb_nerf = nerf_pcd.get_axis_aligned_bounding_box()
aabb_lidar = lidar_pcd.get_axis_aligned_bounding_box()
aabb_nerf.color = (0, 1, 0)     
aabb_lidar.color = (1, 0, 0) 

obb_nerf = nerf_pcd.get_oriented_bounding_box()
obb_lidar = lidar_pcd.get_oriented_bounding_box()
obb_nerf.color = (0, 0, 1)     
obb_lidar.color = (0, 0, 1) 

obb_nerf_sorted = np.sort(obb_nerf.extent)[::-1]
obb_lidar_sorted = np.sort(obb_lidar.extent)[::-1]
aabb_nerf_sorted = np.sort(aabb_nerf.get_extent())[::-1]
aabb_lidar_sorted = np.sort(aabb_lidar.get_extent())[::-1]

print("LIDAR Bounding boxes:")
print(aabb_lidar_sorted)
print("NERF Bounding boxes:")
print(obb_nerf_sorted)

scale_ratios = aabb_lidar_sorted / obb_nerf_sorted
print("Scale ratios:")
print(scale_ratios)

uniform_scale = scale_ratios[1]
print("Uniform scale:")
print(uniform_scale)


nerf_pcd.scale(uniform_scale, center=nerf_pcd.get_center())

print("Nerf Dimensions")
dimensions_nerf = nerf_pcd.get_max_bound() - nerf_pcd.get_min_bound()
print(np.sort(dimensions_nerf)[::-1])

print("Lidar Dimensions")
dimensions_lidar = lidar_pcd.get_max_bound() - lidar_pcd.get_min_bound()
print(np.sort(dimensions_lidar)[::-1])



print("Aligning point clouds centers...")
nerf_center = nerf_pcd.get_center()
lidar_center = lidar_pcd.get_center()

translation = lidar_center - nerf_center
print(translation)

nerf_pcd.translate(translation)

# nerf_pc = nerf_pcd
# lidar_pc = lidar_pcd

# voxel_size = 0.005
# lidar_pc = lidar_pcd_down.voxel_down_sample(voxel_size)
# nerf_pc = nerf_pcd_down.voxel_down_sample(voxel_size)


radius = 0.05 * 2
max_nn = 30

for pc in [nerf_pcd, lidar_pcd]:
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pc.orient_normals_consistent_tangent_plane(k=30)

init_transform = np.eye(4)
distance_threshold = 0.001

reg_colored = o3d.pipelines.registration.registration_colored_icp(
    nerf_pcd, lidar_pcd,
    distance_threshold,
    init_transform,
    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                      relative_rmse=1e-6,
                                                      max_iteration=5000)
)

nerf_pcd.transform(reg_colored.transformation)







# trans_init = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

# reg = o3d.pipelines.registration.registration_icp(
#     nerf_pcd_down, lidar_pcd_down,
#     max_correspondence_distance=0.005,
#     init=np.eye(4),
#     estimation_method=trans_init
# )

# print("Applying transformation to original NeRF point cloud...")
# nerf_pcd.transform(reg.transformation)

# print("Saving scaled & aligned NeRF point cloud...")
# o3d.io.write_point_cloud("nerf_scaled.ply", nerf_pcd)

# Visualize
#nerf_pcd_down, bbox_nerf, bbox_lidar, lidar_pcd_down


# Blue NeRF bounding box
obb_nerf = nerf_pcd.get_oriented_bounding_box()
obb_nerf.color = (0, 0, 1)

aabb_nerf = nerf_pcd.get_axis_aligned_bounding_box()
aabb_nerf.color = (0, 0, 1)

# Red LIDAR Bounding box
obb_lidar = lidar_pcd.get_oriented_bounding_box()
obb_lidar.color = (1, 0, 0)

aabb_lidar = lidar_pcd.get_axis_aligned_bounding_box()
aabb_lidar.color = (1, 0, 0)
lidar_pcd.paint_uniform_color([1, 0, 0])

all_objects = [nerf_pcd, obb_nerf, lidar_pcd, aabb_lidar]
o3d.visualization.draw_geometries(all_objects, point_show_normal=False)
