import numpy as np
import open3d as o3d
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# Configuration
input_dir = "./data/object_sdf_data/lego"
nerf_file = "final_processed_nerf.ply"

# Processing parameters
CLUSTERING_EPS = 0.005
CLUSTERING_MIN_POINTS = 25
DENSE_FILTER_RADIUS = 0.003
DENSE_FILTER_MIN_NEIGHBORS = 20
NORMAL_RADIUS = 0.03
NORMAL_MAX_NN = 100
NORMAL_K = 1000


def load_point_cloud(file_path):
    """Load a point cloud from file with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded point cloud with {len(pcd.points)} points from {file_path}")
    return pcd

def filter_dense_regions(pcd, radius=0.01, min_neighbors=30):
    """Retain only points that have >= min_neighbors within radius."""
    points = np.asarray(pcd.points)
    tree = KDTree(points)

    # Count neighbors for each point
    densities = np.array([len(tree.query_ball_point(p, radius)) for p in points])

    # Keep only dense points
    keep_indices = np.where(densities >= min_neighbors)[0]
    print(f"Keeping {len(keep_indices)} of {len(points)} points (dense regions)")

    return pcd.select_by_index(keep_indices)

def keep_largest_cluster(pcd, eps=0.008, min_points=8):
    """Keep only the largest continuous cluster using DBSCAN."""
    print("Applying DBSCAN to keep largest continuous cluster...")
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if len(labels) != len(pcd.points):
        raise ValueError("DBSCAN label count does not match point cloud size!")

    valid_mask = labels >= 0
    if np.any(valid_mask):
        bincounts = np.bincount(labels[valid_mask])
        print(f"Found {len(bincounts)} clusters")
        
        if bincounts.size > 0:
            largest_cluster_label = np.argmax(bincounts)
            largest_indices = np.where(labels == largest_cluster_label)[0]
            print(f"Largest cluster has {len(largest_indices)} points")
            
            if largest_indices.size > 0:
                pcd = pcd.select_by_index(largest_indices)
            else:
                print("Warning: Largest cluster is empty")
        else:
            print("No clusters found")
    else:
        print("All points labeled as noise by DBSCAN")

    print(f"Final point cloud has {len(pcd.points)} points after keeping largest cluster")
    return pcd

def apply_dense_filtering(pcd, radius=0.004, min_neighbors=30):
    """Apply dense region filtering with fallback parameters."""
    print("Applying dense region filtering...")
    dense_pcd = filter_dense_regions(pcd, radius, min_neighbors)

    # If too aggressive, try more lenient parameters
    if len(dense_pcd.points) == 0:
        print("Dense filtering too strict, trying more lenient parameters...")
        dense_pcd = filter_dense_regions(pcd, radius=0.006, min_neighbors=15)
    
    print(f"Dense point cloud: {len(dense_pcd.points)} points")
    return dense_pcd

def create_poisson_mesh(pcd, depth, output_path, density_percentile=0.05, 
                       normal_radius=0.03, max_nn=100, k=1000, 
                       force_recompute=False):
    """Create and save a Poisson mesh from a point cloud."""
    if os.path.exists(output_path) and not force_recompute:
        print(f"Loading existing Poisson mesh from {output_path}")
        return o3d.io.read_triangle_mesh(output_path)
    
    print(f"Creating Poisson mesh with depth={depth}...")
    
    # Create a copy to avoid modifying the original
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = pcd.points
    if pcd.has_colors():
        pcd_copy.colors = pcd.colors
    if pcd.has_normals():
        pcd_copy.normals = pcd.normals
    
    # Estimate normals if not already present
    if not pcd_copy.has_normals():
        print("Estimating normals...")
        pcd_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
        )
        pcd_copy.orient_normals_consistent_tangent_plane(k=k)
        print(f"Estimated normals for {len(pcd_copy.points)} points")
    
    # Create Poisson mesh
    print(f"Running Poisson reconstruction with depth={depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_copy, depth=depth)
    
    # Clean up mesh based on density
    if densities is not None and len(densities) > 0:
        densities = np.asarray(densities)
        # Check if densities array matches the number of vertices in the mesh
        if len(densities) == len(mesh.vertices):
            density_threshold = np.quantile(densities, density_percentile)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print(f"Removed {np.sum(vertices_to_remove)} low-density vertices")
        else:
            print(f"Warning: Densities array size ({len(densities)}) doesn't match mesh vertices ({len(mesh.vertices)}). Skipping density-based filtering.")
    
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()
    # mesh.remove_non_manifold_edges()
    # mesh.remove_unreferenced_vertices()

    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()

    #mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=30000)
    #mesh.compute_vertex_normals()   


    
    # mesh.compute_vertex_normals()

    # Save mesh
    print(f"Saving mesh to {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def create_alpha_shape_mesh(pcd, alpha, smooth_iterations=0, scale_factor=1.0, sample_points=None):
    """Create an alpha shape mesh from a point cloud with optional processing."""
    print(f"Creating alpha shape mesh with alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh = mesh.remove_non_manifold_edges()
    print(f"Created alpha shape mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Apply smoothing if requested
    if smooth_iterations > 0:
        print(f"Applying Laplacian smoothing with {smooth_iterations} iterations...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)
    
    # Apply scaling if requested
    if scale_factor != 1.0:
        print(f"Scaling mesh by factor {scale_factor}...")
        mesh = mesh.scale(scale_factor, center=mesh.get_center())
    
    # Sample points if requested
    if sample_points is not None:
        print(f"Sampling {sample_points} points from mesh...")
        sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=sample_points)
        print(f"Sampled {len(sampled_pcd.points)} points from alpha mesh")
        
        return mesh, sampled_pcd
    
    
    return mesh

def process_point_cloud():
    """Process the input point cloud with filtering and clustering."""
    print("=== Processing Point Cloud ===")
    
    # Load point cloud
    input_path = os.path.join(input_dir, nerf_file)
    pcd = load_point_cloud(input_path)
    
    # Apply clustering and filtering
    pcd = keep_largest_cluster(pcd, CLUSTERING_EPS, CLUSTERING_MIN_POINTS)
    pcd = apply_dense_filtering(pcd, DENSE_FILTER_RADIUS, DENSE_FILTER_MIN_NEIGHBORS)
    
    return pcd

def create_simplified_meshes(mesh, target_triangle_counts, visualize=False):
    """Create simplified meshes with different levels of simplification."""
    meshes = []
    for target_triangle_count in target_triangle_counts:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy = mesh_copy.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
        mesh_copy.compute_vertex_normals()
        meshes.append(mesh_copy)
    
    if visualize:
        visualize_simplified_meshes(meshes)

    return meshes

def visualize_simplified_meshes(meshes):
    """Visualize a mesh with different levels of simplification in a grid layout."""
    

    # Create a grid visualization using matplotlib
    print("Creating grid visualization...")
    
    # Prepare meshes for visualization
    mesh_names = [f"{len(m.triangles)} triangles" for m in meshes]
    
    # Calculate grid layout
    n_meshes = len(meshes)
    cols = min(4, n_meshes)
    rows = (n_meshes + cols - 1) // cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * rows))
    fig.suptitle(f'Mesh Comparison: {n_meshes} simplified versions', fontsize=16)
    
    for i, (mesh_obj, name) in enumerate(zip(meshes, mesh_names)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Convert mesh to numpy arrays
        vertices = np.asarray(mesh_obj.vertices)
        triangles = np.asarray(mesh_obj.triangles)
        
        # Plot the mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 2], vertices[:, 1], 
                       triangles=triangles, alpha=0.8, edgecolor='black', linewidth=0.1)
        
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Orient the view with Y-axis pointing up
        ax.view_init(elev=20, azim=45)  # elev=20 degrees up, azim=45 degrees rotation
        
        # Set the up vector to Y-axis
        ax.set_proj_type('ortho')
        # Force Y-axis to be up by setting the up vector
        ax._axis3don = True
        # Ensure Y-axis is vertical
        ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

def create_merged_mesh(pcd):
    """Create a merged mesh from alpha shape and original point cloud."""
    print("=== Creating Merged Mesh ===")
    
    output_path = os.path.join(input_dir, "new_merged_mesh.obj")
    
    if os.path.exists(output_path):
        # Create alpha shape mesh with smoothing, scaling, and point sampling
        
        
        
        alpha_mesh, sampled_pcd = create_alpha_shape_mesh(
            pcd, 
            alpha=0.006, 
            smooth_iterations=1000, 
            scale_factor=0.98, 
            sample_points=1
        )

        alpha_mesh2, sampled_pcd2 = create_alpha_shape_mesh(
            pcd, 
            alpha=0.001, 
            smooth_iterations=5, 
            scale_factor=0.95, 
            sample_points=1
        )

        #o3d.visualization.draw_geometries([alpha_mesh2], mesh_show_back_face=True)


        merged = sampled_pcd + pcd + sampled_pcd2
        merged = merged.voxel_down_sample(voxel_size=0.002)
        merged, _ = merged.remove_radius_outlier(nb_points=40, radius=0.01)

        poisson_mesh_12 = create_poisson_mesh(
            merged, 
            depth=12, 
            output_path=input_dir + "/poisson_mesh_12.obj", 
            force_recompute=False
        )

        poisson_mesh_12 = create_poisson_mesh(
            merged, 
            depth=12, 
            output_path=input_dir + "/poisson_mesh_10.obj", 
            force_recompute=False
        )

    #     poisson_mesh_12.remove_unreferenced_vertices()
    #     poisson_mesh_12.remove_duplicated_vertices()
    #     poisson_mesh_12.remove_duplicated_triangles()
    #     poisson_mesh_12.remove_non_manifold_edges()
    #     poisson_mesh_12.remove_degenerate_triangles()
    #     poisson_mesh_12.remove_unreferenced_vertices()


        #
        poisson_mesh_12 = poisson_mesh_12.filter_smooth_laplacian(number_of_iterations=10)
        poisson_mesh_12 = poisson_mesh_12.filter_smooth_taubin(number_of_iterations=10)


        meshes = create_simplified_meshes(poisson_mesh_12, target_triangle_counts=[300, 250, 200, 150, 100], visualize=False)

        other_mesh = meshes[0]
        # other_mesh = other_mesh.filter_smooth_laplacian(number_of_iterations=)

        # o3d.visualization.draw_geometries([meshes[0]], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([other_mesh], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([ other_mesh, alpha_mesh2], mesh_show_back_face=True)
        #o3d.visualization.draw_geometries([poisson_mesh_12, other_mesh], mesh_show_back_face=True)

        # # Create combo mesh from the three meshes
        # combo_mesh = create_combo_mesh(
        #     alpha_mesh=alpha_mesh,
        #     alpha_mesh2=alpha_mesh2, 
        #     poisson_mesh=poisson_mesh_12,
        #     pcd=pcd,
        #     sample_density=30000,  # Reduced for better performance
        #     voxel_size=0.003,      # Slightly larger for cleaner result
        #     outlier_radius=0.015,  # Larger radius for more aggressive outlier removal
        #     outlier_points=30,     # Fewer points required
        #     smoothing_iterations=15,  # More smoothing for smoother edges
        #     simplification_triangles=12000  # Target triangle count
        # )
        
        # # Save the combo mesh
        # combo_output_path = os.path.join(input_dir, "combo_mesh.obj")
        # o3d.io.write_triangle_mesh(combo_output_path, combo_mesh)
        # print(f"Saved combo mesh to {combo_output_path}")
        
        # # Visualize the combo mesh
        # o3d.visualization.draw_geometries([combo_mesh], mesh_show_back_face=True)
        
        # # Also show comparison with original meshes
        # o3d.visualization.draw_geometries([alpha_mesh2, alpha_mesh, poisson_mesh_12, pcd], mesh_show_back_face=True)


        
    
        # o3d.visualization.draw_geometries([alpha_mesh2, alpha_mesh, poisson_mesh_12, pcd], mesh_show_back_face=True)


        
        
        



        exit(0)





        # alpha_mesh2, sampled_pcd2 = create_alpha_shape_mesh(
        #     pcd, 
        #     alpha=0.001, 
        #     smooth_iterations=5, 
        #     scale_factor=1.0, 
        #     sample_points=1
        # )


        # poisson_mesh_8 = create_poisson_mesh(
        #     pcd, 
        #     depth=8, 
        #     output_path=input_dir + "/poisson_mesh_8.obj", 
        #     force_recompute=False
        # )

        



        # #o3d.visualization.draw_geometries([poisson_mesh_12], mesh_show_back_face=True)

        # poisson_mesh_12 = poisson_mesh_12.remove_degenerate_triangles()
        # poisson_mesh_12 = poisson_mesh_12.remove_duplicated_vertices()
        # poisson_mesh_12 = poisson_mesh_12.remove_non_manifold_edges()
        
        # poisson_mesh_12 = poisson_mesh_12.filter_smooth_laplacian(number_of_iterations=100)



        # poisson_mesh_12 = poisson_mesh_12.simplify_quadric_decimation(target_number_of_triangles=10000)
   
        # o3d.visualization.draw_geometries([poisson_mesh_12, alpha_mesh], mesh_show_back_face=True)

        #o3d.visualization.draw_geometries([poisson_mesh_8], mesh_show_back_face=True)


        # poisson_mesh_8 = poisson_mesh_8.remove_degenerate_triangles()
        # poisson_mesh_8 = poisson_mesh_8.remove_duplicated_vertices()
        # poisson_mesh_8 = poisson_mesh_8.remove_duplicated_triangles()
        # poisson_mesh_8 = poisson_mesh_8.remove_non_manifold_edges()
        # poisson_mesh_8 = poisson_mesh_8.remove_unreferenced_vertices()


        # poisson_mesh_8 = poisson_mesh_8.filter_smooth_laplacian(number_of_iterations=10)
        
        # s = sampled_pcd + sampled_pcd2

        # s, _ = s.remove_radius_outlier(nb_points=16, radius=0.01)
        
        
        #o3d.visualization.draw_geometries([combo], mesh_show_back_face=True)
        exit(0)



        
        # pcd_downsampled = pcd.uniform_down_sample(20000)

        
        # mergepcd = sampled_pcd + sampled_pcd2 + pcd_downsampled

        

        


        #mergepcd = mergepcd.remove_duplicated_points()
        #mergepcd = keep_largest_cluster(mergepcd, CLUSTERING_EPS, CLUSTERING_MIN_POINTS)

        # o3d.visualization.draw_geometries([mergepcd], mesh_show_back_face=True)
        # exit(0)
        
        # mergepcd = apply_dense_filtering(mergepcd, DENSE_FILTER_RADIUS, DENSE_FILTER_MIN_NEIGHBORS)
        # mergepcd, _ = mergepcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # mergepcd, _ = mergepcd.remove_radius_outlier(nb_points=16, radius=0.01)
        
        #o3d.visualization.draw_geometries([mergepcd], mesh_show_back_face=True)
        #o3d.visualization.draw_geometries([alpha_mesh, alpha_mesh2], mesh_show_back_face=True)

        
        #Create Poisson mesh from merged point cloud
        # mesh = create_poisson_mesh(
        #     mergepcd, 
        #     depth=6, 
        #     output_path=output_path, 
        #     force_recompute=True
        # )

        # mesh, pcd = create_alpha_shape_mesh(
        #     mergepcd, 
        #     alpha=0.002, 
        #     smooth_iterations=1, 
        #     scale_factor=1.0, 
        #     sample_points=1
        # )

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        exit(0)
        o3d.visualization.draw_geometries([mesh, alpha_mesh2, alpha_mesh], mesh_show_back_face=True)

        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Saved new alpha shape mesh to {output_path}")
        exit(0)


        # alpha_mesh, sampled_pcd = create_alpha_shape_mesh(
        #     pcd, 
        #     alpha=0.01, 
        #     smooth_iterations=5, 
        #     scale_factor=0.95,
        #     sample_points=10
        # )

        # o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)


        
        
        
        # sampled_pcd, _ = sampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # sampled_pcd, _ = sampled_pcd.remove_radius_outlier(nb_points=16, radius=0.01)
        
        # # Merge point clouds
        # merged_pcd = sampled_pcd + pcd
        # merged_pcd = merged_pcd.remove_duplicated_points()

        # merged_pcd = keep_largest_cluster(merged_pcd, CLUSTERING_EPS, CLUSTERING_MIN_POINTS)
        # #merged_pcd = apply_dense_filtering(merged_pcd, DENSE_FILTER_RADIUS, DENSE_FILTER_MIN_NEIGHBORS)
        # merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    


        # #Create Poisson mesh from merged point cloud
        # mesh = create_poisson_mesh(
        #     merged_pcd, 
        #     depth=14, 
        #     output_path=output_path, 
        #     force_recompute=False
        # )

    else:
        print(f"Loading existing mesh from {output_path}")
        mesh = o3d.io.read_triangle_mesh(output_path)
    
    
    # mesh.remove_degenerate_triangles()
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_non_manifold_edges()
    # mesh.remove_unreferenced_vertices()

    # meshes = create_simplified_meshes(mesh, target_triangle_counts=[5000], visualize=False)
        
    # best_mesh = meshes[0]
    # best_mesh.compute_vertex_normals()
    # best_mesh.filter_smooth_laplacian(number_of_iterations=100)


    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    return mesh

def create_combo_mesh(alpha_mesh, alpha_mesh2, poisson_mesh, pcd, 
                     sample_density=50000, voxel_size=0.002, 
                     outlier_radius=0.01, outlier_points=40,
                     smoothing_iterations=10, simplification_triangles=15000):
    """
    Create a combo mesh from multiple meshes and remove low density areas and jagged edges.
    
    Args:
        alpha_mesh: First alpha shape mesh
        alpha_mesh2: Second alpha shape mesh  
        poisson_mesh: Poisson reconstruction mesh
        pcd: Original point cloud
        sample_density: Number of points to sample from each mesh
        voxel_size: Voxel size for downsampling
        outlier_radius: Radius for outlier removal
        outlier_points: Minimum points for outlier removal
        smoothing_iterations: Number of smoothing iterations
        simplification_triangles: Target number of triangles for simplification
    
    Returns:
        combo_mesh: Cleaned and combined mesh
    """
    print("=== Creating Combo Mesh ===")
    
    # Sample points from each mesh to create a dense point cloud
    print("Sampling points from meshes...")
    alpha_points = alpha_mesh.sample_points_poisson_disk(number_of_points=sample_density)
    alpha2_points = alpha_mesh2.sample_points_poisson_disk(number_of_points=sample_density)
    poisson_points = poisson_mesh.sample_points_poisson_disk(number_of_points=sample_density)
    
    # Combine all point clouds
    print("Combining point clouds...")
    combined_pcd = alpha_points + alpha2_points + poisson_points + pcd
    
    # Remove duplicate points
    combined_pcd = combined_pcd.remove_duplicated_points()
    print(f"Combined point cloud has {len(combined_pcd.points)} points after deduplication")
    
    # Downsample to reduce noise and improve processing speed
    print("Downsampling combined point cloud...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled to {len(combined_pcd.points)} points")
    
    # Remove outliers to clean up low density areas
    print("Removing outliers...")
    combined_pcd, _ = combined_pcd.remove_radius_outlier(
        nb_points=outlier_points, 
        radius=outlier_radius
    )
    print(f"After outlier removal: {len(combined_pcd.points)} points")
    
    # Create a new Poisson mesh from the cleaned combined point cloud
    print("Creating Poisson mesh from combined point cloud...")
    combo_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined_pcd, depth=10
    )
    
    # Remove low density areas based on Poisson reconstruction densities
    if densities is not None and len(densities) > 0:
        densities = np.asarray(densities)
        if len(densities) == len(combo_mesh.vertices):
            # Use a more aggressive density threshold to remove more low density areas
            density_threshold = np.quantile(densities, 0.1)  # Keep only top 90% density
            vertices_to_remove = densities < density_threshold
            combo_mesh.remove_vertices_by_mask(vertices_to_remove)
            print(f"Removed {np.sum(vertices_to_remove)} low-density vertices")
    
    # Clean up the mesh
    print("Cleaning up mesh...")
    combo_mesh.remove_unreferenced_vertices()
    combo_mesh.remove_duplicated_vertices()
    combo_mesh.remove_duplicated_triangles()
    combo_mesh.remove_non_manifold_edges()
    combo_mesh.remove_degenerate_triangles()
    combo_mesh.remove_unreferenced_vertices()
    
    # Smooth the mesh to remove jagged edges
    print(f"Applying smoothing with {smoothing_iterations} iterations...")
    combo_mesh = combo_mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iterations)
    
    # Simplify the mesh to reduce complexity while maintaining quality
    print(f"Simplifying mesh to {simplification_triangles} triangles...")
    combo_mesh = combo_mesh.simplify_quadric_decimation(target_number_of_triangles=simplification_triangles)
    
    # Recompute normals
    combo_mesh.compute_vertex_normals()
    
    print(f"Final combo mesh has {len(combo_mesh.vertices)} vertices and {len(combo_mesh.triangles)} triangles")
    
    return combo_mesh

def main():
    """Main function to orchestrate the point cloud processing pipeline."""
    print("Starting Point Cloud Processing Pipeline")
    print("=" * 50)
    
    try:
        # Process point cloud
        # processed_pcd = process_point_cloud()
        
        # # Create merged mesh
        # final_mesh = create_merged_mesh(processed_pcd)

        obj_path = input_dir + "/exports/mesh/mesh.obj"
        material_path = input_dir + "/exports/mesh/material.mtl"
        texture_path = input_dir + "/exports/mesh/material_0.png"
        
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)

        
        print(mesh)

        pcd = mesh.sample_points_uniformly(number_of_points=200000)
        pcd = keep_largest_cluster(pcd, eps=0.008, min_points=8)

        crop_bbox = pcd.get_oriented_bounding_box()
        crop_bbox.color = (1, 0, 0)

        # Print bounding box information
        print("=== Bounding Box Information ===")
        print(f"Center: {crop_bbox.center}")
        print(f"Extent: {crop_bbox.extent}")
        print(f"R (rotation matrix):")
        print(crop_bbox.R)
        
        # Calculate the 8 corners of the bounding box
        corners = crop_bbox.get_box_points()
        print(f"8 corners of bounding box:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: {corner}")
        
        # Also get the axis-aligned bounding box for comparison
        aabb = pcd.get_axis_aligned_bounding_box()
        print(f"\nAxis-aligned bounding box:")
        print(f"  Min: {aabb.min_bound}")
        print(f"  Max: {aabb.max_bound}")
        print(f"  Center: {aabb.get_center()}")
        print(f"  Extent: {aabb.get_extent()}")

        o3d.visualization.draw_geometries([pcd, crop_bbox], mesh_show_back_face=True)        
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        
        exit(0)





        #mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=200000)
        #mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        exit(0)
        
        print("=" * 50)
        print("Pipeline completed successfully!")
        
        # Visualize the final result
        #print("Visualizing final mesh...")
        #o3d.visualization.draw_geometries([final_mesh, processed_pcd], mesh_show_back_face=True)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
