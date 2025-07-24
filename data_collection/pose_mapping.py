import numpy as np
import os
import argparse
import open3d as o3d

def get_object_path(object_name):
    """
    Get the path to the object folder
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        str: Path to the object directory, or None if not found
    """
    # Get the current directory (data_collection) and go up one level to the parent
    current_dir = os.path.dirname(os.path.abspath(__file__))  # data_collection
    parent_dir = os.path.dirname(current_dir)  # parent directory
    object_path = os.path.join(parent_dir, "data", "collected_data", object_name)
    
    if not os.path.exists(object_path):
        print(f"⚠️  No data directory found for object: {object_name}")
        return None
    
    print(f"📁 Object path: {object_path}")
    return object_path

def get_latest_trial_path(object_name):
    """
    Get the path to the latest trial for the given object
    """
    object_path = get_object_path(object_name)
    if not object_path:
        return None

    # Get all trial directories and find the latest one
    trial_dirs = [d for d in os.listdir(object_path) if d.startswith('trial')]
    if not trial_dirs:
        return None
    
    # Extract trial numbers and find the maximum
    trial_numbers = [int(d.replace('trial', '')) for d in trial_dirs]
    return 'trial' + str(max(trial_numbers))

def load_depth_point_clouds(object_name):
    object_path = get_object_path(object_name)
    if not object_path:
        print(f"❌ Object path not found for: {object_name}")
        return None

    trial_path = get_latest_trial_path(object_name)
    if trial_path is None:
        print(f"❌ No trials found for: {object_name}")
        return None

    depth_maps_path = os.path.join(object_path, trial_path, "gripper_camera", "depth")
    if not os.path.exists(depth_maps_path):
        print(f"❌ No depth maps found for: {object_name}")
        return None
    
    depth_maps = []
    for file in os.listdir(depth_maps_path):
        
        depth_map_path = os.path.join(depth_maps_path, file)
        depth_map = cv2.imread(depth_map_path, -1) / 1e3  # Convert from mm to meters
        depth_map = cv2.resize(depth_map, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_map[(depth_map < 0.001) | (depth_map >= np.inf)] = 0
        
        depth_maps.append(depth_map)

        print(f"📁 Depth map: {depth_map}")


def load_poses(object_name):
    """
    Load the poses.npy file for the given object
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        numpy.ndarray: Array of poses, or None if not found
    """
    
        
    object_path = get_object_path(object_name)
    if not object_path:
        print(f"❌ Object path not found for: {object_name}")
        return None

    trial_path = get_latest_trial_path(object_name)
    if trial_path is None:
        print(f"❌ No trials found for: {object_name}")
        return None
    
    poses_path = os.path.join(object_path, trial_path, "gripper_camera", "poses.npy")
    print(f"📁 Loading poses for: {object_name} under {trial_path}")
    
    try:
        poses = np.load(poses_path)
        print(f"✅ Successfully loaded poses with shape: {poses.shape}")
        return poses
    except Exception as e:
        print(f"❌ Error loading poses: {e}")
        return None

def print_poses(poses):
    """
    Print poses information
    """
    print(f"Array shape: {poses.shape}")
    print(f"Number of poses: {len(poses)}")
    
    for i, pose in enumerate(poses):
        print(f"\nPose {i+1}/{len(poses)}:")
        print(f"   Translation (X, Y, Z): {pose[:3, 3]}")
        print(f"   Rotation matrix:")
        print(f"     {pose[:3, 0]}")
        print(f"     {pose[:3, 1]}")
        print(f"     {pose[:3, 2]}")  

def load_nerf(object_name):
    """
    Load the NeRF mesh from the object's mesh export directory
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        str: Path to the mesh file, or None if not found
    """
    # Get the current directory (data_collection) and go up one level to the parent
    current_dir = os.path.dirname(os.path.abspath(__file__))  # data_collection
    parent_dir = os.path.dirname(current_dir)  # parent directory
    mesh_path = os.path.join(parent_dir, "data", "object_sdf_data", object_name, "mesh_export", "mesh_scaled.obj")
    
    if not os.path.exists(mesh_path):
        print(f"⚠️  No mesh file found for object: {object_name}")
        print(f"   Expected path: {mesh_path}")
        return None
    
    print(f"📁 Mesh path: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)

    # Center the mesh at origin
    # bbox = mesh.get_axis_aligned_bounding_box()
    # center = bbox.get_center()
    # print(f"🔄 Centering mesh by translating by: {-center}")
    
    # # Create translation matrix to move mesh to origin
    # translation_matrix = np.eye(4)
    # translation_matrix[:3, 3] = -center  # Negative center to move to origin
    # #mesh.transform(translation_matrix)

    # Verify centering
    bbox_centered = mesh.get_axis_aligned_bounding_box()
    print(f"📏 Centered mesh bounds:")
    print(f"   Center: {bbox_centered.get_center()}")
    print(f"   Min: {bbox_centered.min_bound}")
    print(f"   Max: {bbox_centered.max_bound}")
    print(f"✅ NeRF mesh loaded successfully")
    print(f"📏 Mesh vertices: {len(mesh.vertices)}")
    print(f"📏 Mesh faces: {len(mesh.triangles)}")
    print(f"📏 Mesh bounding box: {mesh.get_axis_aligned_bounding_box()}")
    print(f"📏 Mesh center: {mesh.get_center()}")
    print(f"📏 Mesh size: {mesh.get_max_bound() - mesh.get_min_bound()}")
    return mesh

def create_point(pose):
    
    # Create a sphere using Open3D's sphere geometry
    
    radius = 0.01
    
    # Create a sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    
    # Move the sphere to the pose position
    pose_position = pose[:3, 3]
    sphere.translate(pose_position)
    
    return sphere
    

def get_d_local(camera_pose, d_global_pose):
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    d_local = R.T @ (d_global_pose[:3, 3])
    return d_local

def main():
    parser = argparse.ArgumentParser(description='Load and analyze poses.npy file and NeRF mesh')
    parser.add_argument('object', type=str, help='Name of the object to analyze')
    args = parser.parse_args()
    

    poses = load_poses(args.object)
    if poses is None:
        print(f"❌ Could not load poses for: {args.object}")
    
    # Load NeRF mesh
    mesh = load_nerf(args.object)
    if mesh is None:
        print(f"❌ Could not load NeRF mesh for: {args.object}")

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.25,  # Size of the coordinate frame axes
        origin=[0, 0, 0],  # Place at origin after centering
    )
    
    camera_poses = np.linalg.inv(poses)
    camera_points = []
    camera_pose_frames = []
    digit_left_points = []
    
    global_left_end_point = np.array([36.57, -169.025, 131.016]) / 1000

    
    

    T = camera_poses[0]
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = poses[0]



    # # Convert global end point from Z-up to Y-up
    # R_z2y = np.array([[1, 0, 0], 
    #                   [0, 0, 1], 
    #                   [0, -1, 0]])
    
    p_end_yup = global_left_end_point

    print("Absolute left end point:", camera_poses[0][:3, 3] + p_end_yup)


    # Step 3: Convert point to homogeneous coordinates
    p_end_yup_h = np.append(p_end_yup, 1)

    # Step 4: Apply inverse pose to get local camera frame coords
    p_end_local_h = T_inv @ p_end_yup_h

    # Step 5: Extract XYZ local coordinates
    p_end_local = p_end_local_h[:3]

    print("End point in camera local frame:", p_end_local)


    p_global_h = T @ p_end_local_h
    p_global = p_global_h[:3]

    print("End point in camera global frame:", p_global)

    

    


   
    

    

    
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        
        point = create_point(camera_pose)
        camera_points.append(point)
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(camera_pose)
        camera_pose_frames.append(frame)



        # digit_left_point = R @ d_local_left + t
        # digit_left_pose = np.eye(4)
        # digit_left_pose[:3, 3] = digit_left_point
        # digit_left_point = create_point(digit_left_pose)
        # digit_left_points.append(digit_left_point)

        

    
    o3d.visualization.draw_geometries([mesh, camera_points[0], camera_pose_frames[0], coord_frame])
    
if __name__ == "__main__":
    main()
