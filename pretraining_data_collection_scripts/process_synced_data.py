import bpy
import os
import shutil
import numpy as np
from mathutils import Quaternion, Vector, Matrix
import mathutils
import math
import bmesh
from math import acos, pi
from pathlib import Path
import random

def debug_print(message):
    print(message)
    # bpy.ops.info.reports_display_update()
    # bpy.data.window_managers[0].windows[0].screen.areas[0].type = 'INFO'

debug_print("Starting script...")

#base = "Users/mariapatni/Documents/GitHub/mmint-research"
base = "../"

data_root_folder = base + "/YCB-Slide/dataset/real/"  # Path to real folders
pose_output_folder = base + "/pretrain_rendered_data/poses/"  # Where to save pose files
tactile_images_folder = base + "/pretrain_rendered_data/tactile_images/"  # Where to save tactile images
stl_base_directory = base + "/YCB-slide/dataset/obj_models/"  # Path to .stl files
render_base_output_folder = base +"/pretrain_rendered_data/renders/"  # Where to save rendered images


debug_print("Creating output directories if they don't exist...")
os.makedirs(pose_output_folder, exist_ok=True)
os.makedirs(tactile_images_folder, exist_ok=True)
os.makedirs(render_base_output_folder, exist_ok=True)

debug_print("Checking directories...")
assert os.path.exists(data_root_folder), f"Data root folder does not exist: {data_root_folder}"
assert os.path.exists(pose_output_folder), f"Pose output folder does not exist: {pose_output_folder}"
assert os.path.exists(tactile_images_folder), f"Tactile images folder does not exist: {tactile_images_folder}"
assert os.path.exists(render_base_output_folder), f"Render base output folder does not exist: {render_base_output_folder}"
assert os.path.exists(stl_base_directory), f"STL base directory does not exist: {stl_base_directory}"

debug_print("Setting up rendering settings...")
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.render.resolution_x = 480
bpy.context.scene.render.resolution_y = 640
bpy.context.scene.render.resolution_percentage = 50
bpy.context.scene.cycles.samples = 1
bpy.context.scene.cycles.use_denoising = False
bpy.context.scene.render.use_persistent_data = True
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.preview_samples = 1
bpy.context.scene.cycles.max_bounces = 1
bpy.context.scene.cycles.caustics_reflective = False
bpy.context.scene.cycles.caustics_refractive = False

import bpy
import mathutils
from mathutils import Vector, Quaternion


def generate_tactile_pose_path(object_name, dataset_number, poses):
    debug_print(f"Generating tactile pose path for {object_name}, dataset {dataset_number}")
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.select_all(action='DESELECT')
    stl_file = os.path.join(stl_base_directory, f"{object_name}/nontextured.stl")
    debug_print(f"Loading STL file: {stl_file}")
    if not load_stl_file(stl_file):
        debug_print("Failed to load STL file")
        return
    
    obj = bpy.context.selected_objects[0]
    obj = bpy.context.object
    obj = setup_object(obj)
    
    
    
    cam = setup_camera()
    
    output_base_path = os.path.join(render_base_output_folder, f"{object_name}_dataset_{dataset_number}")
    debug_print(f"Creating output directory: {output_base_path}")
    os.makedirs(output_base_path, exist_ok=True)
    
    debug_print(f"Processing poses... Total poses: {len(poses['DIGIT'])}")
    pose_index = 0
    while pose_index < len(poses['DIGIT']):
        debug_print(f"Processing pose {pose_index}")
        P_c = Vector(poses['DIGIT'][pose_index][:3])
        P_o = Vector(poses[object_name][pose_index][:3])
        quat_c = poses['DIGIT'][pose_index][3:]
        quat_o = poses[object_name][pose_index][3:]
        debug_print(f"Original camera quaternion: {quat_c}")
        #Q_c = Quaternion((quat_c[1], quat_c[2], quat_c[3], quat_c[0])) buggy line of code
        Q_o = Quaternion((quat_o[1], quat_o[2], quat_o[3], quat_o[0]))
        Q_c = quat_c
        
        P_c_relative = P_c - P_o
        debug_print(f"Original camera position: {P_c}")
        debug_print(f"Original object position: {P_o}")
        debug_print(f"Relative camera position: {P_c_relative}")
        
        obj.rotation_quaternion = Q_o
        obj.location = Vector((0, 0, 0))
        
        debug_print(f"Camera Q_c: {Q_c}")
        
        cam.rotation_quaternion = Q_c
        cam.location = P_c_relative
        
#        point_camera_at(cam, Vector((0, 0, 0)))
        #add_marker(P_c_relative, color=(1, 0, 0, 1), size=0.005)
        #add_marker(Vector((0, 0, 0)), color=(0, 1, 0, 1), size=0.005)
        #add_coordinate_frame(Vector((0, 0, 0)), scale=0.1)
        #add_camera_direction_arrow(P_c_relative, cam.rotation_quaternion, length=0.025, color=(1, 0, 0, 1))
        #add_line_between_points(P_c_relative, Vector((0, 0, 0)), color=(0, 0, 1, 1))
        
        
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = os.path.join(output_base_path, f"pose_{pose_index:04d}.png")
        bpy.ops.render.render(write_still=True)
        
        pose_index += 1

def capture_render_image_poses():
    debug_print("Starting render image pose capture...")
    object_list = [
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "042_adjustable_wrench",
        "048_hammer",
        "055_baseball",
        "035_power_drill"
    ]
    for pose_file in os.listdir(pose_output_folder):
        if pose_file.endswith(".npy"):# and pose_file[:-14] == "035_power_drill":
            debug_print(f"Processing pose file: {pose_file}")
            object_name, dataset_number = pose_file.replace(".npy", "").split("_dataset_")
#            if dataset_number != "2":
#                debug_print(f"Skipping {pose_file} for testing")
#                continue
            pose_path = os.path.join(pose_output_folder, pose_file)
            poses = np.load(pose_path, allow_pickle=True).item()
            generate_tactile_pose_path(object_name, dataset_number, poses)

def add_marker(location, size=0.0005, color=(1, 0, 0, 1)):
    debug_print(f"Adding marker at location {location}")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    marker = bpy.context.object
    marker_mat = bpy.data.materials.new(name="MarkerMaterial")
    marker_mat.diffuse_color = color
    marker.data.materials.append(marker_mat)
    bpy.context.view_layer.update()

def add_coordinate_frame(location, scale=0.5):
    debug_print(f"Adding coordinate frame at location {location}")
    colors = {'X': (1, 0, 0, 1), 'Y': (0, 1, 0, 1), 'Z': (0, 0, 1, 1)}
    rotations = {
        'X': mathutils.Quaternion((0, 1, 0), math.radians(90)),
        'Y': mathutils.Quaternion((1, 0, 0), math.radians(-90)),
        'Z': mathutils.Quaternion((1, 0, 0), 0)
    }
    for axis, color in colors.items():
        debug_print(f"Creating {axis} axis")
        bpy.ops.mesh.primitive_cylinder_add(radius=scale * 0.005, depth=scale, location=location)
        arrow = bpy.context.object
        arrow.name = f"Arrow_{axis}"
        arrow.rotation_mode = 'QUATERNION'
        arrow.rotation_quaternion = rotations[axis]
        mat = bpy.data.materials.new(name=f"Mat_{axis}")
        mat.diffuse_color = color
        mat.use_nodes = True
        arrow.data.materials.append(mat)
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()

def load_stl_file(filepath):
    debug_print(f"Loading STL file: {filepath}")
    if not os.path.exists(filepath):
        debug_print("STL file does not exist")
        return False
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj
        debug_print("STL file loaded successfully")
        return True
    except Exception as e:
        debug_print(f"Error loading STL file: {e}")
        return False

def setup_camera():
    debug_print("Setting up camera...")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        debug_print("Creating new camera")
        bpy.ops.object.camera_add(location=(0, 0, 1))
        cam = bpy.context.active_object
        cam.name = "Camera"
        bpy.context.scene.camera = cam
    cam.rotation_euler = (0, 0, 0)
    cam.rotation_mode = 'QUATERNION'
    cam.location = Vector((0, 0, 1))
    cam.data.clip_start = 0.001  # Minimum distance the camera can see
    cam.data.clip_end = 1000.0  # Maximum distance the camera can see
    cam.data.lens = 50 #Default is 50
    cam.data.type = 'PERSP'
    
    # Create a new light
    light_data = bpy.data.lights.new(name="Camera_Light", type='POINT')
    light = bpy.data.objects.new(name="Camera_Light", object_data=light_data)
    light.parent = cam
    light.location = (0, 0, -1)  # Always stays behind the camera in local space
    bpy.context.view_layer.update()
    return cam

def setup_object(obj):
    debug_print("Setting up object...")
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    obj.show_axis = True
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = (1, 0, 0, 0)
    
    #bpy.context.space_data.shading.type = 'SOLID'

    return obj
    
def create_material():
    
    # Create a new material
    mat = bpy.data.materials.new(name="MyMaterial")
    mat.use_nodes = True
    
    # Get the Principled BSDF shader (default shader in Blender)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Light blue (RGBA)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    
    return mat
def point_camera_at(cam, target_position):
    debug_print(f"Pointing camera at {target_position}")
    direction = (target_position - cam.location).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = rot_quat
    bpy.context.view_layer.update()

def add_camera_direction_arrow(position, quaternion, length=0.1, color=(1, 0, 0, 1)):
    debug_print(f"Adding camera direction arrow at {position}")
    
    forward_dir = quaternion @ Vector((0, 0, -1))
    forward_dir = forward_dir.normalized()
    end_point = position + (forward_dir * length)
    
    bpy.ops.mesh.primitive_cylinder_add(
        radius=length/20,
        depth=length*0.8,
        location=position + (forward_dir * length*0.4)
    )
    cylinder = bpy.context.active_object
    cylinder.rotation_mode = 'QUATERNION'
    cylinder.rotation_quaternion = quaternion
    
    direction = end_point - position
    rot_quat = direction.to_track_quat('Z', 'Y')
    cylinder.rotation_mode = 'QUATERNION'
    cylinder.rotation_quaternion = rot_quat
    bpy.ops.mesh.primitive_cone_add(
        radius1=length/20,
        radius2=0,
        depth=length*0.2,
        location=end_point - (forward_dir * length*0.1)
    )
    cone = bpy.context.active_object
    cone.rotation_mode = 'QUATERNION'
    cone.rotation_quaternion = rot_quat
    
    for obj in [cylinder, cone]:
        mat = bpy.data.materials.new(name="ArrowMaterial")
        mat.diffuse_color = color
        obj.data.materials.append(mat)

def add_line_between_points(start, end, color=(0, 0, 1, 1)):
    debug_print(f"Adding line from {start} to {end}")
    curve = bpy.data.curves.new('debug_line', 'CURVE')
    curve.dimensions = '3D'
    polyline = curve.splines.new('POLY')
    polyline.points.add(1)
    polyline.points[0].co = (*start, 1)
    polyline.points[1].co = (*end, 1)
    curveObj = bpy.data.objects.new('debug_line', curve)
    mat = bpy.data.materials.new(name="LineMaterial")
    mat.diffuse_color = color
    curveObj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(curveObj)

def process_synced_data():
    debug_print("Processing synced data...")
    for subdir, _, files in os.walk(data_root_folder):
        for file in files:
            if file == "synced_data.npy":
                debug_print(f"Processing {file} in {subdir}")
                npy_file = os.path.join(subdir, file)
                data = np.load(npy_file, allow_pickle=True).item()
                object_name = os.path.basename(os.path.dirname(subdir))
                dataset_number = os.path.basename(subdir).split('_')[-1]
                poses = data.get('poses', None)
                if poses is not None:
                    debug_print(f"Saving poses for {object_name} dataset {dataset_number}")
                    np.save(os.path.join(pose_output_folder, f"{object_name}_dataset_{dataset_number}.npy"), poses)
                else:
                    debug_print(f"No poses found for {object_name} dataset {dataset_number}")
                    continue
                frames_dir = os.path.join(subdir, "frames")
                target_dir = os.path.join(tactile_images_folder, f"{object_name}_dataset_{dataset_number}_frames")
                os.makedirs(target_dir, exist_ok=True)
                if os.path.exists(frames_dir):
                    debug_print(f"Copying frames from {frames_dir} to {target_dir}")
                    for frame_file in os.listdir(frames_dir):
                        frame_path = os.path.join(frames_dir, frame_file)
                        if os.path.isfile(frame_path):
                            shutil.copy(frame_path, target_dir)

def main():
    debug_print("Starting main function...")
    capture_render_image_poses()
    debug_print("Script completed successfully")

if __name__ == "__main__":
    main()