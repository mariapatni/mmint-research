import bpy
import os

print("Hello\n")
print(bpy.context.preferences.addons.keys())

# Set input and output directories
stl_root_folder = "./YCB-Slide/dataset/obj_models/"  # Path to STL models
output_folder = "./renders/"  # Where to save rendered images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Set rendering settings
bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles for better quality
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

def process_stl_files():
    print(f"Searching for STL files in: {stl_root_folder}")
    
    # Walk through all subdirectories
    for subdir, _, files in os.walk(stl_root_folder):
        for file in files:
            if file.endswith(".stl"):
                stl_file = os.path.join(subdir, file)
                print(f"\nProcessing: {stl_file}")
                
                # Clear previous objects
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Import STL file
                bpy.ops.import_mesh.stl(filepath=stl_file)
                obj = bpy.context.selected_objects[0]
                
                # Center object
                obj.location = (0, 0, 0)
                obj.rotation_euler = (0, 0, 0)
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')

                # Set camera
                cam = bpy.data.objects.get("Camera")
                if not cam:
                    bpy.ops.object.camera_add(location=(0, 0, 0.5))
                    cam = bpy.context.object
                    cam.data.type = 'ORTHO'
                    cam.rotation_euler = (0, 0, 0)  # Point straight down
                    cam.data.ortho_scale = 0.3  # Adjust this to frame the object properly

                bpy.context.scene.camera = cam

                # Add lighting
                light = bpy.data.objects.get("Light")
                if not light:
                    bpy.ops.object.light_add(type='SUN', location=(0, 0, 3))
                    light = bpy.context.object
                    light.rotation_euler = (0, 0, 0)
                    light.data.energy = 5.0  # Increased for better color visibility

                # Set render settings for better color
                bpy.context.scene.render.engine = 'CYCLES'
                bpy.context.scene.render.film_transparent = False
                bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                
                # White background
                bpy.context.scene.world.use_nodes = True
                bg_node = bpy.context.scene.world.node_tree.nodes["Background"]
                bg_node.inputs[0].default_value = (1, 1, 1, 1)  # White background

                # Get object name from path
                object_name = os.path.basename(stl_file).replace('.stl', '')
                output_path = os.path.join(output_folder, f"{object_name}.png")

                # Set render output path
                bpy.context.scene.render.filepath = output_path

                # Render and save image
                bpy.ops.render.render(write_still=True)

                print(f"Rendered: {output_path}")

                # Delete object before next iteration
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

# Run batch rendering
process_stl_files()
print("Batch rendering completed!")
