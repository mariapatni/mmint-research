import bpy
import os

# Path to the downloaded addon zip file
addon_path = os.path.abspath("io_mesh_ply.zip")

# Install the addon
bpy.ops.preferences.addon_install(filepath=addon_path)

# Enable the addon
bpy.ops.preferences.addon_enable(module="io_mesh_ply")

# Save user preferences
bpy.ops.wm.save_userpref() 