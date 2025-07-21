#!/usr/bin/env python3
import trimesh
import os

# Path to the original mesh
mesh_path = 'data/object_sdf_data/lego/exports/mesh_highres/mesh_scaled.obj'

# Check if the file exists
if not os.path.exists(mesh_path):
    print(f"Error: Mesh file not found at {mesh_path}")
    exit(1)

print(f"Loading mesh from: {mesh_path}")
mesh = trimesh.load(mesh_path)

print(f"Original mesh bounds:")
print(f"  X: {mesh.bounds[0][0]:.6f} to {mesh.bounds[1][0]:.6f}")
print(f"  Y: {mesh.bounds[0][1]:.6f} to {mesh.bounds[1][1]:.6f}")
print(f"  Z: {mesh.bounds[0][2]:.6f} to {mesh.bounds[1][2]:.6f}")

# Apply scale factor of 0.001
scale_factor = 0.001
mesh.apply_scale(scale_factor)

print(f"\nScaled mesh bounds (scale factor: {scale_factor}):")
print(f"  X: {mesh.bounds[0][0]:.6f} to {mesh.bounds[1][0]:.6f}")
print(f"  Y: {mesh.bounds[0][1]:.6f} to {mesh.bounds[1][1]:.6f}")
print(f"  Z: {mesh.bounds[0][2]:.6f} to {mesh.bounds[1][2]:.6f}")

# Export the scaled mesh
output_path = 'data/object_sdf_data/lego/exports/mesh_highres/mesh_scaled_down.obj'
mesh.export(output_path)

print(f"\nScaled mesh saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB") 