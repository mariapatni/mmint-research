import numpy as np
from scipy.spatial.transform import Rotation as R

# Define start and end orientations as Euler angles (XYZ order, in degrees)
start_euler = [-90, -67, -89]  # Replace with actual values
end_euler = [66, -250, 70]    # Replace with actual values

# Convert Euler angles to rotation matrices
R_s = R.from_euler('xyz', start_euler, degrees=True).as_matrix()
R_e = R.from_euler('xyz', end_euler, degrees=True).as_matrix()

# Compute the transformation matrix
R_transform = R_e @ R_s.T

# Print the result
print("Rotation Transformation Matrix:\n", np.round(R_transform, 3))

# Print the matrix in the desired format
print("R = (")
for row in R_transform:
    print(f"      ({row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}),")
print(")")