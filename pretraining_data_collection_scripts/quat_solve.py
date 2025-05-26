import numpy as np

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z], dtype=float)

# Your start & end quaternions (original problem statement)
q_start_raw = np.array([-0.1358,   0.6952,   -0.1375,   0.6923])
q_end_raw   = np.array([-0.623,    0.186, -0.751, 0.122])

# Normalize them (generally good practice):
q_start = normalize_quaternion(q_start_raw)
q_end   = normalize_quaternion(q_end_raw)

print("q_start =", q_start)
print("q_end   =", q_end)

# The q_delta you got:
q_delta = np.array([ 0.68198488,  0.62024059, -0.3670193, -0.12447918])

# Let's see what q_delta * q_start gives:
q_test = quaternion_multiply(q_delta, q_start)

# Compare q_test with q_end:
print("\nq_test = q_delta * q_start =", q_test)
print("q_end                      =", q_end)

# Let's look at the difference:
diff = q_test - q_end
print("\nDifference:", diff)

# Also compare normalized versions (sometimes floating-point sign/scaling can differ):
q_test_norm = normalize_quaternion(q_test)
q_end_norm  = normalize_quaternion(q_end)
diff_norm = q_test_norm - q_end_norm
print("\nNormalized difference:", diff_norm)
