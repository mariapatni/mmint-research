import sqlite3
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import random
import argparse

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def load_keypoints(image_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images WHERE name = ?", (image_name,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None
    image_id = row[0]
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    keypoints = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6)
    return keypoints

def visualize_features(image_path, keypoints, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    for x, y, scale, orientation, response, octave in keypoints:
        x_int, y_int = int(x), int(y)
        radius = 3
        cv2.circle(img, (x_int, y_int), radius, (0, 255, 0), 1)
        angle = orientation
        line_length = 10
        end_x = int(x_int + line_length * np.cos(angle))
        end_y = int(y_int + line_length * np.sin(angle))
        cv2.line(img, (x_int, y_int), (end_x, end_y), (0, 255, 0), 1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def get_image_name_id_map(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()
    conn.close()
    id_to_name = {image_id: name for image_id, name in images}
    name_to_id = {name: image_id for image_id, name in images}
    return id_to_name, name_to_id

def get_matches(db_path):
    # Inlier matches from two_view_geometries
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, data FROM two_view_geometries")
    inlier_geometries = cursor.fetchall()
    conn.close()
    extracted_inliers = []
    geometry_size = 9 * 8  # 9 float64 = 72 bytes
    for pair_id, data in inlier_geometries:
        if data is None or len(data) <= geometry_size:
            continue
        matches_blob = data[geometry_size:]
        try:
            inlier_matches = np.frombuffer(matches_blob, dtype=np.uint32).reshape(-1, 2)
            extracted_inliers.append((pair_id, inlier_matches))
            if pair_id == 1 and len(matches_blob) > 0:
                print(f"Pair 0000.png <-> 0050.png:")
                print(f"  matches_blob length: {len(matches_blob)}")
                print(f"  inlier_matches shape: {inlier_matches.shape}")
        except Exception as e:
            print(f"Error unpacking inliers for pair_id {pair_id}: {e}")
            continue
    return extracted_inliers

def pair_id_to_image_ids(pair_id):
    pair_id = np.uint64(pair_id)
    image_id2 = int(pair_id // (2**32))
    image_id1 = int(pair_id % (2**32))
    return image_id1, image_id2

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.8
        value = 0.95
        hsv = np.array([[[hue * 180, saturation * 255, value * 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(map(int, bgr[0][0])))
    return colors

def visualize_matches(image1_path, image2_path, keypoints1, keypoints2, matches, output_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        print(f"Could not read one or both images: {image1_path}, {image2_path}")
        return
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"\nImage 1: {image1_path} ({w1}x{h1})")
    print(f"Image 2: {image2_path} ({w2}x{h2})")
    print(f"Keypoints 1: {len(keypoints1)}, Keypoints 2: {len(keypoints2)}")
    print(f"Num matches (raw): {len(matches)}")

    valid_matches = []
    for idx1, idx2 in matches:
        if idx1 == 4294967295 or idx2 == 4294967295:
            continue
        if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
            valid_matches.append((idx1, idx2))

    print(f"Num valid matches: {len(valid_matches)}")

    if len(valid_matches) > 10:
        selected_matches = random.sample(valid_matches, 10)
    else:
        selected_matches = valid_matches

    colors = generate_distinct_colors(len(selected_matches))
    vis_img = np.hstack((img1, img2))

    print("\nDetailed match information:")
    for i, (idx1, idx2) in enumerate(selected_matches):
        color = colors[i]
        x1, y1 = int(keypoints1[idx1][0]), int(keypoints1[idx1][1])
        x2, y2 = int(keypoints2[idx2][0]), int(keypoints2[idx2][1])

        if i == 0:
            print(f"Image1: {image1_path}, shape: {img1.shape}")
            print(f"Image2: {image2_path}, shape: {img2.shape}")
            print(f"Keypoint1: {keypoints1[idx1]}")
            print(f"Keypoint2: {keypoints2[idx2]}")

        print(f"\nMatch {i}:")
        print(f"  Image 1: point {idx1} at ({x1}, {y1})")
        print(f"  Image 2: point {idx2} at ({x2}, {y2})")
        print(f"  Full keypoint 1 data: {keypoints1[idx1]}")
        print(f"  Full keypoint 2 data: {keypoints2[idx2]}")

        if x1 >= w1 or y1 >= h1:
            print(f"  WARNING: Image 1 point ({x1}, {y1}) outside bounds ({w1}, {h1})")
            continue
        if x2 >= w2 or y2 >= h2:
            print(f"  WARNING: Image 2 point ({x2}, {y2}) outside bounds ({w2}, {h2})")
            continue

        cv2.circle(vis_img, (x1, y1), 5, color, -1)
        cv2.circle(vis_img, (x2 + w1, y2), 5, color, -1)
        cv2.line(vis_img, (x1, y1), (x2 + w1, y2), color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)

def get_all_matches(db_path):
    # All matches from the matches table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, data FROM matches")
    matches_data = cursor.fetchall()
    conn.close()
    extracted_matches = []
    for pair_id, data in matches_data:
        if data is None or len(data) == 0:
            continue
        try:
            matches = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
            extracted_matches.append((pair_id, matches))
        except Exception as e:
            print(f"Error unpacking matches for pair_id {pair_id}: {e}")
            continue
    return extracted_matches

def main():
    parser = argparse.ArgumentParser(description="Visualize COLMAP features.")
    parser.add_argument('--mode', choices=['features'], default='features',
                       help='Visualization mode: features')
    parser.add_argument('--object', type=str, required=True,
                       help='Object ID (e.g., ob_0000001)')
    args = parser.parse_args()
    db_path = f"data/object_sdf_data/{args.object}/database.db"
    image_dir = f"data/object_sdf_data/{args.object}/rgb_sampled"
    feature_vis_dir = f"data/object_sdf_data/{args.object}/feature_visualizations"

    id_to_name, name_to_id = get_image_name_id_map(db_path)

    # Debug print: images in database and directory
    print("Images in database:")
    for image_id, name in id_to_name.items():
        print(f"{image_id}: {name}")
    print("Images in directory:")
    print(sorted(os.listdir(image_dir)))

    if args.mode == 'features':
        clear_directory(feature_vis_dir)
        print(f"Cleared feature visualizations directory: {feature_vis_dir}")
        for image_id, image_name in id_to_name.items():
            keypoints = load_keypoints(image_name, db_path)
            if keypoints is None:
                print(f"No keypoints for {image_id} ({image_name})")
                continue
            image_path = os.path.join(image_dir, image_name)
            output_name = f"{Path(image_name).stem}_features.png"
            output_path = os.path.join(feature_vis_dir, output_name)
            visualize_features(image_path, keypoints, output_path)
            print(f"Processed {image_id}: {image_name}")

if __name__ == "__main__":
    main()
