# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml
import time



def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0'):
  try:
    print(f"Starting worker for object {ob_id} on device {device}")
    start_time = time.time()
    
    torch.cuda.set_device(device)
    est.to_device(device)
    print("Model moved to device")
    
    est.glctx = dr.RasterizeCudaContext(device=device)
    print("RasterizeCudaContext created")

    result = NestDict()
    
    # Pre-load all frames in batch
    print(f"Pre-loading {len(i_frames)} frames...")
    batch_data = []
    for i_frame in i_frames:
        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        id_str = reader.id_strs[i_frame]
        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
        if ob_mask is None:
            print(f"Mask not found for frame {i_frame}, skipping")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue
        batch_data.append((video_id, id_str, color, depth, ob_mask, i_frame))
    
    print(f"Processing {len(batch_data)} valid frames...")
    
    # Process frames in smaller sub-batches to avoid memory issues
    sub_batch_size = 8  # Process 8 frames at a time
    for sub_batch_start in range(0, len(batch_data), sub_batch_size):
        sub_batch_end = min(sub_batch_start + sub_batch_size, len(batch_data))
        sub_batch = batch_data[sub_batch_start:sub_batch_end]
        
        print(f"\nProcessing sub-batch {sub_batch_start//sub_batch_size + 1}/{(len(batch_data) + sub_batch_size - 1)//sub_batch_size}")
        sub_batch_start_time = time.time()
        
        # Pre-compute poses for all frames in sub-batch
        poses = []
        for video_id, id_str, color, depth, ob_mask, i_frame in sub_batch:
            est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
            try:
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
                poses.append((video_id, id_str, pose))
                print(f"Pose computed for frame {i_frame} in {time.time() - sub_batch_start_time:.2f}s")
            except Exception as e:
                print(f"Error in register for frame {i_frame}: {str(e)}")
                poses.append((video_id, id_str, np.eye(4)))
                continue

        # Store results
        for video_id, id_str, pose in poses:
            result[video_id][id_str][ob_id] = pose
            
        sub_batch_time = time.time() - sub_batch_start_time
        print(f"Sub-batch completed in {sub_batch_time:.2f}s (avg {sub_batch_time/len(sub_batch):.2f}s per frame)")

    total_time = time.time() - start_time
    print(f"Worker completed {len(batch_data)} frames in {total_time:.2f}s (avg {total_time/len(batch_data):.2f}s per frame)")
    return result
  except Exception as e:
    print(f"Error in worker: {str(e)}")
    raise e


def run_pose_estimation():
  try:
    print("Starting pose estimation...")
    start_time = time.time()
    
    wp.force_load(device='cuda')
    print("CUDA device loaded")
    
    torch.cuda.synchronize()
    print("CUDA synchronized")
    
    print("Initializing LinemodReader...")
    reader_tmp = LinemodReader(f'{opt.linemod_dir}/lm_test_all/test/000002', split=None)
    print("Reader initialized")

    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir
    print(f"Debug mode: {debug}, Using reconstructed mesh: {use_reconstructed_mesh}")

    res = NestDict()
    print("Creating RasterizeCudaContext...")
    glctx = dr.RasterizeCudaContext()
    print("RasterizeCudaContext created")
    
    print("Creating temporary mesh...")
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    print("Initializing FoundationPose...")
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)
    print("FoundationPose initialized")

    # Get number of CPU cores for parallel processing
    num_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers
    print(f"Using {num_workers} workers for parallel processing")

    total_objects = len(reader_tmp.ob_ids)
    print(f"Processing {total_objects} objects...")
    
    for obj_idx, ob_id in enumerate(reader_tmp.ob_ids):
        obj_start_time = time.time()
        print(f"\nProcessing object {obj_idx+1}/{total_objects} (ID: {ob_id})")
        ob_id = int(ob_id)
        
        try:
            if use_reconstructed_mesh:
                print(f"Loading reconstructed mesh for object {ob_id}...")
                mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
            else:
                print(f"Loading ground truth mesh for object {ob_id}...")
                mesh = reader_tmp.get_gt_mesh(ob_id)
            print(f"Mesh loaded for object {ob_id}")
        except Exception as e:
            print(f"Error loading mesh for object {ob_id}: {str(e)}")
            continue
            
        try:
            symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
            print(f"Loaded symmetry transforms for object {ob_id}")

            video_dir = f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'
            print(f"Loading video directory: {video_dir}")
            reader = LinemodReader(video_dir, split=None)
            video_id = reader.get_video_id()
            print(f"Video ID: {video_id}")
            
            print(f"Resetting object {ob_id}...")
            est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
            print(f"Object {ob_id} reset")

            # Process frames in larger batches
            batch_size = 32  # Process 32 frames at a time
            total_frames = len(reader.color_files)
            print(f"Processing {total_frames} frames in batches of {batch_size}...")

            for batch_start in range(0, total_frames, batch_size):
                batch_start_time = time.time()
                batch_end = min(batch_start + batch_size, total_frames)
                print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size} (frames {batch_start} to {batch_end-1})...")
                
                # Create batch of frame indices
                frame_indices = list(range(batch_start, batch_end))
                
                # Process batch
                try:
                    out = run_pose_estimation_worker(reader, frame_indices, est, debug, ob_id, "cuda:0")
                    for video_id in out:
                        for id_str in out[video_id]:
                            for ob_id in out[video_id][id_str]:
                                res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]
                    batch_time = time.time() - batch_start_time
                    print(f"Batch completed in {batch_time:.2f}s (avg {batch_time/len(frame_indices):.2f}s per frame)")
                    
                    # Clear CUDA cache after each batch
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue

            obj_time = time.time() - obj_start_time
            print(f"\nObject {ob_id} completed in {obj_time:.2f}s")
            print(f"Average time per frame: {obj_time/total_frames:.2f}s")

        except Exception as e:
            print(f"Error processing object {ob_id}: {str(e)}")
            continue

    total_time = time.time() - start_time
    print(f"\nAll objects processed in {total_time:.2f}s")
    
    print("Saving results...")
    with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
    print("Results saved")
    
  except Exception as e:
    print(f"Error in run_pose_estimation: {str(e)}")
    raise e


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD", help="linemod root dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()