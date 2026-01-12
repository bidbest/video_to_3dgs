"""
Module: colmap_pipeline
Contains functions for performing COLMAP-based 3D reconstruction and refinement.
"""

import os
import sys
import shutil
import copy
import numpy as np
import logging
import pycolmap
from typing import Dict, List, Tuple, Set, Optional

from utils import _name_to_ind, make_folders, clean_paths
from metrics import compute_overlaps_in_rec, sort_cameras_by_filename, overlap_between_two_images

def extract_features(db_path, image_path, image_list):
    """
    Run COLMAP feature extraction on the provided images.
    
    Parameters:
        db_path (str): Path to the COLMAP database.
        image_path (str): Path to the image directory.
        image_list (list): List of image filenames.
    """
    if len(image_list) > 1:
        base_path = os.path.dirname(db_path)
        img_list_path = os.path.join(base_path, "image_list.txt")
        with open(img_list_path, "w") as f:
            f.writelines('\n'.join(image_list))
        feat_extraction_cmd = (
            "colmap feature_extractor "
            "--database_path " + db_path +
            " --image_path " + image_path +
            " --ImageReader.single_camera 1 " +
            " --image_list_path " + img_list_path +
            " --ImageReader.camera_model OPENCV " +
            " --SiftExtraction.max_num_features 5000 " +
            " --SiftExtraction.use_gpu " + str(True)
        )
    else:
        feat_extraction_cmd = (
            "colmap feature_extractor "
            "--database_path " + db_path +
            " --image_path " + image_path +
            " --ImageReader.single_camera 1 " +
            " --ImageReader.camera_model OPENCV " +
            " --SiftExtraction.max_num_features 5000 " +
            " --SiftExtraction.use_gpu " + str(True)
        )

    exit_code = os.system(feat_extraction_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

def feature_matching(db_path, sequential):
    """
    Run COLMAP feature matching using either sequential or exhaustive matching.
    
    Parameters:
        db_path (str): Path to the COLMAP database.
        sequential (bool): Flag to choose sequential matching.
    """
    mode = "sequential_matcher" if sequential else "exhaustive_matcher"
    feat_matching_cmd = (
        "colmap " + mode +
        " --database_path " + db_path +
        " --SiftMatching.use_gpu " + str(True)
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

def count_total_images_in_multiple_reconstructions(reconstructions):
    tot_images = 0

    for rec_i in reconstructions.keys():
        tot_images += reconstructions[rec_i].num_images()
    
    return tot_images

def _from_img_name_to_img_num(img_name):
    tmp = img_name.split(".")[0]
    return int(tmp)

def get_images_names_from_reconstructions(reconstructions):

    images_names = []
    for rec_i in reconstructions.keys():
        tmp_rec = reconstructions[rec_i]
        tmp_images_names = sorted([_from_img_name_to_img_num(tmp_rec.images[i].name) for i in tmp_rec.images])

        images_names.append(tmp_images_names)
    
    return images_names

def compute_intervals_that_needs_more_images(rec_img_ids, all_img_ids, increase_ratio = 1.3):

    # Since batch can overlap, we first want to know which images are in reconstructions and which aren't.
    
    all_img_ids_is_rec = np.zeros(len(all_img_ids))
    for tmp in rec_img_ids:
        for ind_img in tmp:
            ind = all_img_ids.index(ind_img)
            all_img_ids_is_rec[ind] = 1

    # import pdb; pdb.set_trace()

    # now we look for images not registered, and we create ranges containing all non registered images
    range_non_rec = []
    n_images_for_range = []
    cnt = 0
    for i in range(len(all_img_ids)):

        val = all_img_ids_is_rec[i]

        if val == 0 and cnt == 0:
            if i > 0:
                start = all_img_ids[i-1]
            else:
                start = all_img_ids[i]
            cnt = 1
            continue
        
        elif val == 0 and cnt >= 1:
            # This means is a consecutive non registered, we count it
            cnt += 1

        elif val == 1 and cnt >= 1:
            # After encountering a non-registered image  we find a registered image. we save this range
            end = all_img_ids[i]
            range_non_rec.append((start, end))
            n_images_for_range.append(cnt+1)
            cnt = 0
    
    # import pdb; pdb.set_trace()
    n_frame_to_generate = np.asarray(n_images_for_range, dtype=np.float32) * increase_ratio
    n_frame_to_generate = np.ceil(n_frame_to_generate).astype(np.int32)
    # import pdb; pdb.set_trace()

    return range_non_rec, n_frame_to_generate
    

def reconstruct(source_path, db_path, image_path, output_path, image_list, existing_sparse='', clean=True, sequential=True, threshold=0.8):
    """
    Perform COLMAP incremental mapping to reconstruct a 3D scene.
    
    Parameters:
        source_path (str): Base directory.
        db_path (str): Path to the COLMAP database.
        image_path (str): Directory containing images.
        output_path (str): Directory to save the reconstruction.
        image_list (list): List of images to use.
        existing_sparse (str, optional): Path to an existing sparse reconstruction.
        clean (bool, optional): Flag to clean the output directory.
        sequential (bool, optional): Use sequential matching if True.
        threshold (float, optional): Minimum fraction of images to be registered.
    
    Returns:
        Reconstruction object if successful, else None.
    """
    extract_features(db_path, image_path, image_list)
    feature_matching(db_path, sequential)
    reconstructions = pycolmap.incremental_mapping(db_path, image_path, output_path, input_path=existing_sparse)

    if clean:
        rec_fold = os.listdir(output_path)
        [shutil.rmtree(os.path.join(output_path, tmp)) for tmp in rec_fold]

    if len(image_list) == 0:
        image_list = os.listdir(image_path)

    min_registered_images = len(image_list) * threshold
    rec = None
    for tmp_rec in reconstructions:
        if reconstructions[tmp_rec].num_images() >= min_registered_images:
            rec = reconstructions[tmp_rec]
            break

    return rec, image_list, reconstructions


def iterative_reconstruc(source_path, db_path, image_path, output_path, image_list, fmw, video_n, max_iter=15):
    """
    Iteratively attempt reconstruction by gradually increasing the number of frames.
    
    Parameters:
        source_path (str): Base directory.
        db_path (str): Path to the COLMAP database.
        image_path (str): Directory containing images.
        output_path (str): Directory for saving reconstruction.
        image_list (list): Initial list of images.
        fmw: FFmpegWrapper object.
        video_n (str): Video filename.
        max_iter (int, optional): Maximum iterations.
    
    Returns:
        Tuple (reconstruction, frames_list) from the successful iteration.
    """
    n_frames = int(fmw.duration / 2)
    print(f" testing reconstruction with {n_frames} frames")
    frames_list = fmw.get_list_of_n_frames(n_frames)
    rec, frames_list, reconstructions = reconstruct(source_path, db_path, image_path, output_path, frames_list)
    itr = 1

    # increase until we get at least 50% ratio of reconstructed images 

    while rec is None and itr < max_iter:
        clean_paths(source_path, video_n, db_path)
        make_folders(source_path)

        # Check percentage reconstructed
        n_rec_images = count_total_images_in_multiple_reconstructions(reconstructions)
        ratio_reconstructed = n_rec_images / len(frames_list)

        ##############################################
        ############# DEBUG ##########################
        rec_images = count_total_images_in_multiple_reconstructions(reconstructions)
        print(f"recunstructed {rec_images} / {len(frames_list)}\nRatio: {rec_images / len(frames_list)}")
        rec_img_names = get_images_names_from_reconstructions(reconstructions)
        all_img_names = [_from_img_name_to_img_num(os.path.basename(i)) for i in frames_list]
        # import pdb; pdb.set_trace()
        ##############################################
        
        if ratio_reconstructed < 0.4:
            # increase until we get at least 40% ratio of reconstructed images 
            n_frames = int(n_frames * 1.3)
            print(f" testing reconstruction with {n_frames} frames")
            frames_list = fmw.get_list_of_n_frames(n_frames)
            rec, frames_list, reconstructions = reconstruct(source_path, db_path, image_path, output_path, frames_list)

        else:
            # Only increase between batches reconstructed 
            rec_img_ids = get_images_names_from_reconstructions(reconstructions)
            # Get all the indexs we are trying to reconstruct 
            all_img_ids = [_from_img_name_to_img_num(os.path.basename(i)) for i in frames_list]

            (pairs_to_interp, n_frame_to_generate) = compute_intervals_that_needs_more_images(rec_img_ids, all_img_ids)
            for pair, n_frames in zip(pairs_to_interp, n_frame_to_generate):
                frames_list.extend(fmw.get_list_of_n_frames(n_frames, pair[0], pair[1]))
            
            frames_list = list(set(frames_list))
            frames_list.sort()
            # import pdb; pdb.set_trace()
            rec, frames_list, reconstructions = reconstruct(source_path, db_path, image_path, output_path, frames_list)

        itr += 1
    
    if itr >= max_iter:
        print("can't perform basic reconstruction!")

    return rec, frames_list

def get_overlap_gap_frames(rec, fmw, min_overlap, new_image_budget):
    """
    Use consecutive overlap gaps (existing logic) to propose candidate frames.
    
    Parameters:
        rec (pycolmap.Reconstruction): Current reconstruction.
        fmw (FFmpegWrapper): Instance providing frame filenames (and ind_to_frame_name).
        min_overlap (int): Minimum required overlap threshold.
        new_image_budget (int): Total number of new images to allocate.
        
    Returns:
        List[str]: List of candidate frame file paths (full paths).
    """
    # Get sorted image IDs from the reconstruction.
    sorted_ids = sort_cameras_by_filename(rec)
    imgs = [rec.images[i] for i in sorted_ids]

    pair_overlaps = []
    for i in range(len(imgs) - 1):
        ov = overlap_between_two_images(imgs[i], imgs[i+1])
        if ov is None:
            ov = 0
        pair_overlaps.append(ov)

    candidate_pairs = []
    deficiencies = []
    for i, ov in enumerate(pair_overlaps):
        if ov < min_overlap:
            img_name1 = imgs[i].name
            img_name2 = imgs[i+1].name
            idx1 = _name_to_ind(img_name1)
            idx2 = _name_to_ind(img_name2)
            if idx2 - idx1 > 1:
                deficiency = min_overlap - ov
                deficiencies.append(deficiency)
                candidate_pairs.append((idx1, idx2))
                
    total_deficiency = sum(deficiencies)
    new_images_set = set()
    for (pair, deficiency) in zip(candidate_pairs, deficiencies):
        (i1, i2) = pair
        if total_deficiency > 0:
            pair_budget = int(round((deficiency / total_deficiency) * new_image_budget))
        else:
            pair_budget = 0
            
        available_slots = i2 - i1 - 1
        pair_budget = min(pair_budget, available_slots)
        
        if pair_budget > 0:
            indices = np.linspace(i1 + 1, i2 - 1, pair_budget)
            indices = [int(round(x)) for x in indices]
            for ind in indices:
                new_images_set.add(fmw.ind_to_frame_name(ind))
    
    new_images_list = sorted(new_images_set)
    return new_images_list


def get_uniform_frames(fmw, current_image_list, uniform_budget):
    """
    Uniformly select candidate frames from the full set of frames that are not
    already in the current list.
    
    Parameters:
        fmw (FFmpegWrapper): Provides fmw.frames (list of frame filenames) and ind_to_frame_name.
        current_image_list (list): Current list of full frame paths.
        uniform_budget (int): Number of new uniform frames to select.
        
    Returns:
        List[str]: List of candidate frame file paths (full paths).
    """
    # fmw.frames is assumed to be a sorted list of base filenames (e.g., "00001234.jpeg")
    all_frames = fmw.frames  
    # Extract base names from currently selected images.
    current_basenames = {os.path.basename(x) for x in current_image_list}
    # Frames that haven't yet been selected:
    available = [f for f in all_frames if f not in current_basenames]
    
    if not available:
        return []
    
    if len(available) <= uniform_budget:
        selected = available
    else:
        step = len(available) / uniform_budget
        indices = [int(round(i * step)) for i in range(uniform_budget)]
        # Ensure indices are within bounds:
        selected = [available[i] for i in indices if i < len(available)]
    
    # Convert base filenames to full paths
    selected_paths = [os.path.join(fmw.tmp_path, name) for name in selected]
    return selected_paths


def compute_new_frame_candidates(rec, current_image_list, fmw, min_overlap, new_image_budget, uniform_budget=10):
    """
    Modular strategy: combine multiple selection strategies to propose new candidate frames.
    
    Parameters:
        rec (pycolmap.Reconstruction): The current sparse model reconstruction.
        current_image_list (list): Current list of image file paths already used.
        fmw (FFmpegWrapper): Provides frame extraction utility.
        min_overlap (int): Minimum allowed overlap for the overlap-gap method.
        new_image_budget (int): Budget to allocate for overlap-gap strategy.
        uniform_budget (int): Budget to allocate for uniform frame selection.
        
    Returns:
        List[str]: Sorted list of new candidate frame paths.
    """
    candidates_overlap = get_overlap_gap_frames(rec, fmw, min_overlap, new_image_budget)
    candidates_uniform = get_uniform_frames(fmw, current_image_list, uniform_budget)
    
    # Placeholder for future strategies (e.g., density based, low coverage region).
    candidates_low_density = []  # To be implemented later.
    
    # Merge all candidates, remove duplicates, and sort.
    new_images_set = set(candidates_overlap + candidates_uniform + candidates_low_density)
    new_images_list = sorted(new_images_set)
    return new_images_list

def incremental_reconstruction(source_path, db_path, image_path, output_path, image_list, fmw, rec, n_images,
                               quality_threshold_avg=100,  
                               min_overlap=25, new_image_budget=50, uniform_budget=10):
    """
    Incrementally improves the reconstruction by adding new images using a combination of
    selection strategies, with an early stopping mechanism based on quality metrics.
    
    Parameters:
      source_path             : Base directory for the reconstruction process.
      db_path, image_path, output_path : Paths used in the reconstruction.
      image_list              : Current list of image paths.
      fmw                     : FFmpegWrapper instance.
      rec                     : Current reconstruction.
      n_images                : Target number of registered images.
      quality_threshold_avg   : Desired minimum average overlap.
      min_overlap             : Minimum allowed overlap between consecutive images.
      new_image_budget        : Maximum new images to add for overlap-gap strategy.
      uniform_budget          : Maximum new images to add for uniform strategy.
    
    Returns:
      rec2                    : The updated reconstruction.
      new_images              : Candidate images from the last iteration.
    """
    # Use the new candidate frame strategy manager.
    new_images = compute_new_frame_candidates(rec, image_list, fmw, min_overlap, new_image_budget, uniform_budget)
    rec2 = rec
    rec2.write_binary(output_path)
    
    while len(rec2.images) < n_images and len(new_images) > 0:
        # Update candidate selection on each iteration.
        new_images = compute_new_frame_candidates(rec2, image_list, fmw, min_overlap, new_image_budget, uniform_budget)
        print(f"Adding {len(new_images)} new images based on combined strategies")
        image_list.extend(new_images)
        # Remove duplicates while preserving order.
        image_list = list(dict.fromkeys(image_list))
        
        tmp, _, _ = reconstruct(source_path, db_path, image_path, output_path, image_list, output_path, clean=False)
        if tmp is None:
            print(len(image_list))
            print("New reconstruction failed ...")
            break

        rec2 = tmp
        rec2.write_binary(output_path)
        
        # Evaluate quality by computing overlap metrics.
        sorted_ids = sort_cameras_by_filename(rec2)
        imgs = [rec2.images[i] for i in sorted_ids]
        overlaps = [overlap_between_two_images(imgs[i], imgs[i+1]) for i in range(len(imgs)-1)]
        
        if overlaps:
            avg_overlap = np.mean(overlaps)
            min_ovl = np.min(overlaps)
        else:
            avg_overlap = 0
            min_ovl = 0
        
        print(f"Current average overlap: {avg_overlap}, current minimum overlap: {min_ovl}")
        
        # Early stopping: if quality metrics are met, break.
        if (quality_threshold_avg is not None and avg_overlap >= quality_threshold_avg) \
           and (min_overlap is not None and min_ovl >= min_overlap):
            print(f"Early stopping triggered. Average overlap {avg_overlap} exceeds threshold "
                  f"{quality_threshold_avg} and minimum overlap {min_ovl} exceeds threshold {min_overlap}.")
            break

    return rec2, new_images



def filter_rec(rec_orig, img_path):
    """
    Filter the reconstruction to remove points and images with poor quality.
    
    Parameters:
        rec_orig: Original reconstruction object.
        img_path (str): Path to the images used in reconstruction.
    
    Returns:
        Filtered reconstruction object.
    """
    rec = copy.deepcopy(rec_orig)
    pcd = rec.points3D
    ids = np.array(list(rec.point3D_ids()))

    pcd_o3d = np.array([pcd[p].xyz for p in pcd])
    # The conversion to an Open3D PointCloud is omitted here;
    # it can be performed externally if visualization is needed.
    pcd_col = np.array([pcd[p].color for p in pcd])
    n_views = np.array([pcd[p].track.length() for p in pcd])
    rep_error = np.array([pcd[p].error for p in pcd])

    thr_views = np.percentile(n_views, 5)
    thr_error = np.percentile(rep_error, 95)

    # Remove 3D points that do not meet quality criteria.
    to_remove = np.where((((n_views > thr_views) * (rep_error < thr_error)) * 1) == 0)[0]
    to_remove_ids = ids[to_remove]
    [rec.delete_point3D(i) for i in to_remove_ids]

    imgs = [rec.images[i] for i in rec.images]
    ids = np.array([i for i in rec.images])
    n_points2d = np.asarray([i.num_points2D() for i in imgs])
    n_points3d = np.asarray([i.num_points3D for i in imgs])
    ratio_2d3d = n_points3d / n_points2d

    thr_2dviews = np.percentile(n_points2d, 10)
    thr_ratio = np.percentile(ratio_2d3d, 10)

    # Remove images that do not meet the 2D and 3D point criteria.
    to_remove = np.where(((n_points2d > thr_2dviews) * (ratio_2d3d > thr_ratio)) == 0)[0]
    to_remove_ids = ids[to_remove]
    [rec.deregister_image(i) for i in to_remove_ids]
    for i in to_remove_ids:
        p_to_rem = rec.images[i].get_observation_points2D()
        [rec.delete_point3D(i.point3D_id) for i in p_to_rem]
        os.remove(os.path.join(img_path, rec.images[i].name))
        del rec.images[i]

    print(f"Original images: {len(rec_orig.images)} -> Filtered images: {len(rec.images)}")
    print(f"Original 3D points: {len(rec_orig.points3D)} -> Filtered 3D points: {len(rec.points3D)}")
    return rec

def interpolate_all_frames(rec, fmw):
    """
    Interpolate camera poses for all frames in the video.
    
    Parameters:
        rec: COLMAP reconstruction object
        fmw: FFmpegWrapper instance
        
    Returns:
        Dictionary mapping frame indices to (camera_pose, confidence)
    """
    from pose_interpolation import interpolate_camera_poses, validate_interpolation
    
    # Validate interpolation accuracy
    validation_metrics = validate_interpolation(rec, fmw)
    print("Interpolation validation metrics:")
    for key, value in validation_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Interpolate poses for all frames
    all_poses = interpolate_camera_poses(rec, fmw)
    print(f"Interpolated poses for {len(all_poses)} frames")
    
    return all_poses

def do_one(source_path, n_images, clean=False, minimal=False, full=False, full_res=False, average_overlap=100):
    """
    Main pipeline function to process a video, perform reconstruction,
    and generate undistorted outputs.
    
    Parameters:
        source_path (str): Base directory containing video and images.
        n_images (int): Target number of images for reconstruction.
        clean (bool, optional): Flag to clean existing paths before processing.
        minimal (bool, optional): Use minimal frame selection after final reconstruction.
        full (bool, optional): Use all frame selection after final reconstruction.
        full_res (bool, optional): Extract final frames at full resolution.
        average_overlap (int, optional): Target average overlap between frames.
    """
    files_n = os.listdir(source_path)
    video_n = None
    for f in files_n:
        if f.lower().endswith(('.mp4', '.mov', '.avi')):
            video_n = f
            break

    if video_n is None and (not ("input" in files_n)):
        exit(1)

    video_p = os.path.join(source_path, video_n)
    input_p = os.path.join(source_path, 'input')
    distorted_path = os.path.join(source_path, "distorted")
    distorted_sparse_path = os.path.join(distorted_path, "sparse")
    distorted_sparse_final_path = os.path.join(distorted_path, "sparse_final")
    sparse_path = os.path.join(source_path, "sparse/0")
    db_path = os.path.join(distorted_path, "database.db")
    if clean:
        clean_paths(source_path, video_n, db_path)
    make_folders(source_path)

    from video_processing import FFmpegWrapper
    fmw = FFmpegWrapper(video_p, input_p)

    n_frames = int(fmw.duration)
    frames_list = fmw.get_list_of_n_frames(n_frames)

    if os.path.isfile(os.path.join(distorted_path, "orig_distorted", "images.bin")):
        print("Loading original reconstruction")
        rec = pycolmap.Reconstruction(os.path.join(distorted_path, "orig_distorted"))
        frames_list = [os.path.join(fmw.tmp_path, rec.images[i].name) for i in rec.images]
    else:
        rec, frames_list = iterative_reconstruc(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, video_n)
        rec.write_binary(distorted_sparse_path)
        shutil.copytree(distorted_sparse_path, os.path.join(os.path.dirname(distorted_sparse_path), "orig_distorted"))
    
    print(rec.summary())

    if os.path.isfile(os.path.join(distorted_path, "sparse/0/", "images.bin")):
        print("Loading dense reconstruction")
        rec2 = pycolmap.Reconstruction(os.path.join(distorted_path, "sparse/0/"))
        frames_list = [os.path.join(fmw.tmp_path, rec2.images[i].name) for i in rec2.images]
    else:
        rec2, frames_list = incremental_reconstruction(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, rec, n_images, quality_threshold_avg=average_overlap)
                            
    print(rec2.summary())
    db_fin_path = os.path.join(distorted_path, "database_final.db")
    distorted_sparse_0_path = os.path.join(distorted_sparse_path, "0")
    os.makedirs(distorted_sparse_0_path, exist_ok=True)
    if os.path.isfile(os.path.join(sparse_path, "images.bin")):
        print("Loading final reconstruction")
        final = pycolmap.Reconstruction(sparse_path)
        frames_list = [os.path.join(input_p, final.images[i].name) for i in final.images]
    else:
        if minimal:
            frame_indices = select_minimal_image_subset(rec2, overlap_threshold=average_overlap)

        elif not full: # if full we keep all frames
            frame_indices = select_filtered_image_subset(rec2, max_num_images=n_images)

        else:
            # Keep all images
            sorted_ids = sort_cameras_by_filename(rec2)
            frame_indices = sorted([_name_to_ind(rec2.images[i].name) for i in sorted_ids])

        fmw.extract_specific_frames(frame_indices, full_res=full_res)
        final, _, _ = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, sequential=False, image_list=[])
        
        if final is None:
            print(" Cannot reconstruct with full matcher. Using sequential")
            final, _, _ = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, sequential=True, image_list=[])
            # sys.exit(1)

        final.write_binary(distorted_sparse_0_path)
    
    print(final.summary())

    if not minimal:
        from metrics import compute_overlaps_in_rec
        overl, _ = compute_overlaps_in_rec(rec)
        overl2, _ = compute_overlaps_in_rec(rec2)
        overlf, _ = compute_overlaps_in_rec(final)
        print(f"Original SFM: min_overlap: {np.min(overl)}; average_overl: {np.mean(overl)}; reconstruction summary: {rec.summary()}\n\n")
        print(f"Incremental SFM: min_overlap: {np.min(overl2)}; average_overl: {np.mean(overl2)}; reconstruction summary: {rec2.summary()}\n\n")
        print(f"Final SFM: min_overlap: {np.min(overlf)}; average_overl: {np.mean(overlf)}; reconstruction summary: {final.summary()}\n\n")

        
        if not full:
            print("filtering reconstruction....")
            final_filtered = filter_rec(final, input_p)
        else:
            final_filtered = final

        print(final_filtered.summary())
        shutil.rmtree(distorted_sparse_0_path)
        os.makedirs(distorted_sparse_0_path, exist_ok=True)
        final_filtered.write_binary(distorted_sparse_0_path)


    img_undist_cmd = (
        "colmap image_undistorter "
        " --image_path " + input_p +
        " --input_path " + distorted_sparse_0_path +
        " --output_path " + source_path +
        " --output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def do_one_robust(source_path, n_images, clean=False, minimal=False, full=False, full_res=False, average_overlap=100,
                 random_ratio=0.2, pruning_threshold=0.05, coverage_weight=0.4, triangulation_weight=0.3,
                 diversity_weight=0.2, confidence_weight=0.1):
    """
    Enhanced pipeline function to process a video, perform reconstruction with robust frame selection,
    and generate undistorted outputs. This version uses camera pose interpolation to make better
    decisions about which frames to include.
    
    Parameters:
        source_path (str): Base directory containing video and images.
        n_images (int): Target number of images for reconstruction.
        clean (bool, optional): Flag to clean existing paths before processing.
        minimal (bool, optional): Use minimal frame selection after final reconstruction.
        full (bool, optional): Use all frame selection after final reconstruction.
        full_res (bool, optional): Extract final frames at full resolution.
        average_overlap (int, optional): Target average overlap between frames.
        random_ratio (float, optional): Ratio of frames to select randomly.
        pruning_threshold (float, optional): Threshold for pruning low-contribution frames.
        coverage_weight (float, optional): Weight for coverage score in frame selection.
        triangulation_weight (float, optional): Weight for triangulation score in frame selection.
        diversity_weight (float, optional): Weight for diversity score in frame selection.
        confidence_weight (float, optional): Weight for confidence score in frame selection.
    """
    files_n = os.listdir(source_path)
    video_n = None
    for f in files_n:
        if f.lower().endswith(('.mp4', '.mov', '.avi')):
            video_n = f
            break

    if video_n is None and (not ("input" in files_n)):
        exit(1)

    video_p = os.path.join(source_path, video_n)
    input_p = os.path.join(source_path, 'input')
    distorted_path = os.path.join(source_path, "distorted")
    distorted_sparse_path = os.path.join(distorted_path, "sparse")
    distorted_sparse_final_path = os.path.join(distorted_path, "sparse_final")
    sparse_path = os.path.join(source_path, "sparse/0")
    db_path = os.path.join(distorted_path, "database.db")
    
    if clean:
        clean_paths(source_path, video_n, db_path)
    make_folders(source_path)

    from video_processing import FFmpegWrapper
    fmw = FFmpegWrapper(video_p, input_p)

    n_frames = int(fmw.duration)
    frames_list = fmw.get_list_of_n_frames(n_frames)

    # Step 1: Get initial reconstruction (same as in do_one)
    if os.path.isfile(os.path.join(distorted_path, "orig_distorted", "images.bin")):
        print("Loading original reconstruction")
        rec = pycolmap.Reconstruction(os.path.join(distorted_path, "orig_distorted"))
        frames_list = [os.path.join(fmw.tmp_path, rec.images[i].name) for i in rec.images]
    else:
        rec, frames_list = iterative_reconstruc(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, video_n)
        rec.write_binary(distorted_sparse_path)
        shutil.copytree(distorted_sparse_path, os.path.join(os.path.dirname(distorted_sparse_path), "orig_distorted"))
    
    print(rec.summary())

    # Step 2: Interpolate camera poses for all frames
    print("Interpolating camera poses for all frames...")
    all_poses = interpolate_all_frames(rec, fmw)
    
    # Step 3: Use the interpolated poses to guide the incremental reconstruction
    if os.path.isfile(os.path.join(distorted_path, "sparse/0/", "images.bin")):
        print("Loading dense reconstruction")
        rec2 = pycolmap.Reconstruction(os.path.join(distorted_path, "sparse/0/"))
        frames_list = [os.path.join(fmw.tmp_path, rec2.images[i].name) for i in rec2.images]
    else:
        # Use advanced frame selection strategy
        from visibility_prediction import compute_visibility_sampled
        from frame_selection import adaptive_frame_selection

        n_new_frames = n_images - len(frames_list)

        # Only perform if more frame needed.
        if n_new_frames > 0:
            print("Computing visibility for all frames...")
            camera_id = next(iter(rec.cameras))
            camera = rec.cameras[camera_id]
            all_visibility = compute_visibility_sampled(rec, all_poses, camera)

            print("Selecting frames using adaptive strategy...")
            selected_frames = adaptive_frame_selection(
                rec,
                all_poses,
                all_visibility,
                n_frames=n_new_frames,
                random_ratio=random_ratio,
                coverage_weight=coverage_weight,
                triangulation_weight=triangulation_weight,
                diversity_weight=diversity_weight,
                confidence_weight=confidence_weight
            )
            print(f"Selected {len(selected_frames)} frames for reconstruction")
            
            # Convert frame indices to file paths
            selected_frames_paths = [fmw.ind_to_frame_name(idx) for idx in selected_frames]
            
            # Add selected frames to the existing frames list
            frames_list.extend(selected_frames_paths)
            
            # Remove duplicates while preserving order
            frames_list = list(dict.fromkeys(frames_list))
            
            # Perform reconstruction with the selected frames
            print("Performing reconstruction with selected frames...")
            rec2, _, _ = reconstruct(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, distorted_sparse_path, clean=False)
            
        else:
            rec2 = rec
        
        if rec2 is None:
            print("Enhanced frame selection failed, falling back to incremental reconstruction...")
            rec2, frames_list = incremental_reconstruction(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, rec, n_images, quality_threshold_avg=average_overlap)
        else:
            # Prune low-contribution frames using advanced pruning
            print("Pruning low-contribution frames using advanced analysis...")
            from frame_selection import prune_frames
            rec2, removed_frames = prune_frames(rec2, threshold=pruning_threshold)
            rec2.write_binary(distorted_sparse_path)
            
            # Update frames list after pruning
            frames_list = [os.path.join(fmw.tmp_path, rec2.images[i].name) for i in rec2.images]
            
            print(f"Removed {len(removed_frames)} frames, keeping {len(frames_list)} frames")

    print(rec2.summary())
    
    # Step 4: Final reconstruction (same as in do_one)
    db_fin_path = os.path.join(distorted_path, "database_final.db")
    distorted_sparse_0_path = os.path.join(distorted_sparse_path, "0")
    os.makedirs(distorted_sparse_0_path, exist_ok=True)
    
    if os.path.isfile(os.path.join(sparse_path, "images.bin")):
        print("Loading final reconstruction")
        final = pycolmap.Reconstruction(sparse_path)
        frames_list = [os.path.join(input_p, final.images[i].name) for i in final.images]
    else:

        sorted_ids = sort_cameras_by_filename(rec2)
        frame_indices = sorted([_name_to_ind(rec2.images[i].name) for i in sorted_ids])

        fmw.extract_specific_frames(frame_indices, full_res=full_res)
        final, _, _ = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, sequential=False, image_list=[])
        
        if final is None:
            print("Cannot reconstruct with full matcher. Using sequential")
            final, _, _ = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, sequential=True, image_list=[])

        final.write_binary(distorted_sparse_0_path)
    
    print(final.summary())

    if not minimal:
        from metrics import compute_overlaps_in_rec
        overl, _ = compute_overlaps_in_rec(rec)
        overl2, _ = compute_overlaps_in_rec(rec2)
        overlf, _ = compute_overlaps_in_rec(final)
        print(f"Original SFM: min_overlap: {np.min(overl)}; average_overl: {np.mean(overl)}; reconstruction summary: {rec.summary()}\n\n")
        print(f"Incremental SFM: min_overlap: {np.min(overl2)}; average_overl: {np.mean(overl2)}; reconstruction summary: {rec2.summary()}\n\n")
        print(f"Final SFM: min_overlap: {np.min(overlf)}; average_overl: {np.mean(overlf)}; reconstruction summary: {final.summary()}\n\n")
        final_filtered = final

        print(final_filtered.summary())
        shutil.rmtree(distorted_sparse_0_path)
        os.makedirs(distorted_sparse_0_path, exist_ok=True)
        final_filtered.write_binary(distorted_sparse_0_path)

    # Step 5: Image undistortion (same as in do_one)
    img_undist_cmd = (
        "colmap image_undistorter "
        " --image_path " + input_p +
        " --input_path " + distorted_sparse_0_path +
        " --output_path " + source_path +
        " --output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def select_minimal_image_subset(rec, overlap_threshold=50, max_gap=30):
    """
    Select a minimal subset of image indices while preserving global coverage.

    This version looks at consecutive images (sorted by filename) and 
    adds an image if the overlap with the previous kept image is below a threshold.
    Additionally, if the frame index difference (gap) between two images exceeds
    'max_gap', it will add an image even if the overlap is high. This helps maintain 
    global coverage in minimal mode.

    Parameters:
        rec: pycolmap.Reconstruction object.
        overlap_threshold (int): Minimum required overlap between anchor and candidate.
        max_gap (int): Maximum allowable gap (in frame index difference) before adding an image.
        
    Returns:
        List[int]: Sorted list of selected frame indices (as integers from filename).
    """
    from metrics import sort_cameras_by_filename, overlap_between_two_images
    from utils import _name_to_ind

    sorted_ids = sort_cameras_by_filename(rec)
    imgs = rec.images

    if not sorted_ids:
        return []

    # Start with the first image as an anchor.
    kept_ids = [sorted_ids[0]]

    for current in sorted_ids[1:]:
        last_kept = kept_ids[-1]

        # Compute overlap between the last kept image and current candidate.
        ov = overlap_between_two_images(imgs[last_kept], imgs[current])
        # Get frame indices from filenames (assumes name like "00001234.jpeg")
        idx_last = _name_to_ind(imgs[last_kept].name)
        idx_current = _name_to_ind(imgs[current].name)
        gap = idx_current - idx_last

        # Enforce a maximum gap: if gap > max_gap, add the image to preserve coverage.
        if gap > max_gap:
            kept_ids.append(current)
        else:
            # Otherwise, if overlap is below threshold (indicating a significant viewpoint change), add it.
            if ov < overlap_threshold:
                kept_ids.append(current)
            # If overlap is high and gap is small, skip the candidate.
    
    # Convert the kept image names to indices.
    selected_indices = sorted([_name_to_ind(imgs[i].name) for i in kept_ids])
    print(f"[Minimal Subset] Selected {len(selected_indices)} frames out of {len(sorted_ids)}")
    return selected_indices


def select_filtered_image_subset(rec,
                                  max_num_images=None,
                                  keep_unique_point_ratio=0.1,
                                  min_2d3d_ratio_percentile=10,
                                  min_triangulated_percentile=10,
                                  min_geometric_distance=0.1):
    """
    Select a filtered subset of images based on contribution to the 3D reconstruction.

    Parameters:
        rec (pycolmap.Reconstruction): The original reconstruction.
        max_num_images (int): Optional cap on number of images to retain.
        keep_unique_point_ratio (float): Min ratio of unique 3D points to retain an image.
        min_2d3d_ratio_percentile (float): Filter out images below this percentile of 2D–3D match ratio.
        min_triangulated_percentile (float): Filter out images below this percentile of triangulated 3D points.
        min_geometric_distance (float): Min Euclidean distance to retain geometric diversity.

    Returns:
        List[int]: Sorted list of selected frame indices (int).
    """
    images = rec.images
    points = rec.points3D
    image_ids = sort_cameras_by_filename(rec)

    img_stats = []  # List of dicts with info for each image

    # Build point -> image_id map
    point_to_images = {}
    for pid, pt in points.items():
        obs = [el.image_id for el in pt.track.elements]
        for i in obs:
            point_to_images.setdefault(i, []).append(pid)

    # Count how many images see each point
    point_obs_count = {}
    for pid, pt in points.items():
        point_obs_count[pid] = len(pt.track.elements)

    # Compute per-image stats
    for img_id in image_ids:
        img = images[img_id]
        n2d = img.num_points2D()
        n3d = img.num_points3D
        ratio = n3d / max(n2d, 1e-8)

        seen_points = point_to_images.get(img_id, [])
        unique_pts = [pid for pid in seen_points if point_obs_count[pid] == 1]
        unique_ratio = len(unique_pts) / max(len(seen_points), 1e-8)

        cam_pos = img.cam_from_world.translation
        img_stats.append({
            "id": img_id,
            "name": img.name,
            "index": _name_to_ind(img.name),
            "num_points2D": n2d,
            "num_points3D": n3d,
            "2d3d_ratio": ratio,
            "unique_ratio": unique_ratio,
            "cam_pos": cam_pos,
        })

    # Convert to arrays for filtering
    ratios = np.array([s["2d3d_ratio"] for s in img_stats])
    n3ds = np.array([s["num_points3D"] for s in img_stats])
    uniqs = np.array([s["unique_ratio"] for s in img_stats])

    thr_ratio = np.percentile(ratios, min_2d3d_ratio_percentile)
    thr_n3d = np.percentile(n3ds, min_triangulated_percentile)

    # Initial keep mask
    keep = []
    for s in img_stats:
        if s["2d3d_ratio"] >= thr_ratio or s["num_points3D"] >= thr_n3d:
            keep.append(True)
        elif s["unique_ratio"] >= keep_unique_point_ratio:
            keep.append(True)
        else:
            keep.append(False)

    filtered = [s for s, k in zip(img_stats, keep) if k]

    # Enforce spatial/geometric diversity
    selected = []
    selected_positions = []

    for s in sorted(filtered, key=lambda x: -x["num_points3D"]):  # greedy: high contributor first
        pos = s["cam_pos"]
        if all(np.linalg.norm(pos - p) > min_geometric_distance for p in selected_positions):
            selected.append(s)
            selected_positions.append(pos)

    # Optional cap on total number
    if max_num_images and len(selected) > max_num_images:
        selected = selected[:max_num_images]

    selected_indices = sorted([s["index"] for s in selected])
    print(f"[Filtered Subset] Kept {len(selected_indices)} out of {len(image_ids)} frames.")
    return selected_indices

