"""
Module: metrics
Contains functions to compute quality metrics for reconstructions.
"""

import numpy as np

def sort_cameras_by_filename(reconstruction):
    """
    Sort images in the reconstruction by their filename and return their IDs.
    
    Parameters:
        reconstruction: A reconstruction object with an 'images' attribute.
        
    Returns:
        List of image IDs sorted by filename.
    """
    sorted_images = sorted(reconstruction.images.items(), key=lambda x: x[1].name)
    sorted_image_ids = [img_id for img_id, _ in sorted_images]
    return sorted_image_ids

def identify_peaks(displacements, image_names, percentile=80):
    """
    Identify peaks in displacement data based on a percentile threshold.
    
    Parameters:
        displacements (list): List of displacement values between consecutive frames.
        image_names (list): List of image names corresponding to the displacements.
        percentile (int, optional): Percentile threshold for peak detection.
        
    Returns:
        List of tuples representing image pairs where peaks occur.
    """
    threshold = np.percentile(displacements, percentile)
    peak_pairs = [(image_names[i], image_names[i+1]) for i in range(len(displacements)) if displacements[i] >= threshold]
    return peak_pairs

def compute_displacement_from_sorted(reconstruction):
    """
    Compute the Euclidean displacement between consecutive camera positions after sorting by filename.
    
    Parameters:
        reconstruction: A reconstruction object with camera positions.
        
    Returns:
        Tuple (displacements, peak_pairs) where displacements is a list of distances, 
        and peak_pairs is a list of image name pairs indicating significant changes.
    """
    sorted_image_ids = sort_cameras_by_filename(reconstruction)
    positions = [reconstruction.images[i].cam_from_world.translation for i in sorted_image_ids]
    displacements = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
    tmp_peak_pairs = identify_peaks(displacements, sorted_image_ids, percentile=80)
    return displacements, tmp_peak_pairs

def overlap_between_two_images(image1, image2):
    """
    Compute the overlap between two images based on shared observation indices.
    
    Parameters:
        image1, image2: Image objects with a method get_observation_point2D_idxs().
        
    Returns:
        Integer count of overlapping observation indices.
    """
    a = set(image1.get_observation_point2D_idxs())
    b = set(image2.get_observation_point2D_idxs())
    return len(a.intersection(b))

def compute_overlaps_in_rec(rec):
    """
    Compute overlaps for consecutive images in a reconstruction.
    
    Parameters:
        rec: A reconstruction object with images.
        
    Returns:
        Tuple (overl, peak_pairs) where 'overl' is a list of overlap counts and 
        'peak_pairs' contains image ID pairs that are below a threshold.
    """
    sorted_image_ids = sort_cameras_by_filename(rec)
    imgs = [rec.images[i] for i in sorted_image_ids]
    overl = [overlap_between_two_images(imgs[i], imgs[i+1])
             for i in range(len(imgs)-1)]
    threshold = np.percentile(overl, 40)
    peak_pairs = [
        (sorted_image_ids[i], sorted_image_ids[i+1])
        for i in range(len(overl))
        if overl[i] <= threshold
    ]
    return overl, peak_pairs


def is_reconstruction_good(rec, local_overlap_threshold=20, global_overlap_threshold=50, max_bad_pairs_ratio=0.0):
    """
    Evaluate the quality of a reconstruction based on overlap metrics between consecutive images.
    
    Local quality is defined by requiring that the overlap for each image pair is above
    a minimum threshold. Global quality is measured as the average overlap across all pairs.
    
    Parameters:
        rec: Reconstruction object with an 'images' attribute.
        local_overlap_threshold (int): Minimum acceptable overlap count for each consecutive image pair.
                                       This value should be determined empirically.
        global_overlap_threshold (int): Minimum acceptable average overlap across all consecutive pairs.
        max_bad_pairs_ratio (float): Maximum allowed ratio of image pairs below the local threshold.
                                     Set to 0.0 for a strict requirement.
    
    Returns:
        bool: True if the reconstruction meets both local and global quality criteria, False otherwise.
    """
    import numpy as np

    # Compute overlaps using the existing function in this module.
    overl, _ = compute_overlaps_in_rec(rec)
    
    if not overl:
        # Not enough pairs to judge the reconstruction quality.
        print("No overal images!")
        return False

    # Evaluate local quality: count image pairs below the local threshold.
    bad_pairs = [o for o in overl if o < local_overlap_threshold]
    print(f"There are {len(bad_pairs)} bad bad_pairs out of {len(overl)}. Threshold: {local_overlap_threshold}")

    ratio_bad = len(bad_pairs) / len(overl)
    local_good = (ratio_bad <= max_bad_pairs_ratio)
    
    print(f"Average overlap: {np.mean(overl)}; target: {global_overlap_threshold}")
    # Evaluate global quality: average overlap should be at least the global threshold.
    global_good = (np.mean(overl) >= global_overlap_threshold)
    
    return local_good and global_good
