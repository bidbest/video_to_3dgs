"""
Module: main_video_process
Entry point for running the COLMAP reconstruction pipeline.
"""

import argparse
from colmap_pipeline import do_one, do_one_robust

def main(args):
    """
    Main function to parse arguments and run the pipeline.
    
    Parameters:
        args: Parsed command line arguments.
    """
    source_path = args.source_path
    n_images = args.number_of_frames
    clean = args.clean
    full = args.full
    full_res = args.full_res
    
    if args.robust:
        print("Using robust reconstruction pipeline...")
        # Pass additional parameters to the robust pipeline
        do_one_robust(
            source_path,
            n_images,
            clean,
            minimal=args.minimal,
            full=full,
            full_res=full_res,
            random_ratio=args.random_ratio,
            pruning_threshold=args.pruning_threshold,
            coverage_weight=args.coverage_weight,
            triangulation_weight=args.triangulation_weight,
            diversity_weight=args.diversity_weight,
            confidence_weight=args.confidence_weight
        )
    else:
        print("Using standard reconstruction pipeline...")
        do_one(source_path, n_images, clean, minimal=args.minimal, full=full, full_res=full_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COLMAP reconstruction pipeline for video processing")
    parser.add_argument("--source_path", "-s", required=True, type=str,
                        help="Path to the directory containing the video")
    parser.add_argument("--number_of_frames", "-n", default=200, type=int,
                        help="Target number of frames for reconstruction (default: 200)")
    parser.add_argument("--clean", "-c", action='store_true',
                        help="Clean existing paths before processing")
    parser.add_argument("--minimal", "-m", action='store_true',
                        help="Use minimal frame selection after final reconstruction")
    parser.add_argument("--full", "-f", action='store_true',
                        help="Use all frame selection after final reconstruction")
    parser.add_argument("--full_res", action='store_true',
                        help="Extract final frames at full resolution")
    parser.add_argument("--robust", "-r", action='store_true',
                        help="Use robust reconstruction pipeline with pose interpolation")
    
    # Additional parameters for the robust pipeline
    parser.add_argument("--random_ratio", type=float, default=0.2,
                        help="Ratio of frames to select randomly (default: 0.2)")
    parser.add_argument("--pruning_threshold", type=float, default=0.05,
                        help="Threshold for pruning low-contribution frames (default: 0.05)")
    parser.add_argument("--coverage_weight", type=float, default=0.4,
                        help="Weight for coverage score in frame selection (default: 0.4)")
    parser.add_argument("--triangulation_weight", type=float, default=0.3,
                        help="Weight for triangulation score in frame selection (default: 0.3)")
    parser.add_argument("--diversity_weight", type=float, default=0.2,
                        help="Weight for diversity score in frame selection (default: 0.2)")
    parser.add_argument("--confidence_weight", type=float, default=0.1,
                        help="Weight for confidence score in frame selection (default: 0.1)")
    
    args = parser.parse_args()
    main(args)
