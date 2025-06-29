# import os
# import cv2
# import argparse
# import numpy as np
# import torch
# from transformers import AutoImageProcessor, AutoModel
# from transformers import SiglipModel, Siglip2Model, CLIPModel, Dinov2Model
# from PIL import Image
# from tqdm import tqdm
# import multiprocessing

# def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv')):
#     video_files = []
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.lower().endswith(exts):
#                 full_path = os.path.join(root, file)
#                 rel_dir = os.path.relpath(root, input_dir)
#                 video_files.append((full_path, rel_dir, file))
#     return video_files

# def extract_cls_features(frames, processor, model, device):
#     images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
#     inputs = processor(images=images, return_tensors="pt").to(device)
#     with torch.no_grad():
#         if isinstance(model, (Siglip2Model, SiglipModel)):
#             feats = model.get_image_features(**inputs)
#         elif isinstance(model, (CLIPModel, )):
#             feats = model.get_image_features(**inputs)
#         elif isinstance(model, (Dinov2Model, )):
#             outputs = model(**inputs)
#             feats = outputs.last_hidden_state[:, 0, :]
#         feats = torch.nn.functional.normalize(feats, dim=1)
#     return feats.cpu().numpy()

# def estimate_shot_boundaries(
#     video_path, min_len, max_len, sim_thresh, std_factor, device, model_name, processor_name
# ):
#     # Load processor and model
#     processor = AutoImageProcessor.from_pretrained(processor_name)
#     model = AutoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
#     model.eval()

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open {video_path}")
#         return []

#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     if len(frames) == 0:
#         return [0]
#     # Batch extract features
#     BATCH = 32
#     all_feats = []
#     for i in tqdm(range(0, len(frames), BATCH), desc=os.path.basename(video_path)):
#         feats = extract_cls_features(frames[i:i+BATCH], processor, model, device)
#         all_feats.append(feats)
#     feats = np.concatenate(all_feats, axis=0)

#     sims = np.sum(feats[1:] * feats[:-1], axis=1)
#     mean_sim = np.mean(sims)
#     std_sim = np.std(sims)

#     boundaries = [0]
#     last_shot = 0
#     for i, sim in enumerate(sims):
#         shot_len = i + 1 - last_shot
#         is_outlier = abs(sim - mean_sim) > std_factor * std_sim
#         is_below = sim < sim_thresh
#         if (is_outlier or is_below) and min_len <= shot_len <= max_len:
#             boundaries.append(i + 1)
#             last_shot = i + 1
#         elif shot_len > max_len:
#             boundaries.append(last_shot + max_len)
#             last_shot += max_len
#     if boundaries[-1] != len(frames):
#         boundaries.append(len(frames))
#     return boundaries

# def save_shots(video_path, rel_dir, filename, output_dir, shot_bounds, min_len, max_len):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open {video_path}")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     shot_output_dir = os.path.join(output_dir, rel_dir)
#     os.makedirs(shot_output_dir, exist_ok=True)

#     for idx in range(len(shot_bounds) - 1):
#         start = shot_bounds[idx]
#         end = shot_bounds[idx + 1]
#         shot_len = end - start
#         if shot_len < min_len or shot_len > max_len:
#             continue
#         chunk_filename = f"{os.path.splitext(filename)[0]}_shot{idx:04d}.mp4"
#         chunk_path = os.path.join(shot_output_dir, chunk_filename)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
#         if not out.isOpened():
#             print(f"Failed to write {chunk_path}")
#             continue
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start)
#         for _ in range(shot_len):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
#         out.release()
#     cap.release()

# def process_video(args_tuple):
#     video_tuple, args, local_device = args_tuple
#     video_path, rel_dir, filename = video_tuple
#     shot_bounds = estimate_shot_boundaries(
#         video_path, args.min_frame_len, args.max_frame_len, args.sim_thresh, args.std_factor,
#         local_device, args.model_name, args.processor_name
#     )
#     if len(shot_bounds) <= 1:
#         print(f"No shots found in {video_path}")
#         return
#     save_shots(
#         video_path, rel_dir, filename,
#         args.output_dir, shot_bounds,
#         args.min_frame_len, args.max_frame_len
#     )

# def get_available_gpus():
#     if not torch.cuda.is_available():
#         return []
#     return list(range(torch.cuda.device_count()))

# def main():
#     parser = argparse.ArgumentParser(description="Visual-Encoder-based semantic shot detection")
#     parser.add_argument('-i', "--input_dir", type=str, required=True, help="Input video directory")
#     parser.add_argument('-o', "--output_dir", type=str, required=True, help="Output shot directory")
#     parser.add_argument('-n', "--num_workers", type=int, default=1, help="Number of parallel workers (per-GPU, not shared)")
#     parser.add_argument("--min_frame_len", type=int, default=30, help="Minimum shot length (frames)")
#     parser.add_argument("--max_frame_len", type=int, default=3000, help="Maximum shot length (frames)")
#     parser.add_argument("--sim_thresh", type=float, default=0.8, help="Similarity threshold for shot boundary")
#     parser.add_argument("--std_factor", type=float, default=1.0, help="Similarity std Times for shot boundary")
#     parser.add_argument("--model_name", type=str, default="facebook/dinov2-small", help="HuggingFace model name")
#     parser.add_argument("--processor_name", type=str, default=None, help="HuggingFace processor name (default: same as model_name)")
#     args = parser.parse_args()

#     # Default processor name to model name if not set
#     if args.processor_name is None:
#         args.processor_name = args.model_name

#     video_files = find_video_files(args.input_dir)
#     if not video_files:
#         print("No video files found.")
#         return

#     gpus = get_available_gpus()
#     if not gpus:
#         devices = ['cpu'] * args.num_workers
#     else:
#         # Assign GPU index round-robin to each worker
#         devices = [f'cuda:{gpus[i % len(gpus)]}' for i in range(args.num_workers)]

#     tasks = []
#     for i, video_tuple in enumerate(video_files):
#         # Assign device per task, round-robin
#         local_device = devices[i % len(devices)]
#         tasks.append((video_tuple, args, local_device))

#     if args.num_workers == 1:
#         for t in tasks:
#             process_video(t)
#     else:
#         with multiprocessing.Pool(args.num_workers) as pool:
#             pool.map(process_video, tasks)

# if __name__ == "__main__":
#     main()

import os
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm
import subprocess
import shutil

# PySceneDetect is the core library for scene detection.
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def check_dependencies():
    """Checks if ffmpeg is in the system's PATH, as it's required by PySceneDetect for splitting."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv')):
    """Recursively finds all video files in the input directory."""
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                # Get the relative path to reconstruct the output directory structure.
                rel_dir = os.path.relpath(root, input_dir)
                if rel_dir == ".": rel_dir = ""
                video_files.append((full_path, rel_dir, file))
    return video_files

def process_video(video_tuple, args):
    """
    Processes a single video file to detect scenes and split it into shots.
    This function is the target for each worker in the multiprocessing pool.
    """
    video_path, rel_dir, filename = video_tuple
    base_name = os.path.splitext(filename)[0]
    pid = os.getpid()

    try:
        # Open the video and create a SceneManager.
        video = open_video(video_path)
        scene_manager = SceneManager()

        # Add a content detector. The threshold is the most important parameter to tune.
        # A lower value makes it more sensitive, a higher value less sensitive.
        # Common values are between 27 and 32.
        scene_manager.add_detector(ContentDetector(threshold=args.threshold, min_scene_len=args.min_frame_len))

        # Perform scene detection.
        scene_manager.detect_scenes(video=video, show_progress=False)

        # Get the list of detected scenes.
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            print(f"[Worker {pid}] No scenes detected in {filename}, skipping.")
            return

        # Define the output directory for the shots.
        shot_output_dir = os.path.join(args.output_dir, rel_dir)
        os.makedirs(shot_output_dir, exist_ok=True)

        # Use PySceneDetect's built-in ffmpeg splitter, which is fast and efficient.
        # It uses `-c copy` by default, ensuring lossless and fast splitting.
        # We define a custom file name pattern.
        file_name_pattern = f"{base_name}_shot_$SCENE_NUMBER.mp4"
        
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_file_template=os.path.join(shot_output_dir, file_name_pattern),
            show_progress=False,
            suppress_output=True # Hide ffmpeg command output.
        )
        
        # Optional: Filter shots by length after splitting.
        # This is an alternative to using min_scene_len in the detector.
        # For now, we rely on min_scene_len.

        print(f"[Worker {pid}] Finished processing {filename}, found {len(scene_list)} shots.")

    except Exception as e:
        print(f"An error occurred while processing {filename} on worker {pid}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="High-performance, multi-process shot detector using PySceneDetect.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Input directory of video chunks.")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="Output directory for detected shots.")
    parser.add_argument('-n', "--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers (default: all CPU cores).")
    parser.add_argument('-t', "--threshold", type=float, default=28.0, 
                        help="Detection sensitivity threshold for ContentDetector (default: 28.0).\n"
                             "Lower values are more sensitive, higher values are less sensitive.")
    parser.add_argument("--min_frame_len", type=int, default=24, help="Minimum shot length in frames (default: 24).")
    args = parser.parse_args()

    check_dependencies()
    
    video_files = find_video_files(args.input_dir)
    if not video_files:
        print("No video files found.")
        return
    print(f"Found {len(video_files)} video files to process with {args.num_workers} workers.")

    # Create a partial function to pass the fixed 'args' to the process_video function.
    process_func = partial(process_video, args=args)

    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Use tqdm to show a progress bar for the overall process.
            # imap_unordered is efficient as it yields results as soon as they are ready.
            list(tqdm(pool.imap_unordered(process_func, video_files), total=len(video_files), desc="Overall Progress"))
    else:
        # Single-process mode for easier debugging.
        print("Running in single-process mode.")
        for video_file in tqdm(video_files, desc="Overall Progress"):
            process_func(video_file)

    print("\nAll videos have been processed.")

if __name__ == "__main__":
    main()