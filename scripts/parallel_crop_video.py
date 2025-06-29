import os
import argparse
import subprocess
import multiprocessing
from functools import partial
from tqdm import tqdm
import shutil
import sys

def check_ffmpeg():
    """Checks if ffmpeg is installed and available in the system's PATH."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in your system's PATH.")
        print("Please install ffmpeg to use this script.")
        sys.exit(1)

def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Recursively finds all video files in the input directory and returns
    a list of (input_path, output_path) tuples.
    """
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                input_path = os.path.join(root, file)
                # Create the corresponding output path, preserving directory structure.
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
                tasks.append((input_path, output_path))
    return tasks

def crop_video(task, crop_filter, preset):
    """
    The worker function that processes a single video file.
    It takes a task tuple (input_path, output_path) and the crop filter string.
    """
    input_path, output_path = task
    pid = os.getpid()

    try:
        # Ensure the output directory for the current file exists.
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the ffmpeg command.
        # -vf crop=... : applies the video filter for cropping.
        # -c:v libx264 : re-encodes the video with the popular H.264 codec.
        # -preset: trades encoding speed for file size. 'veryfast' is a good balance.
        # -c:a copy : copies the audio stream without re-encoding (fast and lossless).
        # -y : overwrites the output file if it exists.
        # -nostdin: prevents ffmpeg from waiting for stdin, important for subprocesses.
        command = [
            'ffmpeg',
            '-nostdin',
            '-i', input_path,
            '-vf', crop_filter,
            '-c:v', 'libx264',
            '-preset', preset,
            '-c:a', 'copy',
            '-y',
            output_path
        ]

        # Execute the command, hiding its output for a cleaner console.
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return f"[Worker {pid}] Successfully cropped {os.path.basename(input_path)}"
        
    except subprocess.CalledProcessError as e:
        return f"[Worker {pid}] Failed to crop {os.path.basename(input_path)}. FFmpeg returned non-zero exit status."
    except Exception as e:
        return f"[Worker {pid}] An unexpected error occurred with {os.path.basename(input_path)}: {e}"

def main(args):
    check_ffmpeg()

    # Validate coordinates.
    if not (args.x2 > args.x1 and args.y2 > args.y1):
        print("Error: Invalid coordinates. Ensure x2 > x1 and y2 > y1.")
        return

    # Calculate crop dimensions for the ffmpeg filter.
    crop_w = args.x2 - args.x1
    crop_h = args.y2 - args.y1
    crop_x = args.x1
    crop_y = args.y1
    crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    
    print(f"Crop filter set to: {crop_filter}")

    tasks = find_video_files(args.input_dir)
    if not tasks:
        print("No video files found in the input directory.")
        return
        
    print(f"Found {len(tasks)} videos to process with {args.num_workers} workers.")

    # Use functools.partial to create a new function with the crop_filter argument already filled in.
    worker_func = partial(crop_video, crop_filter=crop_filter, preset=args.preset)

    # Create a multiprocessing pool to execute tasks in parallel.
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Use tqdm to create a progress bar.
        # pool.imap_unordered processes tasks in parallel and yields results as they complete.
        for result in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Cropping Videos"):
            # You can optionally print results from workers for more detailed logging.
            # print(result)
            pass

    print("\nAll videos have been cropped successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A high-performance script to crop all videos in a directory in parallel using FFmpeg."
    )
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Directory containing the source videos.")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="Directory to save the cropped videos.")
    parser.add_argument("--coords", type=str, required=True, help="Top-left and bottom-right coordinates as 'x1,y1,x2,y2'.")
    parser.add_argument('-n', "--num_workers", type=int, default=8, help="Number of parallel FFmpeg processes to run.")
    parser.add_argument('--preset', type=str, default='veryfast', help="FFmpeg encoding preset (e.g., ultrafast, veryfast, fast, medium).")

    # Parse arguments and then pass them to the main function.
    # This makes the code cleaner and easier to potentially import.
    try:
        args = parser.parse_args()
        # Parse the coordinate string.
        coords_list = [int(c.strip()) for c in args.coords.split(',')]
        if len(coords_list) != 4:
            raise ValueError
        args.x1, args.y1, args.x2, args.y2 = coords_list
        main(args)
    except ValueError:
        print("Error: Invalid format for --coords. Please use 'x1,y1,x2,y2', for example: '640,0,1280,720'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")