import os
import csv
import argparse
import subprocess
import multiprocessing
import numpy as np
from tqdm import tqdm
import shutil
import sys
from datetime import timedelta
import cv2

def check_ffmpeg_installed():
    """
    Checks if ffmpeg and ffprobe are available in the system's PATH.
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Error: ffmpeg or ffprobe is not installed or not in the system's PATH.", file=sys.stderr)
        print("Please install the FFmpeg suite to run this script.", file=sys.stderr)
        sys.exit(1)

def find_video_files(input_dir):
    """Recursively finds all video files in the input directory."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def analyze_video(filepath):
    """
    Analyzes a video file using an FFmpeg-to-OpenCV pipeline.
    Returns a tuple: (filepath, duration_str, resolution_str, avg_score).
    """
    duration_str = 'N/A'
    resolution_str = 'N/A'
    width, height = 0, 0
    average_score = -1.0

    # Step 1: Get metadata with ffprobe. This is essential to know frame dimensions.
    try:
        probe_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'csv=p=0:s=,'
        ]
        probe_command.append(filepath)
        
        probe_result = subprocess.run(
            probe_command, capture_output=True, text=True, check=True
        )
        
        # Parse ffprobe output, e.g., "1920,1080,30.123"
        w_str, h_str, dur_str = probe_result.stdout.strip().split(',')
        width, height = int(w_str), int(h_str)
        resolution_str = f"{width}x{height}"
        duration_str = str(timedelta(seconds=float(dur_str)))

    except (subprocess.CalledProcessError, ValueError, IndexError):
        # If ffprobe fails, we cannot proceed.
        return (filepath, 'PROBE_FAILED', 'PROBE_FAILED', -1.0)

    # Step 2: Create an FFmpeg pipeline to decode video to raw frames (stdout).
    ffmpeg_command = [
        'ffmpeg',
        '-i', filepath,
        '-f', 'image2pipe',      # Output format is a pipe of images
        '-pix_fmt', 'bgr24',     # Pixel format OpenCV understands
        '-vcodec', 'rawvideo',   # Codec is raw video data
        '-'                      # Output to stdout
    ]

    process = None
    try:
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        scores = []
        frame_size = width * height * 3  # 3 bytes per pixel for bgr24

        while True:
            # Read one frame's worth of data from the pipe.
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                break
            
            # Convert the raw byte data to a NumPy array.
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            
            # Calculate blurriness using the variance of the Laplacian.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(laplacian_var)
            
        if scores:
            average_score = np.mean(scores)

    except Exception:
        # In case of any error during the pipeline processing.
        average_score = -1.0
    finally:
        if process:
            process.terminate() # Ensure the ffmpeg process is killed.

    return (filepath, duration_str, resolution_str, round(average_score, 5))

def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze video files for clarity using an FFmpeg and OpenCV pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Directory to scan recursively for video files.")
    parser.add_argument('-o', '--output_csv', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('-n', '--num_workers', type=int, default=8, help="Number of parallel processes to use (default: all CPU cores).")
    args = parser.parse_args()

    check_ffmpeg_installed()

    print(f"Scanning for video files in '{args.input_dir}'...")
    video_files = find_video_files(args.input_dir)

    if not video_files:
        print("No video files found. Exiting.")
        return

    print(f"Found {len(video_files)} videos. Starting analysis with {args.num_workers} workers...")

    # Use a multiprocessing pool for parallel execution.
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(analyze_video, video_files),
            total=len(video_files),
            desc="Analyzing Videos"
        ))

    # Write the results to the CSV file.
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filepath', 'duration', 'resolution', 'average_laplacian_score'])
            writer.writerows(results)
        print(f"\nAnalysis complete. Results saved to '{args.output_csv}'")
    except IOError as e:
        print(f"\nError writing to CSV file: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Set start method for better compatibility on macOS and Windows.
    multiprocessing.set_start_method('spawn', force=True)
    main()