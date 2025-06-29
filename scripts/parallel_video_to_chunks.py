import os
import argparse
import multiprocessing
import subprocess
import json
import sys
import shutil
from functools import partial

def check_dependencies():
    """Checks if ffmpeg and ffprobe are in the system's PATH."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    if not shutil.which("ffprobe"):
        print("Error: ffprobe not found. Please ensure it is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Recursively finds video files in the input directory.
    
    Returns a list of tuples, each containing (full_path, relative_dir, filename).
    """
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                # Create a path relative to the input directory to build the output structure.
                rel_dir = os.path.relpath(root, input_dir)
                if rel_dir == ".":
                    rel_dir = "" # The relative path is empty if in the root.
                video_files.append((full_path, rel_dir, file))
    return video_files

def split_video_with_ffmpeg(video_tuple, output_dir, chunk_size_frames):
    """
    Core function to split video into chunks using ffprobe and ffmpeg (no audio).
    """
    video_path, rel_dir, filename = video_tuple
    base_name = os.path.splitext(filename)[0]
    process_id = os.getpid() # Get current process ID for logging.
    
    print(f"[PID: {process_id}] Processing: {video_path}")

    # Step 1: Get video info using ffprobe.
    try:
        # Command to get duration and frame rate in JSON format.
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0', # Select only the first video stream.
            '-show_entries', 'format=duration:stream=r_frame_rate',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        # Extract duration and frame rate from the JSON output.
        total_duration = float(info['format']['duration'])
        frame_rate_str = info['streams'][0]['r_frame_rate']
        num, den = map(int, frame_rate_str.split('/'))
        fps = num / den
        
        if fps == 0:
            raise ValueError("Frame rate reported as 0.")

    except (subprocess.CalledProcessError, KeyError, ValueError, IndexError) as e:
        print(f"[PID: {process_id}] Error: Failed to get video info for '{filename}'. Skipping. Details: {e}", file=sys.stderr)
        return

    # Step 2: Calculate time and chunk parameters.
    start_time = total_duration * 0.1
    end_time = total_duration * 0.9
    usable_duration = end_time - start_time
    
    chunk_duration_seconds = chunk_size_frames / fps

    if usable_duration < chunk_duration_seconds:
        print(f"[PID: {process_id}] Warning: Usable duration for '{filename}' ({usable_duration:.2f}s) is less than a single chunk duration ({chunk_duration_seconds:.2f}s). Skipping.")
        return

    # Step 3: Prepare output directory and build the ffmpeg command.
    chunk_output_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(chunk_output_dir, exist_ok=True)
    
    output_pattern = os.path.join(chunk_output_dir, f"{base_name}_%04d.mp4")

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,         # Input file.
        '-ss', str(start_time),   # Global start time (trims first 10%).
        '-to', str(end_time),     # Global end time (trims last 10%).
        '-c:v', 'copy',           # Explicitly copy the video stream.
        '-an',                    # The key change: disable audio (Audio No).
        '-f', 'segment',          # Use the segment muxer for chunking.
        '-segment_time', str(chunk_duration_seconds), # Duration of each chunk.
        '-reset_timestamps', '1', # Reset timestamps for each chunk to start from 0.
        '-y',                     # Overwrite output files if they exist.
        output_pattern            # Output file naming pattern.
    ]

    # Step 4: Execute the ffmpeg command.
    print(f"[PID: {process_id}] Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"[PID: {process_id}] Successfully processed and chunked: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"[PID: {process_id}] Error: FFmpeg failed while processing '{filename}'.", file=sys.stderr)
        print(f"FFmpeg stderr:\n{e.stderr}", file=sys.stderr)


def main():
    """Main function to parse arguments and start the multiprocessing pool."""
    parser = argparse.ArgumentParser(
        description="Splits a directory of videos into chunks in parallel using FFmpeg (no audio). Trims the first and last 10% of each video.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Input directory containing video files.")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="Output directory to store the video chunks.")
    parser.add_argument('-c', "--chunk_size", type=int, required=True, help="The size of each chunk in frames.")
    parser.add_argument('-n', "--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers to use (default: number of CPU cores).")
    args = parser.parse_args()

    # 1. Check for dependencies.
    check_dependencies()

    # 2. Find all video files.
    print(f"Searching for video files in '{args.input_dir}'...")
    video_files = find_video_files(args.input_dir)
    if not video_files:
        print("No video files found.")
        return
    print(f"Found {len(video_files)} video files.")

    # 3. Process videos using a multiprocessing pool.
    print(f"Starting parallel processing with {args.num_workers} workers...")
    
    # Use partial to bind the fixed arguments (output_dir, chunk_size) to the processing function.
    process_func = partial(split_video_with_ffmpeg, output_dir=args.output_dir, chunk_size_frames=args.chunk_size)

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.map(process_func, video_files)
        
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()