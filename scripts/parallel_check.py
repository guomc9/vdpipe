import os
import argparse
import subprocess
import multiprocessing
from functools import partial
from tqdm import tqdm
import shutil
import sys
from datetime import datetime

def check_ffmpeg():
    """Checks if ffmpeg is installed and available in the system's PATH."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    """Recursively finds all video files in the input directory."""
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                video_files.append(os.path.join(root, file))
    return video_files

def check_video_integrity(filepath):
    """
    Checks a single video file for corruption using ffmpeg.
    
    Returns a tuple: (filepath, status, error_message)
    where status is either 'OK' or 'CORRUPTED'.
    """
    try:
        # Command to fully decode the video and discard the output.
        # -v error: Only show fatal errors.
        # -f null -: Forces full processing by sending output to a null muxer.
        command = [
            'ffmpeg',
            '-v', 'error',
            '-i', filepath,
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # A file is corrupted if ffmpeg returns a non-zero exit code or prints to stderr.
        if result.returncode != 0 or result.stderr:
            error_details = result.stderr.strip()
            return (filepath, 'CORRUPTED', error_details)
        else:
            return (filepath, 'OK', '')
            
    except Exception as e:
        # Catch any Python errors during the subprocess call.
        return (filepath, 'CORRUPTED', f"Python error during check: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="A high-performance script to batch-check video files for corruption in parallel using FFmpeg.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Input directory containing video files.")
    parser.add_argument('-l', "--log_file", type=str, default=None, help="Optional: File to save the list of corrupted files and their errors.")
    parser.add_argument('-n', "--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel FFmpeg processes to run (default: all CPU cores).")
    args = parser.parse_args()

    check_ffmpeg()

    print(f"Searching for video files in '{args.input_dir}'...")
    video_files = find_video_files(args.input_dir)
    if not video_files:
        print("No video files found.")
        return
        
    print(f"Found {len(video_files)} video files. Starting check with {args.num_workers} workers...")

    corrupted_files = []
    ok_count = 0

    # Create a multiprocessing pool to execute checks in parallel.
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Use tqdm for a progress bar. imap_unordered is efficient.
        results = list(tqdm(pool.imap_unordered(check_video_integrity, video_files), total=len(video_files), desc="Checking videos"))

    # Process results after all checks are complete.
    for filepath, status, error_message in results:
        if status == 'CORRUPTED':
            corrupted_files.append((filepath, error_message))
        else:
            ok_count += 1

    # --- Final Report ---
    print("\n" + "="*20 + " Integrity Check Report " + "="*20)
    print(f"Total files scanned: {len(video_files)}")
    print(f"Healthy files: \033[92m{ok_count}\033[0m") # Green color
    print(f"Corrupted files: \033[91m{len(corrupted_files)}\033[0m") # Red color

    if corrupted_files:
        print("\n--- List of Corrupted Files ---")
        for filepath, error in corrupted_files:
            print(f"\nFile: {filepath}")
            print(f"  \033[93mError: {error}\033[0m") # Yellow color

        if args.log_file:
            print(f"\nWriting corrupted file log to: {args.log_file}")
            try:
                with open(args.log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Video Corruption Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*40 + "\n")
                    for filepath, error in corrupted_files:
                        f.write(f"File: {filepath}\n")
                        f.write(f"Error: {error}\n")
                        f.write("-" * 20 + "\n")
                print("Log file written successfully.")
            except Exception as e:
                print(f"Error: Could not write to log file. Reason: {e}", file=sys.stderr)
    
    print("\nCheck complete.")

if __name__ == "__main__":
    # Set start method for compatibility, especially on macOS and Windows.
    multiprocessing.set_start_method('spawn', force=True)
    main()