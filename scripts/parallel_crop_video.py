import os
import cv2
import argparse
from multiprocessing import Pool

def crop_video(task):
    video_path = task["video_path"]
    output_path = task["output_path"]
    left, top, right, bottom = task["crop_bounds"]
    frame_start = task.get("frame_start", None)
    frame_end = task.get("frame_end", None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    crop_width = right - left
    crop_height = bottom - top
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
    if not out.isOpened():
        print(f"Failed to write to {output_path}")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = frame_start if frame_start is not None else 0
    end = frame_end if frame_end is not None else total_frames

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= end:
            break
        if frame_idx >= start:
            crop = frame[top:bottom, left:right]
            out.write(crop)
            written += 1
        frame_idx += 1

    out.release()
    cap.release()
    if written == 0 and os.path.exists(output_path):
        os.remove(output_path)
        print(f"No frames written to {output_path}, file deleted.")

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel crop video tool")
    parser.add_argument("--input_dir", required=True, help="Input video directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--left", type=int, required=True, help="Left crop boundary")
    parser.add_argument("--top", type=int, required=True, help="Top crop boundary")
    parser.add_argument("--right", type=int, required=True, help="Right crop boundary")
    parser.add_argument("--bottom", type=int, required=True, help="Bottom crop boundary")
    parser.add_argument("--frame_start", type=int, default=None, help="Start frame index (inclusive)")
    parser.add_argument("--frame_end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    return parser.parse_args()

def find_video_files(input_dir):
    exts = ['.mp4', '.avi', '.mov', '.mkv']
    files = []
    for fname in os.listdir(input_dir):
        if any(fname.lower().endswith(ext) for ext in exts):
            files.append(os.path.join(input_dir, fname))
    return files

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    video_files = find_video_files(args.input_dir)
    tasks = []
    for video_path in video_files:
        fname = os.path.basename(video_path)
        output_path = os.path.join(args.output_dir, fname)
        task = {
            "video_path": video_path,
            "output_path": output_path,
            "crop_bounds": (args.left, args.top, args.right, args.bottom),
        }
        if args.frame_start is not None:
            task["frame_start"] = args.frame_start
        if args.frame_end is not None:
            task["frame_end"] = args.frame_end
        tasks.append(task)

    with Pool(args.num_workers) as pool:
        pool.map(crop_video, tasks)

if __name__ == "__main__":
    main()