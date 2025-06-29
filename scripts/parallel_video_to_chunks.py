import os
import cv2
import argparse
import multiprocessing

def find_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv')):
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_dir)
                video_files.append((full_path, rel_dir, file))
    return video_files

def get_chunk_tasks(video_tuple, output_dir, chunk_size, begin_rate=0.05, end_rate=0.95):
    video_path, rel_dir, filename = video_tuple
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start_frame = int(frame_count * begin_rate)
    end_frame = int(frame_count * end_rate)
    usable_frames = end_frame - start_frame
    if usable_frames < chunk_size:
        return []
    chunk_output_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(chunk_output_dir, exist_ok=True)
    chunk_tasks = []
    chunk_idx = 0
    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        if chunk_end - chunk_start < chunk_size:
            break  # skip last incomplete chunk
        chunk_filename = f"{os.path.splitext(filename)[0]}_{chunk_idx:04d}.mp4"
        chunk_path = os.path.join(chunk_output_dir, chunk_filename)
        chunk_tasks.append({
            "video_path": video_path,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "output_path": chunk_path
        })
        chunk_idx += 1
    return chunk_tasks

def process_chunk(task):
    video_path = task["video_path"]
    chunk_start = task["chunk_start"]
    chunk_end = task["chunk_end"]
    output_path = task["output_path"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Failed to write to {output_path}")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
    frame_written = 0
    for _ in range(chunk_end - chunk_start):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
        frame_written += 1

    out.release()
    cap.release()
    if frame_written == 0:
        print(f"No frames written to {output_path}, file may be invalid.")
        if os.path.exists(output_path):
            os.remove(output_path)

def main():
    parser = argparse.ArgumentParser(description="Parallel chunked video splitting")
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Input video directory")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="Output chunk directory")
    parser.add_argument('-c', "--chunk_size", type=int, required=True, help="Chunk size (frames)")
    parser.add_argument('-n', "--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument('-b', "--begin_rate", type=float, default=0.05)
    parser.add_argument('-e', "--end_rate", type=float, default=0.95)
    args = parser.parse_args()

    video_files = find_video_files(args.input_dir)
    all_tasks = []
    for video_tuple in video_files:
        all_tasks.extend(get_chunk_tasks(video_tuple, args.output_dir, args.chunk_size, args.begin_rate, args.end_rate))

    if not all_tasks:
        print("No video chunks to process.")
        return

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.map(process_chunk, all_tasks)

if __name__ == "__main__":
    main()