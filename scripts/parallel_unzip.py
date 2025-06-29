import argparse
import os
import subprocess
import zipfile
from multiprocessing import Pool

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Parallel unzip folders from a zip archive.")
    parser.add_argument('-z', '--zip', required=True, help='Path to the zip archive.')
    parser.add_argument('-t', '--target', required=True, help='Target directory for extraction.')
    
    # Group for mutually exclusive folder sources
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--folders', default=None,
                        help='(Optional) Text file with folders to extract, one per line.')
    group.add_argument('-s', '--subfolder', default=None,
                        help='(Optional) A subfolder path inside the zip to scan for child directories to extract.')

    parser.add_argument('-e', '--extracted', required=False,
                        help='(Optional) Text file with already extracted folders, one per line, to be skipped.')
    parser.add_argument('-p', '--processes', type=int, default=8, help='Max concurrent processes for extraction.')
    return parser.parse_args()

def load_folders_from_file(folders_file):
    """Loads a list of folders from a text file."""
    try:
        # Using utf-8 as a more common default, but can be changed if needed.
        with open(folders_file, 'r', encoding='utf-8') as f:
            return set(line.strip().rstrip('/') + '/' for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: Folders file not found at '{folders_file}'")
        return set()
    except Exception as e:
        print(f"Error reading folders file: {e}")
        return set()

def get_folders_from_zip(zip_file_path, base_path=None):
    """
    Scans a zip archive and returns a set of folders to be extracted.
    - If base_path is None, returns all top-level folders.
    - If base_path is provided, returns all immediate child directories of base_path.
    """
    if not os.path.exists(zip_file_path):
        print(f"Error: Zip file not found at '{zip_file_path}'")
        return set()

    folders = set()
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            if base_path:
                # Ensure base_path ends with a slash for correct matching
                base_path = base_path.strip().rstrip('/') + '/'
                print(f"Scanning for child directories inside '{base_path}'...")
                for member in zf.namelist():
                    if member.startswith(base_path):
                        # Get the rest of the path after the base_path
                        relative_path = member[len(base_path):]
                        # Find the first directory level in the relative path
                        if '/' in relative_path:
                            child_dir = relative_path.split('/', 1)[0] + '/'
                            folders.add(base_path + child_dir)
            else:
                print("Scanning for top-level folders in the archive root...")
                for member in zf.namelist():
                    if '/' in member:
                        top_level_folder = member.split('/', 1)[0] + '/'
                        folders.add(top_level_folder)
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred while scanning the zip file: {e}")
        
    return folders

def unzip_folder(args_tuple):
    """Worker function to unzip a single folder from the archive."""
    zip_file, folder, target_dir = args_tuple
    print(f"Extracting: {folder}")
    try:
        cmd = ['unzip', '-n', zip_file, f'{folder}*', '-d', target_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"Done: {folder}")
        else:
            print(f"Finished with issues or error for: {folder}\nStderr: {result.stderr.strip()}")
            
    except FileNotFoundError:
        print("Error: 'unzip' command not found. Please ensure it is installed and in your PATH.")
    except Exception as e:
        print(f"An exception occurred while processing {folder}: {e}")

def main():
    """Main function to coordinate the extraction process."""
    args = parse_args()
    
    folders_to_process = set()

    # Determine the source of the folder list
    if args.folders:
        print(f"Loading folder list from file: {args.folders}")
        folders_to_process = load_folders_from_file(args.folders)
    elif args.subfolder:
        folders_to_process = get_folders_from_zip(args.zip, base_path=args.subfolder)
    else:
        folders_to_process = get_folders_from_zip(args.zip, base_path=None)

    if not folders_to_process:
        print("No folders found to process. Exiting.")
        return
    
    print(f"Found {len(folders_to_process)} target folders.")

    # Load the list of already extracted folders to skip them
    extracted = set()
    if args.extracted and os.path.exists(args.extracted):
        print(f"Loading list of already extracted folders from: {args.extracted}")
        extracted = load_folders_from_file(args.extracted)
    
    # Determine the final list of folders to extract
    to_extract = sorted(list(folders_to_process - extracted))
    
    if not to_extract:
        print("All target folders have already been extracted. Nothing to do.")
        return
        
    print(f"Total folders to extract: {len(to_extract)}")
    
    # Prepare argument tuples for the multiprocessing pool
    task_args = [(args.zip, folder, args.target) for folder in to_extract]
    
    # Run the extraction in parallel
    with Pool(processes=args.processes) as pool:
        pool.map(unzip_folder, task_args)
        
    print("\nAll extraction tasks are complete.")

if __name__ == '__main__':
    main()