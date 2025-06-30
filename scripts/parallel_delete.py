import os
import csv
import argparse
import multiprocessing
from tqdm import tqdm
import sys

def read_paths_from_csv(csv_filepath, column_name):
    """Reads a list of file paths from a specific column in a CSV file."""
    paths = []
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    paths.append(row[column_name])
                except KeyError:
                    print(f"Error: Column '{column_name}' not found in the CSV file.", file=sys.stderr)
                    print(f"Available columns are: {reader.fieldnames}", file=sys.stderr)
                    sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.", file=sys.stderr)
        sys.exit(1)
    return paths

def delete_file(filepath):
    """
    Worker function to delete a single file.
    Returns the filepath and a status message.
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return (filepath, 'DELETED')
        else:
            return (filepath, 'NOT_FOUND')
    except Exception as e:
        return (filepath, f'ERROR: {e}')

def main():
    parser = argparse.ArgumentParser(
        description="Deletes files in parallel based on a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--csv_file',
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        '-p', '--path_column',
        type=str,
        default='filepath',
        help="The name of the column containing the file paths (default: 'filepath')."
    )
    parser.add_argument(
        '-n', '--num_workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use (default: all CPU cores)."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform a dry run. Shows what would be deleted without actually deleting anything."
    )
    args = parser.parse_args()

    # Read the list of files to be deleted.
    file_paths = read_paths_from_csv(args.csv_file, args.path_column)

    if not file_paths:
        print("No file paths found in the CSV. Exiting.")
        return

    print(f"Found {len(file_paths)} file paths in '{args.csv_file}'.")

    # Handle the dry-run scenario.
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("The following files would be deleted:")
        for path in file_paths:
            print(f"[DRY RUN] Would delete: {path}")
        print("\nNo files were actually deleted.")
        return

    # Safety confirmation prompt before proceeding with actual deletion.
    try:
        confirm = input(f"\n!!! WARNING !!!\nYou are about to permanently delete {len(file_paths)} files.\nThis action cannot be undone. Are you sure you want to continue? (type 'yes' to confirm): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled by user.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\nDeletion cancelled by user.")
        return

    # Use a multiprocessing pool to delete files in parallel.
    print("\nStarting deletion process...")
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(delete_file, file_paths),
            total=len(file_paths),
            desc="Deleting Files"
        ))

    # Tally the results.
    deleted_count = sum(1 for _, status in results if status == 'DELETED')
    not_found_count = sum(1 for _, status in results if status == 'NOT_FOUND')
    error_count = sum(1 for _, status in results if 'ERROR' in status)

    print("\n--- Deletion Summary ---")
    print(f"Successfully deleted: {deleted_count}")
    print(f"Files not found:     {not_found_count}")
    print(f"Errors encountered:  {error_count}")

    if error_count > 0:
        print("\nFiles that failed to delete:")
        for path, status in results:
            if 'ERROR' in status:
                print(f"- {path} ({status})")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()