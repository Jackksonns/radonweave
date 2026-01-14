import os 
import requests
import tqdm
import hashlib
import zipfile
import tarfile
import concurrent.futures
import time
import subprocess
import sys

def check_aria2_available():
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: aria2 not found, will use the standard download method")
        print("Recommended to install aria2 for faster download: sudo apt-get install aria2")
        return False

def download_file_aria2(url, filename, max_connections=16, max_tries=5):
    print(f"Downloading with aria2 (high speed): {filename}")
    cmd = [
        "aria2c",
        "-x", str(max_connections),
        "-s", str(max_connections),
        "-k", "1M",
        "--max-tries=" + str(max_tries),
        "--retry-wait=3",
        "--check-certificate=false",
        "--file-allocation=none",
        "--console-log-level=notice",
        "-d", os.path.dirname(filename) or ".",
        "-o", os.path.basename(filename),
        url
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"aria2 download completed: {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"aria2 download failed: {filename}")
        print(f"Error message: {e}")
        return False

def download_file_fallback(url, filename=None, chunk_size=8192, verbose=False):
    if not filename:
        local_filename = os.path.join(".", url.split('/')[-1])
    else:
        local_filename = filename
    print(f"Using fallback download method: {local_filename}")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        file_size = int(r.headers.get('Content-Length', 0))
        with open(local_filename, 'wb') as fp:
            with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(local_filename)) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fp.write(chunk)
                        pbar.update(len(chunk))
        print(f"Fallback download completed: {local_filename}")
        return True
    except Exception as e:
        print(f"Fallback download failed: {local_filename}, error: {e}")
        return False

def compute_md5sum(filename, show_pbar=True):
    CHUNK_SIZE = 1024 * 1024
    hash_md5 = hashlib.md5()
    file_size = os.stat(filename).st_size
    num_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    with open(filename, 'rb') as fp:
        for _ in tqdm.tqdm(range(num_chunks), unit='chunks', desc=f'MD5 {os.path.basename(filename)}', disable=not show_pbar):
            chunk = fp.read(CHUNK_SIZE)
            if chunk:
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_file_integrity(filepath, expected_checksum):
    if not os.path.exists(filepath):
        return False, "File does not exist"
    print(f"Verifying file integrity: {os.path.basename(filepath)}")
    actual_checksum = "md5:" + compute_md5sum(filepath)
    if actual_checksum == expected_checksum:
        return True, "Checksum passed"
    else:
        return False, f"Checksum failed: expected {expected_checksum}, got {actual_checksum}"

def extract_archive(archive_path, extract_dir, target_files=None):
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")
    archive_ext = archive_path.split('.')[-1].lower()
    extracted_files = []
    print(f"Starting extraction: {os.path.basename(archive_path)}")
    if archive_ext == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            if target_files:
                files_to_extract = [f for f in all_files if any(target in f for target in target_files)]
                print(f"Extracting {len(files_to_extract)} target files from {os.path.basename(archive_path)}")
            else:
                files_to_extract = all_files
                print(f"Extracting all {len(files_to_extract)} files from {os.path.basename(archive_path)}")
            for file in tqdm.tqdm(files_to_extract, desc="Extracting files"):
                zip_ref.extract(file, extract_dir)
                extracted_files.append(file)
    elif archive_ext in ['tar', 'gz', 'bz2', 'xz']:
        mode = 'r'
        if archive_ext == 'gz':
            mode = 'r:gz'
        elif archive_ext == 'bz2':
            mode = 'r:bz2'
        elif archive_ext == 'xz':
            mode = 'r:xz'
        with tarfile.open(archive_path, mode) as tar_ref:
            all_files = tar_ref.getnames()
            if target_files:
                files_to_extract = [f for f in all_files if any(target in f for target in target_files)]
                print(f"Extracting {len(files_to_extract)} target files from {os.path.basename(archive_path)}")
            else:
                files_to_extract = all_files
                print(f"Extracting all {len(files_to_extract)} files from {os.path.basename(archive_path)}")
            for file in tqdm.tqdm(files_to_extract, desc="Extracting files"):
                tar_ref.extract(file, extract_dir)
                extracted_files.append(file)
    else:
        raise ValueError(f"Unsupported archive format: {archive_ext}")
    print(f"Extraction completed: {os.path.basename(archive_path)}")
    return len(extracted_files)

def get_zenodo_record_files(record_id):
    url = f'https://zenodo.org/api/records/{record_id}'
    print(f"Fetching file list for Zenodo record {record_id}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Unable to fetch Zenodo record: {response.status_code}")
    data = response.json()
    files = data['files']
    file_map = {}
    for f in files:
        filename = f['key']
        file_map[filename] = {
            'url': f['links']['self'],
            'size': f['size'],
            'checksum': f['checksum'],
            'type': filename.split('.')[-1].lower()
        }
    print(f"Found {len(files)} files")
    return file_map

def download_and_extract_archive(archive_info, download_dir, target_files=None, max_retries=3, use_aria2=True):
    filename = archive_info['filename']
    url = archive_info['url']
    expected_checksum = archive_info['checksum']
    local_path = os.path.join(download_dir, filename)
    if os.path.exists(local_path):
        print(f"Archive already exists, checking integrity: {filename}")
        is_valid, message = verify_file_integrity(local_path, expected_checksum)
        if is_valid:
            print(f"File is intact, extracting directly: {filename}")
            extracted_count = extract_archive(local_path, download_dir, target_files)
            return filename, f"extracted:{extracted_count}"
        else:
            print(f"File is incomplete, re-downloading: {filename} - {message}")
            os.remove(local_path)
    for attempt in range(max_retries):
        try:
            print(f"Starting archive download ({attempt+1}/{max_retries}): {filename}")
            if use_aria2 and check_aria2_available():
                download_success = download_file_aria2(url, local_path)
            else:
                download_success = download_file_fallback(url, local_path)
            if not download_success:
                continue
            is_valid, message = verify_file_integrity(local_path, expected_checksum)
            if is_valid:
                print(f"Download succeeded, starting extraction: {filename}")
                extracted_count = extract_archive(local_path, download_dir, target_files)
                return filename, f"success:{extracted_count}"
            else:
                print(f"File checksum verification failed: {filename} - {message}")
                if os.path.exists(local_path):
                    os.remove(local_path)
        except Exception as e:
            print(f"Download failed: {filename} (attempt {attempt+1}/{max_retries}), error: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return filename, f"failed: {e} after {max_retries} attempts"
    return filename, "failed: unknown error"

def identify_required_archives(missing_files, zenodo_files):
    required_archives = {}
    for missing_file in missing_files:
        if 'ground_truth_train' in missing_file:
            archive_name = 'ground_truth_train.zip'
        elif 'ground_truth_validation' in missing_file:
            archive_name = 'ground_truth_validation.zip'
        elif 'ground_truth_test' in missing_file:
            archive_name = 'ground_truth_test.zip'
        elif 'observation_train' in missing_file:
            archive_name = 'observation_train.zip'
        elif 'observation_validation' in missing_file:
            archive_name = 'observation_validation.zip'
        elif 'observation_test' in missing_file:
            archive_name = 'observation_test.zip'
        else:
            print(f"Unable to determine which archive file {missing_file} belongs to")
            continue
        if archive_name in zenodo_files:
            if archive_name not in required_archives:
                required_archives[archive_name] = {
                    'filename': archive_name,
                    'url': zenodo_files[archive_name]['url'],
                    'size': zenodo_files[archive_name]['size'],
                    'checksum': zenodo_files[archive_name]['checksum'],
                    'missing_files': []
                }
            required_archives[archive_name]['missing_files'].append(missing_file)
        else:
            print(f"Warning: archive {archive_name} is not in the Zenodo record")
    return required_archives

def cleanup_archives(download_dir, archive_names):
    print("\nCleaning up archive files...")
    removed_count = 0
    for archive_name in archive_names:
        archive_path = os.path.join(download_dir, archive_name)
        if os.path.exists(archive_path):
            try:
                os.remove(archive_path)
                print(f"Deleted: {archive_name}")
                removed_count += 1
            except Exception as e:
                print(f"Delete failed: {archive_name} - {e}")
    print(f"Cleanup completed: deleted {removed_count} archive files")

def verify_final_files(download_dir, expected_files):
    print("\nVerifying final file completeness...")
    missing_files = []
    existing_files = []
    for expected_file in expected_files:
        file_path = os.path.join(download_dir, expected_file)
        if os.path.exists(file_path):
            existing_files.append(expected_file)
        else:
            missing_files.append(expected_file)
    print(f"File summary:")
    print(f"  Total files: {len(expected_files)}")
    print(f"  Existing files: {len(existing_files)}")
    print(f"  Missing files: {len(missing_files)}")
    if missing_files:
        print("Missing files:")
        for missing in missing_files[:10]:
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more files")
    return len(missing_files) == 0

def parallel_download_and_extract(record_id, missing_files_path, download_dir=None, max_workers=3, max_retries=3, use_aria2=True, cleanup_after=True):
    with open(missing_files_path, 'r') as f:
        missing_files = [line.strip() for line in f if line.strip()]
    print(f"Found {len(missing_files)} missing files")
    if download_dir is None:
        download_dir = os.path.dirname(missing_files_path) or "."
    os.makedirs(download_dir, exist_ok=True)
    zenodo_files = get_zenodo_record_files(record_id)
    required_archives = identify_required_archives(missing_files, zenodo_files)
    if not required_archives:
        print("No required archives found to download")
        return False
    print(f"Need to download {len(required_archives)} archives:")
    archive_names = []
    for archive_name, info in required_archives.items():
        print(f"  - {archive_name} (contains {len(info['missing_files'])} missing files)")
        archive_names.append(archive_name)
    download_tasks = []
    for archive_name, archive_info in required_archives.items():
        download_tasks.append((archive_info, download_dir, archive_info['missing_files'], max_retries, use_aria2))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_and_extract_archive, *task): task[0]['filename'] for task in download_tasks}
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((filename, f"exception: {e}"))
    success_count = sum(1 for _, status in results if status.startswith("success") or status.startswith("extracted"))
    failed_count = sum(1 for _, status in results if status.startswith("failed") or status.startswith("exception"))
    print(f"\nDownload and extraction summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    for filename, status in results:
        print(f"  {filename}: {status}")
    all_complete = verify_final_files(download_dir, missing_files)
    if cleanup_after and all_complete:
        cleanup_archives(download_dir, archive_names)
    return all_complete and failed_count == 0

def main():
    ZENODO_RECORD_ID = "3384092"
    MISSING_FILES_PATH = "/root/autodl-tmp/datasets/lodopab/missing_files.txt"
    DOWNLOAD_DIR = None
    MAX_WORKERS = 10
    MAX_RETRIES = 5
    USE_ARIA2 = True
    CLEANUP_AFTER = True
    print("Starting high-speed download and extraction of missing files...")
    print(f"  Zenodo Record ID: {ZENODO_RECORD_ID}")
    print(f"  Worker threads: {MAX_WORKERS}")
    print(f"  Max retries: {MAX_RETRIES}")
    print(f"  Use aria2 acceleration: {'Yes' if USE_ARIA2 else 'No'}")
    print(f"  Cleanup archives after download: {'Yes' if CLEANUP_AFTER else 'No'}")
    start_time = time.time()
    try:
        success = parallel_download_and_extract(
            ZENODO_RECORD_ID,
            MISSING_FILES_PATH,
            DOWNLOAD_DIR,
            MAX_WORKERS,
            MAX_RETRIES,
            USE_ARIA2,
            CLEANUP_AFTER
        )
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if success:
            print("All files downloaded, extracted, and verified successfully")
            print("The dataset should now be complete")
        else:
            print("Some files failed to process; please check the error messages above")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
