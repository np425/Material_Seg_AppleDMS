import os
import json
import gzip
import requests
import posixpath
import urllib.parse
from concurrent import futures
from tqdm import tqdm
import time
from threading import Lock
import logging
import random

# --- CONFIGURATION ---
DATA_PATH = "/home/nvidia/material_classification/datasets/apple-dms/DMS_v1"
DOWNLOAD_FOLDER = os.path.join(DATA_PATH, "raw_images")
LOG_FILE = os.path.join(DATA_PATH, "download_report.log")
MAX_THREADS = 50 
# ---------------------

# Setup Logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# Global Lock for Flickr throttling
flickr_lock = Lock()

def get_filename(url):
    return posixpath.split(urllib.parse.urlparse(url).path)[1]

def download_one_image(session, datum, download_folder):
    original_url = datum['openimages_metadata']['OriginalURL']
    filename = get_filename(original_url)
    if not filename: return "skipped_bad_name"
    
    local_path = os.path.join(download_folder, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return "skipped_exists"

    image_id = datum['openimages_metadata'].get('ImageID')
    split = datum['openimages_metadata'].get('Subset', 'train') # Default to train if missing

    # --- SOURCE 1: AWS S3 (V7 Live Bucket) ---
    # Fast, but missing old files.
    if image_id:
        s3_url = f"https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"
        try:
            r = session.get(s3_url, timeout=3)
            if r.status_code == 200:
                with open(local_path, 'wb') as f: f.write(r.content)
                return "success_s3_aws"
        except: pass

    # --- SOURCE 2: Google Cloud Storage (V4 Archive) ---
    # The "Time Machine" mirror for 2018 files.
    if image_id:
        # Note: 'validation' and 'test' are standard, but check 'train' specifically
        gcs_url = f"https://storage.googleapis.com/openimages/2018_04/{split}/{image_id}.jpg"
        try:
            r = session.get(gcs_url, timeout=3)
            if r.status_code == 200:
                with open(local_path, 'wb') as f: f.write(r.content)
                return "success_gcs_mirror"
        except: pass

    # --- SOURCE 3: Flickr (Original Source) ---
    # Slow, rate-limited, but the final truth.
    with flickr_lock:
        try:
            # Random sleep between 2.0 and 4.0 seconds
            # This mimics human browsing behavior much better than a fixed 0.5s
            sleep_time = random.uniform(2.0, 4.0)
            time.sleep(sleep_time)

            r = session.get(original_url, timeout=10)
            if r.status_code == 200:
                with open(local_path, 'wb') as f: f.write(r.content)
                return "success_flickr"
            elif r.status_code == 410:
                logging.error(f"DEAD: {filename} (410 Gone from all sources)")
                return "failed_410"
            elif r.status_code == 429:
                logging.error(f"THROTTLED: {filename} (Flickr 429)")
                time.sleep(60)
                return "failed_429"
        except Exception as e:
            pass

    return "failed_not_found"

def main():
    if not os.path.exists(DOWNLOAD_FOLDER): os.makedirs(DOWNLOAD_FOLDER)
    
    print(f"Reading {os.path.join(DATA_PATH, 'info.json.gz')}...")
    with gzip.open(os.path.join(DATA_PATH, 'info.json.gz'), 'rb') as f:
        data = json.load(f)

    print(f"Queueing {len(data)} images...")
    print("Strategy: AWS S3 -> Google V4 Mirror -> Flickr")
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=1)
    session.mount('https://', adapter)

    stats = {}
    
    with futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures_map = {executor.submit(download_one_image, session, d, DOWNLOAD_FOLDER): d for d in data}
        
        pbar = tqdm(futures.as_completed(futures_map), total=len(data))
        for future in pbar:
            result = future.result()
            stats[result] = stats.get(result, 0) + 1
            
            # Update desc
            pbar.set_description(f"AWS:{stats.get('success_s3_aws',0)} GCS:{stats.get('success_gcs_mirror',0)} Flkr:{stats.get('success_flickr',0)}")

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()