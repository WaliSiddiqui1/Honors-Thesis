import os
import requests
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC  # PyHDF for HDF4 support

# Earthdata token --> one of the ways to directly download data from MODIS. Needs to be obtained throught Nasa's website
EARTHDATA_TOKEN = "..."

# CMR API endpoint
CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# MODIS Products we want to use
MODIS_PRODUCTS = {
    "MOD35_L2": "MOD35_L2",  # Cloud Mask
    "MOD021KM": "MOD021KM",  # Calibrated Radiances
}

# Date range for study
START_DATE = datetime(2023, 3, 1) # Example start date
END_DATE = datetime(2023, 3, x) # any date works

# Output directory
DOWNLOAD_DIR = "modis_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Headers for NASA API authentication
HEADERS = {
    "Authorization": f"Bearer {EARTHDATA_TOKEN}",
    "User-Agent": "Python/3.x MODIS-Downloader/1.0",
}

### --- Function to Query MODIS Files ---
def get_modis_files(product, start_date, end_date):
    """Query NASA CMR API to get MODIS file URLs."""
    date = start_date
    file_urls = []
    
    print(f"\n Searching for {product} files from {start_date.date()} to {end_date.date()}...")

    while date <= end_date:
        date_str = date.strftime("%Y-%m-%dT00:00:00Z")
        next_date = date + timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%dT00:00:00Z")

        params = {
            "short_name": product,
            "temporal": f"{date_str}/{next_date_str}",
            "page_size": 100,
            "provider": "LAADS",
            "bounding_box": "-180,-90,180,90"
        }

        print(f" Querying for {product} on {date.date()}...")
        response = requests.get(CMR_GRANULE_SEARCH_URL, params=params, headers=HEADERS)

        if response.status_code != 200:
            print(f" CMR Query Failed: Status {response.status_code}")
            print(f" Response: {response.text[:200]}...")
            date += timedelta(days=1)
            continue

        granules = response.json().get("feed", {}).get("entry", [])
        print(f" Found {len(granules)} granules.")

        for granule in granules:
            granule_id = granule.get('id', 'Unknown')
            print(f" Processing granule: {granule_id}")

            found_url = False
            for link in granule.get("links", []):
                href = link.get("href", "")
                if href.endswith(".hdf"):
                    print(f" Found link: {href}")
                    if "opendap" not in href.lower() and "s3" not in href.lower():
                        file_urls.append(href)
                        found_url = True
                        break
            
            if not found_url:
                # Construct fallback URL
                granule_name = granule.get("title", "")
                if granule_name.endswith(".hdf"):
                    constructed_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{granule_name}"
                    print(f" Constructed URL: {constructed_url}")
                    file_urls.append(constructed_url)

        date += timedelta(days=1)

    return file_urls

### --- Function to Download MODIS Files ---
def download_file(file_url):
    """Download a MODIS HDF4 file."""
    file_path = os.path.join(DOWNLOAD_DIR, os.path.basename(file_url))
    if os.path.exists(file_path):
        print(f" Skipping {file_path}, already downloaded.")
        return file_path

    print(f" Downloading {file_url}...")

    try:
        response = requests.get(file_url, headers=HEADERS, stream=True, timeout=60)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f" Downloaded: {file_path}")
            return file_path
        else:
            print(f" ERROR: Failed to download {file_url} | Status: {response.status_code}")
            return None

    except Exception as e:
        print(f" ERROR: Exception while downloading {file_url}: {e}")
        return None

### --- Function to Process HDF4 MODIS Files ---
def process_hdf_file(file_path):
    """Process an HDF4 file and extract a dataset for analysis."""
    try:
        print(f" Processing {file_path}...")

        # Open HDF4 file
        hdf = SD(file_path, SDC.READ)
        
        # Print all available datasets
        datasets = hdf.datasets()
        print(f"📂 Available datasets in {file_path}:")
        for name, info in datasets.items():
            print(f"  - {name} (shape: {info[0]}, dtype: {info[3]})")

        # Try to extract a dataset (Cloud Mask / Radiance)
        for dataset_name in datasets.keys():
            if "Cloud_Mask" in dataset_name or "Radiance" in dataset_name:
                data = hdf.select(dataset_name)[:]  # Read as NumPy array
                print(f" Extracted dataset '{dataset_name}' with shape {data.shape}")
                return np.array(data)

        print(f" No suitable dataset found in {file_path}")
        return None

    except Exception as e:
        print(f" ERROR processing {file_path}: {e}")
        return None

### --- Main Execution ---
if __name__ == "__main__":
    print("\n Starting MODIS Data Download...")

    # Get MODIS file URLs
    all_file_urls = []
    for product_name, shortname in MODIS_PRODUCTS.items():
        urls = get_modis_files(shortname, START_DATE, END_DATE)
        all_file_urls.extend(urls)

    # Download files
    downloaded_files = []
    for file_url in all_file_urls:
        file_path = download_file(file_url)
        if file_path:
            downloaded_files.append(file_path)

    print("\n Download Summary:")
    print(f"- Total files found: {len(all_file_urls)}")
    print(f"- Successfully downloaded: {len(downloaded_files)}")

    # Process downloaded files
    if downloaded_files:
        print("\n Processing downloaded files...")
        processed_data = []
        
        for file_path in downloaded_files:
            data = process_hdf_file(file_path)
            if data is not None:
                processed_data.append(data)
        
        if processed_data:
            print(f"\n Successfully processed {len(processed_data)} files.")

            # Convert to TensorFlow tensors
            for i, data in enumerate(processed_data):
                tensor = tf.convert_to_tensor(data, dtype=tf.float32)
                print(f" Dataset {i+1}: Tensor shape {tensor.shape}")

            print("\n Data processing complete!")
        else:
            print("\n No data was successfully processed.")
    else:
        print("\n No files were downloaded, so no data to process.")

