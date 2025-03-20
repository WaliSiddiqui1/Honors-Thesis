import os
import requests
import h5py
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

# Your Earthdata token
EARTHDATA_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6IndhbGlzIiwiZXhwIjoxNzQ3MDI1Mjk5LCJpYXQiOjE3NDE4NDEyOTksImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.LE9IgYXXeTXohGZCNoSJ2VYqWWem8cgT7I-GmdgyEuHfa4RRZowlxXSeC3oVIKUKCSDyoEo2uvJN8-ge8HdR1q05yh7xZIie1V55MerRmeslw2j9bgXrrH5DZtQpP_iINfDFd226PtMtjI4J75N_KOuEZ2IcfVPKF6tKXFOAroePRItEY_bZr5q5X6LDiEnxo8mD4E3-2iLnPmL9-k954Pv1oIJsFKMwr7B62XvsJXlwEhI4F8fC_OM6ZEJEHGb9_zBhB_tHqnmWZgD8b_KkJhO5CjUDO2orMNQhCqxCB6E_JpCBmnbSyeyol0nop4Qy3ttpmWOHjri3LY6s-C-2EA"

# CMR API endpoint
CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Only use products we know exist
MODIS_PRODUCTS = {
    "MOD35_L2": "MOD35_L2",  # Cloud Mask
    "MOD021KM": "MOD021KM",  # Calibrated Radiances
}

# Date range for study
START_DATE = datetime(2023, 3, 1)
END_DATE = datetime(2023, 3, 3)

# Output directory
DOWNLOAD_DIR = "modis_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Headers for NASA API authentication
HEADERS = {
    "Authorization": f"Bearer {EARTHDATA_TOKEN}",
    "User-Agent": "Python/3.x MODIS-Downloader/1.0",
}

# Function to get MODIS file URLs - FIXED based on API error
def get_modis_files(product, start_date, end_date):
    date = start_date
    file_urls = []
    
    print(f"\nSearching for {product} files from {start_date.date()} to {end_date.date()}...")
    
    while date <= end_date:
        date_str = date.strftime("%Y-%m-%dT00:00:00Z")
        next_date = date + timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%dT00:00:00Z")
        
        # FIXED: Removed 'format' parameter which was causing errors
        params = {
            "short_name": product,
            "temporal": f"{date_str}/{next_date_str}",
            "page_size": 100,
            "provider": "LAADS",
            "bounding_box": "-180,-90,180,90"
        }

        print(f"  Querying for {product} on {date.date()}...")
        response = requests.get(CMR_GRANULE_SEARCH_URL, params=params, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"  ❌ CMR Query Failed: Status {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
            date += timedelta(days=1)
            continue
        
        granules = response.json().get("feed", {}).get("entry", [])
        print(f"  Found {len(granules)} granules")
        
        for granule in granules:
            granule_id = granule.get('id', 'Unknown')
            print(f"  Processing granule: {granule_id}")
            
            # Extract the download URL
            found_url = False
            for link in granule.get("links", []):
                href = link.get("href", "")
                if href.endswith(".hdf"):
                    print(f"  Found link: {href}")
                    
                    # For data in the LAADS archive, we need the direct download URL
                    if "opendap" not in href.lower() and "s3" not in href.lower():
                        file_urls.append(href)
                        found_url = True
                        print(f"  ✅ Added download URL: {href}")
                        break
            
            if not found_url:
                # If no direct download link is found, try to construct one
                granule_name = granule.get("title", "")
                if granule_name.endswith(".hdf"):
                    # Try to construct a direct download URL
                    # This is a common pattern for LAADS data
                    constructed_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{granule_name}"
                    print(f"  🔧 Constructed URL: {constructed_url}")
                    file_urls.append(constructed_url)
        
        date += timedelta(days=1)
    
    return file_urls

# Function to download a single file
def download_file(file_url):
    file_path = os.path.join(DOWNLOAD_DIR, os.path.basename(file_url))
    if os.path.exists(file_path):
        print(f"Skipping {file_path}, already downloaded.")
        return file_path
    
    print(f"Downloading {file_url}...")
    
    try:
        # Try direct download first
        response = requests.get(file_url, headers=HEADERS, stream=True, timeout=60)
        
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {file_path} ✅")
            return file_path
        else:
            print(f"ERROR: Failed to download {file_url} | Status: {response.status_code}")
            
            # If direct download fails, try using the EarthData Login redirect
            # This handles cases where authentication is required
            print("Trying alternate download method with EarthData Login...")
            session = requests.Session()
            response = session.get(file_url)
            
            if response.status_code == 302:  # Redirect
                redirect_url = response.headers['Location']
                print(f"Following redirect to: {redirect_url}")
                response = session.get(redirect_url, headers=HEADERS, stream=True)
                
                if response.status_code == 200:
                    with open(file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f"Downloaded: {file_path} ✅")
                    return file_path
            
            print("All download attempts failed.")
            return None
            
    except Exception as e:
        print(f"ERROR: Exception while downloading {file_url}: {e}")
        return None

# Process the HDF files
def process_hdf_file(file_path):
    """Process an HDF file and extract a dataset for analysis."""
    try:
        print(f"Processing {file_path}...")
        with h5py.File(file_path, 'r') as hdf:
            # Print all datasets for debugging
            print(f"Available datasets in {file_path}:")
            
            def print_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  - {name} (shape: {obj.shape}, dtype: {obj.dtype})")
            
            hdf.visititems(print_datasets)
            
            # Try to find a dataset with cloud mask or radiance
            for key in hdf.keys():
                if isinstance(hdf[key], h5py.Dataset):
                    data = np.array(hdf[key])
                    print(f"Extracted dataset '{key}' with shape {data.shape}")
                    return data
                
                # If it's a group, look for datasets inside
                if isinstance(hdf[key], h5py.Group):
                    for subkey in hdf[key].keys():
                        if isinstance(hdf[key][subkey], h5py.Dataset):
                            data = np.array(hdf[key][subkey])
                            print(f"Extracted dataset '{key}/{subkey}' with shape {data.shape}")
                            return data
            
            print(f"No suitable dataset found in {file_path}")
            return None
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Starting MODIS data download...")
    
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
    
    print(f"\nDownload summary:")
    print(f"- Total files found: {len(all_file_urls)}")
    print(f"- Successfully downloaded: {len(downloaded_files)}")
    
    # Process downloaded files
    if downloaded_files:
        print("\nProcessing downloaded files...")
        processed_data = []
        
        for file_path in downloaded_files:
            data = process_hdf_file(file_path)
            if data is not None:
                processed_data.append(data)
        
        # Check if we have any processed data
        if processed_data:
            print(f"\nSuccessfully processed {len(processed_data)} files.")
            
            # Convert to TensorFlow tensors
            for i, data in enumerate(processed_data):
                tensor = tf.convert_to_tensor(data, dtype=tf.float32)
                print(f"Dataset {i+1}: Tensor shape {tensor.shape}")
                
            print("\nData processing complete!")
        else:
            print("\nNo data was successfully processed.")
    else:
        print("\nNo files were downloaded, so no data to process.")

