import os
import numpy as np
import dask.array as da
from sklearn.model_selection import train_test_split
from pyhdf.SD import SD, SDC
import json
import gc
import zarr
from tqdm import tqdm

# Directories
DATA_DIR = "modis_data"
SAVE_DIR = "processed_data_optimized"
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory '{DATA_DIR}' does not exist")
    exit(1)

hdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".hdf")]
if len(hdf_files) == 0:
    print(f"ERROR: No .hdf files found in {DATA_DIR}")
    exit(1)

print(f"Found {len(hdf_files)} HDF4 files in {DATA_DIR}")

def extract_data_and_label(file_path):
    try:
        hdf = SD(file_path, SDC.READ)
        datasets = hdf.datasets()
        print(f"\nProcessing {file_path} | Datasets: {list(datasets.keys())}")

        if "Cloud_Mask" not in datasets:
            print(f"WARNING: Skipping {file_path} - No 'Cloud_Mask' dataset found.")
            return None, None

        data = hdf.select("Cloud_Mask")[:]
        data = np.array(data, dtype=np.float32)

        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        elif data.ndim > 3:
            print(f"WARNING: Unexpected data dimensions {data.ndim} in {file_path}")
            data = data[:, :, :min(data.shape[2], 10)]

        cloud_mask_byte = data[:, :, 0].astype(np.uint8)
        cloud_mask_flags = (cloud_mask_byte & 0b00000110) >> 1

        return data, cloud_mask_flags
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return None, None

# Extract metadata from HDF4
def extract_metadata(file_path):
    try:
        hdf = SD(file_path, SDC.READ)
        metadata = {}
        
        for attr_name, attr_value in hdf.attributes().items():
            if attr_name not in ['CoreMetadata.0', 'StructMetadata.0', 'archiveMetadata.0']:
                metadata[attr_name] = str(attr_value)  
        
        metadata['file_id'] = os.path.basename(file_path)
        file_date = os.path.basename(file_path).split('_')[1] if '_' in os.path.basename(file_path) else ''
        metadata['date'] = file_date

        try:
            if 'Latitude' in hdf.datasets() and 'Longitude' in hdf.datasets():

                lat_dataset = hdf.select('Latitude')
                lon_dataset = hdf.select('Longitude')
                lat_data = lat_dataset[:]
                lon_data = lon_dataset[:]

                metadata['lat_min'] = float(np.min(lat_data))
                metadata['lat_max'] = float(np.max(lat_data))
                metadata['lon_min'] = float(np.min(lon_data))
                metadata['lon_max'] = float(np.max(lon_data))

                metadata['lat_resolution'] = float((metadata['lat_max'] - metadata['lat_min']) / lat_data.shape[0])
                metadata['lon_resolution'] = float((metadata['lon_max'] - metadata['lon_min']) / lon_data.shape[1])
            else:
                pass
        except Exception as e:
            print(f"Warning: Could not extract geolocation data: {e}")
            metadata['lat_min'] = 0.0
            metadata['lat_max'] = 0.0
            metadata['lon_min'] = 0.0
            metadata['lon_max'] = 0.0
            metadata['lat_resolution'] = 0.0
            metadata['lon_resolution'] = 0.0
            
        return metadata
    except Exception as e:
        print(f"ERROR extracting metadata from {file_path}: {e}")
        return {'file_id': os.path.basename(file_path)}

# Normalize data to [0,1] range
def normalize_data(data):
    """Normalize input data to [0,1] range using robust scaling"""
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    for i in range(data.shape[-1]):
        band = data[:, :, i].astype(np.float32)

        valid_pixels = band[~np.isnan(band) & (band != 0)]
        if len(valid_pixels) > 0:
            p1, p99 = np.percentile(valid_pixels, (1, 99))
            band = np.clip(band, p1, p99)
            band = (band - p1) / (p99 - p1 + 1e-8)
        normalized_data[:, :, i] = band
        
    return normalized_data

# Split into patches with location tracking
def split_into_patches_and_labels(data, label_mask, metadata, patch_size=128, stride=64):
    h, w = data.shape[:2]
    c = data.shape[2] if data.ndim > 2 else 1
    patches = []
    labels = []
    patch_metadata = []
    
    normalized_data = normalize_data(data)
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = normalized_data[i:i + patch_size, j:j + patch_size, :]
            patch_label_mask = label_mask[i:i + patch_size, j:j + patch_size]
            
            if np.all(patch == 0) or np.all(np.isnan(patch)):
                continue

            counts = np.bincount(patch_label_mask.flatten(), minlength=4)
            majority_label = np.argmax(counts)
            
            if counts[majority_label] / np.sum(counts) < 0.7:
                continue
                
            binary_label = 0 if majority_label < 2 else 1
            
            patch_meta = metadata.copy()
            if all(k in metadata for k in ['lat_min', 'lat_resolution', 'lon_min', 'lon_resolution']):
                patch_meta['center_lat'] = metadata['lat_min'] + (i + patch_size/2) * metadata['lat_resolution']
                patch_meta['center_lon'] = metadata['lon_min'] + (j + patch_size/2) * metadata['lon_resolution']
            
            patch_meta['patch_i'] = i
            patch_meta['patch_j'] = j
            patch_meta['binary_label'] = binary_label
            patch_meta['detailed_label'] = int(majority_label)
            patch_meta['location_id'] = f"{metadata['file_id']}_{i}_{j}"
            
            patches.append(patch)
            labels.append(binary_label)
            patch_metadata.append(patch_meta)
    
    return patches, labels, patch_metadata

def create_zarr_store(path, shape, chunks, dtype=np.float32):
    return zarr.open(
        path,
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=dtype
    )

# Process files in smaller batches
BATCH_SIZE = 20  
MAX_SAMPLES_PER_BATCH = 1000  
X_all = []
y_all = []
metadata_all = []
file_counter = 0
total_samples = 0
labels_by_batch = []  

for idx, file_name in enumerate(tqdm(hdf_files, desc="Processing HDF files")):
    file_path = os.path.join(DATA_DIR, file_name)
    
    data, label_mask = extract_data_and_label(file_path)
    if data is None or label_mask is None:
        continue
        
    metadata = extract_metadata(file_path)
    
    patches, labels, patch_metadata = split_into_patches_and_labels(data, label_mask, metadata)

    del data, label_mask
    gc.collect()
    
    if not patches:
        print(f"No valid patches extracted from {file_name}")
        continue

    if len(patches) > MAX_SAMPLES_PER_BATCH:
        print(f"Too many patches ({len(patches)}) from {file_name}, sampling {MAX_SAMPLES_PER_BATCH}")
        indices = np.random.choice(len(patches), MAX_SAMPLES_PER_BATCH, replace=False)
        patches = [patches[i] for i in indices]
        labels = [labels[i] for i in indices]
        patch_metadata = [patch_metadata[i] for i in indices]
        
    X_all.extend(patches)
    y_all.extend(labels)
    metadata_all.extend(patch_metadata)
    
    # Save batch when we reach BATCH_SIZE or at the end
    if len(X_all) >= BATCH_SIZE or idx == len(hdf_files) - 1:
        if X_all:
            print(f"Saving batch {file_counter} with {len(X_all)} patches...")
            
            X_batch = da.from_array(np.array(X_all), chunks=(1, 128, 128, -1))
            y_batch = da.from_array(np.array(y_all), chunks=(100,))
            
            X_batch.to_zarr(os.path.join(SAVE_DIR, f"X_batch_{file_counter}.zarr"), overwrite=True)
            
            np.save(os.path.join(SAVE_DIR, f"y_batch_{file_counter}.npy"), y_batch.compute())
            
            batch_info = {
                "batch_id": file_counter,
                "size": len(X_all),
                "label_counts": {
                    "0": int(np.sum(np.array(y_all) == 0)),
                    "1": int(np.sum(np.array(y_all) == 1))
                }
            }
            
            with open(os.path.join(SAVE_DIR, f"metadata_batch_{file_counter}.json"), 'w') as f:
                json.dump(metadata_all, f)
                
            with open(os.path.join(SAVE_DIR, f"info_batch_{file_counter}.json"), 'w') as f:
                json.dump(batch_info, f)
            
            labels_by_batch.append(np.array(y_all))
            
            total_samples += len(X_all)
            X_all = []
            y_all = []
            metadata_all = []
            file_counter += 1
            
            gc.collect()

print("\nPreprocessing complete!")
print(f"Total samples processed: {total_samples}")

# Create data split index mapping using batch information
print("Creating data splits...")

all_batch_counts = []
all_y = []
batch_indices = []
sample_indices = []

for i in range(file_counter):
    try:
        y_batch = np.load(os.path.join(SAVE_DIR, f"y_batch_{i}.npy"))
        batch_size = len(y_batch)
        all_batch_counts.append(batch_size)
        all_y.append(y_batch)
        
        batch_indices.extend([i] * batch_size)
        sample_indices.extend(range(batch_size))
        
        del y_batch
    except Exception as e:
        print(f"Error loading batch {i}: {e}")
        continue
    
print(f"Total samples to split: {sum(all_batch_counts)}")

#Combine all labels for stratification
all_y_combined = np.concatenate(all_y)
all_indices = np.arange(len(all_y_combined))

train_idx, temp_idx = train_test_split(
    all_indices, test_size=0.3, stratify=all_y_combined, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=all_y_combined[temp_idx], random_state=42)

train_mappings = [(batch_indices[i], sample_indices[i]) for i in train_idx]
val_mappings = [(batch_indices[i], sample_indices[i]) for i in val_idx]
test_mappings = [(batch_indices[i], sample_indices[i]) for i in test_idx]

del all_y, all_y_combined, all_indices, batch_indices, sample_indices
gc.collect()

print(f"Split created: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples")

with open(os.path.join(SAVE_DIR, "train_batch_indices.json"), 'w') as f:
    json.dump(train_mappings, f)
with open(os.path.join(SAVE_DIR, "val_batch_indices.json"), 'w') as f:
    json.dump(val_mappings, f)
with open(os.path.join(SAVE_DIR, "test_batch_indices.json"), 'w') as f:
    json.dump(test_mappings, f)

# Define a highly optimized function to process splits in small chunks and save directly to zarr
def save_split_to_zarr(split_name, mappings):
    print(f"Processing {split_name} split...")
    
    batch_to_indices = {}
    for batch_idx, sample_idx in mappings:
        if batch_idx not in batch_to_indices:
            batch_to_indices[batch_idx] = []
        batch_to_indices[batch_idx].append(sample_idx)
    
    batches = sorted(batch_to_indices.keys())
    print(f"Total batches in {split_name}: {len(batches)}")
    
    total_size = sum(len(batch_to_indices[batch_idx]) for batch_idx in batches)
    print(f"Total samples in {split_name}: {total_size}")
    
    sample_shape = None
    for batch_idx in batches:
        if batch_to_indices[batch_idx]:  
            try:
                X_batch = da.from_zarr(os.path.join(SAVE_DIR, f"X_batch_{batch_idx}.zarr"))
                if X_batch.shape[0] > 0:
                    sample_shape = X_batch[0].shape
                    break
            except Exception as e:
                print(f"Error reading batch {batch_idx}: {e}")
                continue
    
    if not sample_shape or total_size == 0:
        print(f"Warning: No samples found for {split_name} split")
        return
        
    print(f"Sample shape for {split_name}: {sample_shape}")
    
    zarr_path = os.path.join(SAVE_DIR, f"X_{split_name}.zarr")
    X_final = zarr.open(
        zarr_path, 
        mode='w',
        shape=(total_size, *sample_shape),
        chunks=(100, *sample_shape),  
        dtype=np.float32
    )
    
    y_final = np.zeros(total_size, dtype=np.int32)
 
    metadata_split = []
    
    chunk_size = 2  
    current_idx = 0  
    

    for chunk_start in tqdm(range(0, len(batches), chunk_size), 
                            desc=f"Building {split_name} dataset", 
                            total=(len(batches) + chunk_size - 1)//chunk_size):
        chunk_batches = batches[chunk_start:chunk_start+chunk_size]
        
        for batch_idx in chunk_batches:
            # Skip if no samples needed from this batch
            if batch_idx not in batch_to_indices or not batch_to_indices[batch_idx]:
                continue
                
            sample_indices = batch_to_indices[batch_idx]
            
            try:
                
                X_batch = da.from_zarr(os.path.join(SAVE_DIR, f"X_batch_{batch_idx}.zarr"))
                y_batch = np.load(os.path.join(SAVE_DIR, f"y_batch_{batch_idx}.npy"))
                
                with open(os.path.join(SAVE_DIR, f"metadata_batch_{batch_idx}.json"), 'r') as f:
                    metadata_batch = json.load(f)

                n_samples = len(sample_indices)
                
                if n_samples > 0:
   
                    mini_chunk_size = 50  
                    
                    for mini_start in range(0, n_samples, mini_chunk_size):
                        mini_end = min(mini_start + mini_chunk_size, n_samples)
                        mini_indices = sample_indices[mini_start:mini_end]
                        mini_size = len(mini_indices)
                        
                        
                        X_mini = X_batch[mini_indices].compute()  
                        X_final[current_idx:current_idx+mini_size] = X_mini
                        
                        y_final[current_idx:current_idx+mini_size] = y_batch[mini_indices]

                        for i in mini_indices:
                            if i < len(metadata_batch):
                                metadata_split.append(metadata_batch[i])

                        current_idx += mini_size

                        del X_mini
                        gc.collect()

                del X_batch, y_batch, metadata_batch
                gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Save labels and metadata
    print(f"Saving {split_name} labels and metadata...")
    np.save(os.path.join(SAVE_DIR, f"y_{split_name}.npy"), y_final)
    
    with open(os.path.join(SAVE_DIR, f"{split_name}_metadata.json"), 'w') as f:
        json.dump(metadata_split, f)
    
    print(f" Saved {split_name} split with {current_idx} samples")
    return

# Process each split
save_split_to_zarr("train", train_mappings)
save_split_to_zarr("val", val_mappings)
save_split_to_zarr("test", test_mappings)

print(f" All data saved to {SAVE_DIR}")
