import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
import gc
import h5py  

# Directory where processed data is stored
PROCESSED_DIR = "..."
RESULTS_DIR = "feature_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_sample(split="train", max_samples=5000):
    """Load a sample of data for feature importance analysis"""
    print(f"Loading sample of {max_samples} examples from {split} set...")
    
    # Instead of zarr, we'll use numpy arrays directly or HDF5 files
    # First try to load data from numpy files if they exist
    X_path = os.path.join(PROCESSED_DIR, f"X_{split}.npy")
    y_path = os.path.join(PROCESSED_DIR, f"y_{split}.npy")
    
    # If numpy files don't exist, try HDF5
    h5_path = os.path.join(PROCESSED_DIR, f"X_{split}.h5")
    
    # Load labels first (should be smaller)
    if os.path.exists(y_path):
        y = np.load(y_path)
        print(f"Loaded labels with shape: {y.shape}")
    else:
        print(f"Error: Could not find labels file at {y_path}")
        return None, None, 0
    
    # Try to load data
    if os.path.exists(X_path):
        # For numpy files
        print(f"Loading from numpy file: {X_path}")
        # Use memory mapping to handle large files
        X_full = np.load(X_path, mmap_mode='r')
    elif os.path.exists(h5_path):
        # For HDF5 files
        print(f"Loading from HDF5 file: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            # Get shape information without loading data
            dset = f['data']
            X_shape = dset.shape
            n_channels = X_shape[-1] if len(X_shape) >= 3 else 1
            
            # Sample a subset of the data
            n_samples = min(max_samples, X_shape[0])
            indices = np.random.choice(X_shape[0], n_samples, replace=False)
            
            # Load in smaller chunks
            X_samples = []
            chunk_size = 500
            for i in tqdm(range(0, len(indices), chunk_size)):
                chunk_indices = indices[i:i+chunk_size]
                X_chunk = dset[chunk_indices]
                X_samples.append(X_chunk)
            
            X_full = np.vstack(X_samples)
    else:
        # Try looking for batch files
        batch_files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith(f"X_batch_") and (f.endswith(".npy") or f.endswith(".h5"))]
        
        if not batch_files:
            print(f"Error: Could not find data files in {PROCESSED_DIR}")
            return None, None, 0
        
        print(f"Loading from batch files: found {len(batch_files)} batch files")
        X_samples = []
        y_samples = []
        
        for batch_file in tqdm(batch_files[:5]):  # Limit to first 5 batches
            batch_path = os.path.join(PROCESSED_DIR, batch_file)
            batch_id = batch_file.split('_')[-1].split('.')[0]
            y_batch_path = os.path.join(PROCESSED_DIR, f"y_batch_{batch_id}.npy")
            
            try:
                if batch_file.endswith(".npy"):
                    X_batch = np.load(batch_path)
                else:
                    with h5py.File(batch_path, 'r') as f:
                        X_batch = f['data'][:]
                
                if os.path.exists(y_batch_path):
                    y_batch = np.load(y_batch_path)
                else:
                    print(f"Warning: No labels found for batch {batch_id}, skipping")
                    continue
                
                # Take a sample from each batch
                batch_samples = min(max_samples // 5, len(X_batch))
                if batch_samples < len(X_batch):
                    batch_indices = np.random.choice(len(X_batch), batch_samples, replace=False)
                    X_batch = X_batch[batch_indices]
                    y_batch = y_batch[batch_indices]
                
                X_samples.append(X_batch)
                y_samples.append(y_batch)
            except Exception as e:
                print(f"Error loading batch {batch_id}: {e}")
        
        if not X_samples:
            print("Error: Could not load any data batches")
            return None, None, 0
        
        X_full = np.vstack(X_samples)
        y = np.concatenate(y_samples)
    
    # Determine number of samples and channels
    n_samples = min(max_samples, X_full.shape[0])
    sample_shape = X_full[0].shape
    n_channels = sample_shape[-1] if len(sample_shape) > 0 else 1
    
    print(f"Sample shape: {sample_shape}, Number of channels: {n_channels}")
    
    # Load a random sample of data
    if n_samples < X_full.shape[0]:
        indices = np.random.choice(X_full.shape[0], n_samples, replace=False)
        X_sample = X_full[indices]
        y_sample = y[indices]
    else:
        X_sample = X_full
        y_sample = y
    
    print(f"Loaded {X_sample.shape[0]} samples with shape {X_sample.shape}")
    return X_sample, y_sample, n_channels

def extract_channel_features(X):
    """Extract statistical features from each channel for analysis"""
    print("Extracting features from channels...")
    n_samples, height, width, n_channels = X.shape
    features = np.zeros((n_samples, n_channels * 5))
    
    for i in range(n_channels):
        channel_data = X[:, :, :, i]
        
        # Calculate channel statistics (5 features per channel)
        features[:, i*5 + 0] = np.nanmean(channel_data, axis=(1, 2))  # Mean
        features[:, i*5 + 1] = np.nanstd(channel_data, axis=(1, 2))   # Standard deviation
        
        # Handle potential memory issues with median and percentiles
        for j in tqdm(range(n_samples), desc=f"Processing channel {i+1}/{n_channels}"):
            sample_data = channel_data[j].flatten()
            sample_data = sample_data[~np.isnan(sample_data)]  # Remove NaN values
            
            if len(sample_data) > 0:
                features[j, i*5 + 2] = np.median(sample_data)  # Median
                features[j, i*5 + 3] = np.percentile(sample_data, 25)  # 25th percentile
                features[j, i*5 + 4] = np.percentile(sample_data, 75)  # 75th percentile
    
    # Replace any remaining NaN values
    features = np.nan_to_num(features)
    return features

def calculate_feature_importance(X_features, y):
    """Calculate feature importance using multiple methods"""
    print("Calculating feature importance...")
    
    # Method 1: Random Forest Feature Importance
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_features, y)
    rf_importance = rf_model.feature_importances_
    
    # Method 2: Mutual Information (information gain)
    mi_importance = mutual_info_classif(X_features, y, random_state=42)
    
    # Method 3: AUC for each feature
    auc_importance = []
    for i in range(X_features.shape[1]):
        try:
            auc = roc_auc_score(y, X_features[:, i])
            # Adjust AUC if it's below 0.5 (worse than random)
            auc = max(auc, 1 - auc)
            auc_importance.append(auc)
        except:
            auc_importance.append(0.5)  # Default to random performance
    
    auc_importance = np.array(auc_importance)
    
    return rf_importance, mi_importance, auc_importance

def aggregate_channel_importance(feature_importances, n_channels):
    """Aggregate importance scores across all features for each channel"""
    print("Aggregating channel importance...")
    
    rf_importance, mi_importance, auc_importance = feature_importances
    
    # Each channel has 5 features
    channel_importance = {
        'random_forest': np.zeros(n_channels),
        'mutual_info': np.zeros(n_channels),
        'auc': np.zeros(n_channels)
    }
    
    for i in range(n_channels):
        feature_indices = slice(i*5, (i+1)*5)
        channel_importance['random_forest'][i] = np.mean(rf_importance[feature_indices])
        channel_importance['mutual_info'][i] = np.mean(mi_importance[feature_indices])
        channel_importance['auc'][i] = np.mean(auc_importance[feature_indices])
    
    # Create combined score (normalized)
    combined_importance = (
        channel_importance['random_forest'] / max(np.max(channel_importance['random_forest']), 1e-10) +
        channel_importance['mutual_info'] / max(np.max(channel_importance['mutual_info']), 1e-10) +
        (channel_importance['auc'] - 0.5) / max(np.max(np.maximum(channel_importance['auc'] - 0.5, 0.001)), 1e-10)
    ) / 3
    
    channel_importance['combined'] = combined_importance
    
    return channel_importance

def visualize_channel_importance(channel_importance, metadata_path=None):
    """Create visualizations of channel importance"""
    print("Creating visualizations...")
    
    # Try to load channel names from metadata if available
    channel_names = [f"Channel {i+1}" for i in range(len(channel_importance['combined']))]
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if isinstance(metadata, list) and len(metadata) > 0:
                    # Check if metadata has channel information
                    if 'channels' in metadata[0]:
                        channel_names = metadata[0]['channels']
        except:
            pass
    
    # Create DataFrame for plotting
    n_channels = len(channel_importance['combined'])
    df = pd.DataFrame({
        'Channel': channel_names,
        'Channel_Index': list(range(n_channels)),
        'Random Forest': channel_importance['random_forest'],
        'Mutual Information': channel_importance['mutual_info'],
        'AUC Score': channel_importance['auc'] - 0.5,  # Adjust AUC to start from 0
        'Combined Score': channel_importance['combined']
    })
    
    # Sort by combined importance
    df = df.sort_values('Combined Score', ascending=False)
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Combined score - bar plot
    plt.subplot(2, 1, 1)
    bars = plt.bar(df['Channel'], df['Combined Score'], color='cornflowerblue')
    plt.title('Combined Channel Importance for Cloud Detection', fontsize=16)
    plt.xlabel('MODIS Channel', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Highlight top 3 channels
    top3_indices = df.iloc[:3]['Channel_Index'].values
    for i, bar in enumerate(bars[:3]):
        bar.set_color('darkred')
    
    # Individual metrics - horizontal bar chart
    plt.subplot(2, 1, 2)
    metrics = ['Random Forest', 'Mutual Information', 'AUC Score']
    
    top_channels = df.iloc[:10]['Channel'].values  # Focus on top 10 channels
    top_indices = df.iloc[:10]['Channel_Index'].values
    
    for i, metric in enumerate(metrics):
        plt.barh([c + i*0.25 for c in range(len(top_channels))], 
                df.iloc[:10][metric].values, 
                height=0.2, 
                label=metric,
                alpha=0.7)
    
    plt.yticks([i + 0.25 for i in range(len(top_channels))], top_channels)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance by Different Metrics (Top 10 Channels)', fontsize=16)
    plt.legend(loc='best')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'channel_importance.png'), dpi=300)
    
    # Save top channels information
    top_channels_df = df.iloc[:3][['Channel', 'Channel_Index', 'Combined Score']]
    top_channels_df.to_csv(os.path.join(RESULTS_DIR, 'top_three_channels.csv'), index=False)
    
    print(f"Top 3 channels by importance: {df.iloc[:3]['Channel'].values}")
    print(f"Their indices: {df.iloc[:3]['Channel_Index'].values}")
    
    return top_channels_df

def analyze_channel_correlations(X, top_channels_df):
    """Analyze correlations between top channels and create visualizations"""
    print("Analyzing channel correlations...")
    
    # Get indices of top 3 channels
    top_indices = top_channels_df['Channel_Index'].values
    
    # Extract data for these channels
    n_samples = min(1000, X.shape[0])  # Limit samples for memory
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    
    # Flatten spatial dimensions and extract top channels
    X_flat = X[sample_indices].reshape(n_samples, -1, X.shape[-1])[:, :, top_indices]
    
    # Create correlation matrix
    corr_matrix = np.corrcoef([X_flat[:, :, i].flatten() for i in range(3)])
    
    # Create heatmap of correlations
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    channel_names = top_channels_df['Channel'].values
    plt.xticks(np.arange(3), channel_names, rotation=45)
    plt.yticks(np.arange(3), channel_names)
    
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                     ha='center', va='center', 
                     color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    plt.title('Correlation Between Top 3 Channels')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'channel_correlations.png'), dpi=300)
    
    # Create scatterplot matrix for the top 3 channels
    plt.figure(figsize=(12, 10))
    
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i*3 + j + 1)
            
            if i == j:
                # Histogram on diagonal
                plt.hist(X_flat[:, ::100, i].flatten(), bins=50, alpha=0.7)
                plt.title(f'{channel_names[i]} Distribution')
            else:
                # Scatterplot on off-diagonal
                plt.scatter(X_flat[:, ::500, j].flatten(), X_flat[:, ::500, i].flatten(), 
                           alpha=0.1, s=1)
                plt.xlabel(channel_names[j])
                plt.ylabel(channel_names[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'channel_scatterplot_matrix.png'), dpi=300)

def load_directly_from_modis_files(max_files=600, max_samples=5000):
    """Alternative data loading directly from MODIS HDF files"""
    print(f"Trying to load data directly from MODIS files...")
    
    from pyhdf.SD import SD, SDC
    
    # Find original MODIS files
    data_dir = "modis_data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found")
        return None, None, 0
    
    hdf_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf")]
    if not hdf_files:
        print(f"Error: No HDF files found in {data_dir}")
        return None, None, 0
    
    print(f"Found {len(hdf_files)} HDF4 files. Processing up to {max_files} files.")
    
    all_data = []
    all_labels = []
    n_channels = 0
    
    for idx, file_name in enumerate(tqdm(hdf_files[:max_files])):
        try:
            file_path = os.path.join(data_dir, file_name)
            hdf = SD(file_path, SDC.READ)
            datasets = hdf.datasets()
            
            # Check for Cloud_Mask
            if "Cloud_Mask" not in datasets:
                print(f"Warning: Skipping {file_name} - No 'Cloud_Mask' dataset found.")
                continue
            
            # Extract data and labels
            data = hdf.select("Cloud_Mask")[:]
            data = np.array(data, dtype=np.float32)
            
            if data.ndim == 3:
                data = np.moveaxis(data, 0, -1)
            elif data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            elif data.ndim > 3:
                data = data[:, :, :min(data.shape[2], 10)]
            
            cloud_mask_byte = data[:, :, 0].astype(np.uint8)
            cloud_mask_flags = (cloud_mask_byte & 0b00000110) >> 1
            
            # Split into patches
            patch_size = 128
            stride = 64
            h, w = data.shape[:2]
            
            patches = []
            labels = []
            
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = data[i:i + patch_size, j:j + patch_size, :]
                    patch_label_mask = cloud_mask_flags[i:i + patch_size, j:j + patch_size]
                    
                    if np.all(patch == 0) or np.all(np.isnan(patch)):
                        continue
                    
                    counts = np.bincount(patch_label_mask.flatten(), minlength=4)
                    majority_label = np.argmax(counts)
                    
                    if counts[majority_label] / np.sum(counts) < 0.7:
                        continue
                    
                    binary_label = 0 if majority_label < 2 else 1
                    
                    # Normalize patch
                    normalized_patch = np.zeros_like(patch, dtype=np.float32)
                    for c in range(patch.shape[-1]):
                        band = patch[:, :, c].astype(np.float32)
                        valid_pixels = band[~np.isnan(band) & (band != 0)]
                        if len(valid_pixels) > 0:
                            p1, p99 = np.percentile(valid_pixels, (1, 99))
                            band = np.clip(band, p1, p99)
                            band = (band - p1) / (p99 - p1 + 1e-8)
                        normalized_patch[:, :, c] = band
                    
                    patches.append(normalized_patch)
                    labels.append(binary_label)
                    
                    if len(patches) >= max_samples // max_files:
                        break
                if len(patches) >= max_samples // max_files:
                    break
            
            if patches:
                all_data.extend(patches)
                all_labels.extend(labels)
                n_channels = patches[0].shape[-1]
                print(f"Extracted {len(patches)} patches from {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    if not all_data:
        print("Error: Could not extract any data from MODIS files")
        return None, None, 0
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"Loaded total of {len(X)} samples with {n_channels} channels")
    return X, y, n_channels

def main():
    print("Starting feature importance analysis...")
    
    # Try to load processed data
    X, y, n_channels = load_data_sample(split="train", max_samples=5000)
    
    # If loading processed data fails, try loading directly from MODIS files
    if X is None:
        print("Could not load processed data. Trying to load directly from MODIS files...")
        X, y, n_channels = load_directly_from_modis_files(max_files=600, max_samples=5000)
    
    if X is None or n_channels == 0:
        print("ERROR: Failed to load data. Please check your data files.")
        return
    
    # Extract features from channels
    X_features = extract_channel_features(X)
    
    # Calculate feature importance
    feature_importances = calculate_feature_importance(X_features, y)
    
    # Aggregate importance per channel
    channel_importance = aggregate_channel_importance(feature_importances, n_channels)
    
    # Visualize results
    try:
        import json
        metadata_path = os.path.join(PROCESSED_DIR, "train_metadata.json")
        top_channels_df = visualize_channel_importance(channel_importance, metadata_path)
    except ImportError:
        top_channels_df = visualize_channel_importance(channel_importance, None)
    
    # Analyze correlations between top channels
    analyze_channel_correlations(X, top_channels_df)
    
    # Save importance scores
    np.savez(os.path.join(RESULTS_DIR, 'channel_importance_scores.npz'), 
             random_forest=channel_importance['random_forest'],
             mutual_info=channel_importance['mutual_info'],
             auc=channel_importance['auc'],
             combined=channel_importance['combined'])
    
    print("\nFeature importance analysis complete!")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
