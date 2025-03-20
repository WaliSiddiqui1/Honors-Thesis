import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyhdf.SD import SD, SDC  # For HDF4 files (MODIS uses HDF4)

# Directory where MODIS NumPy arrays are stored
DATA_DIR = "modis_data"

# Function to load and visualize HDF files as images
def visualize_hdf_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found, skipping: {file_path}")
        return
    
    try:
        # Try to open as HDF5 first
        try:
            with h5py.File(file_path, "r") as hdf:
                # Print available datasets for debugging
                print(f"Available datasets in {os.path.basename(file_path)}:")
                hdf.visit(lambda name: print(f" - {name}"))
                
                # Ask user which dataset to visualize
                print("\nChoose a dataset to visualize (enter the full path):")
                dataset_path = input("Dataset path: ").strip()
                
                if dataset_path not in hdf:
                    print(f"⚠️ Dataset '{dataset_path}' not found.")
                    return
                
                data = np.array(hdf[dataset_path])
                
        except OSError:
            # If HDF5 fails, try HDF4 (most MODIS files are HDF4)
            hdf = SD(file_path, SDC.READ)
            
            # Print available datasets
            datasets = hdf.datasets()
            print(f"Available datasets in {os.path.basename(file_path)}:")
            for idx, name in enumerate(datasets.keys()):
                print(f"{idx} - {name}")
            
            # Ask user which dataset to visualize
            print("\nChoose a dataset to visualize (enter the number):")
            dataset_idx = int(input("Dataset number: ").strip())
            
            dataset_name = list(datasets.keys())[dataset_idx]
            data = hdf.select(dataset_name)[:]
        
        # Get data shape and handle multidimensional arrays
        print(f"Dataset shape: {data.shape}")
        
        if len(data.shape) > 2:
            print(f"This is a {len(data.shape)}-dimensional dataset.")
            print("Which dimension/layer would you like to visualize?")
            
            if len(data.shape) == 3:
                layer = int(input(f"Enter layer index (0-{data.shape[0]-1}): "))
                data = data[layer]
            else:
                print("Complex dataset - using first 2D slice")
                data = data[0]  # Take first slice for higher dimensions
        
        # Normalize the data for better visualization
        if np.max(data) - np.min(data) == 0:
            print(f"⚠️ Data is empty or uniform, skipping.")
            return
        
        # Handle NaN and Inf values
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

        plt.figure(figsize=(10, 8))
        plt.imshow(data_norm, cmap='viridis')
        plt.colorbar(label="Normalized Value")
        plt.title(f"Visualization of {os.path.basename(file_path)}")
        plt.show()

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")

# Function to list all HDF files and let user select one
def select_hdf_file():
    hdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".hdf", ".h5", ".he5"))]
    
    if not hdf_files:
        print("No HDF files found in the directory.")
        return None
    
    print("Available HDF files:")
    for i, file_name in enumerate(hdf_files):
        print(f"{i} - {file_name}")
    
    try:
        selection = int(input("Select a file by number: "))
        if 0 <= selection < len(hdf_files):
            return os.path.join(DATA_DIR, hdf_files[selection])
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Please enter a valid number.")
        return None

# Main execution
if __name__ == "__main__":
    while True:
        file_path = select_hdf_file()
        if file_path:
            visualize_hdf_file(file_path)
        
        if input("View another file? (y/n): ").lower() != 'y':
            break