import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Directory where preprocessed NumPy data is stored
DATA_DIR = "modis_data"

# First, check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory '{DATA_DIR}' does not exist")
    exit(1)

# List all files in directory to verify content
print(f"Files in {DATA_DIR}:")
npy_count = 0
for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        npy_count += 1
    print(f" - {file}")

if npy_count == 0:
    print(f"ERROR: No .npy files found in {DATA_DIR}")
    exit(1)

# Load preprocessed NumPy arrays with better error handling
def load_data():
    numpy_data = []
    labels = []
    
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".npy"):
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                data = np.load(file_path, allow_pickle=True)
                
                print(f"Loaded {file_name}, shape: {data.shape}, dtype: {data.dtype}")
                
                # Check if the array has enough dimensions for slicing
                if len(data.shape) < 3:
                    print(f"WARNING: {file_name} has fewer than 3 dimensions, skipping")
                    continue
                
                # Extract features and labels - adjust this based on your actual data structure
                try:
                    feature_data = data[:, :, :-1]  # All bands except last one
                    label_data = data[:, :, -1]     # Last band is the cloud mask
                    
                    print(f"  - Feature shape: {feature_data.shape}")
                    print(f"  - Label shape: {label_data.shape}")
                    
                    numpy_data.append(feature_data)
                    labels.append(label_data)
                except IndexError as e:
                    print(f"WARNING: Could not slice {file_name}: {e}")
                    continue
                    
            except Exception as e:
                print(f"ERROR loading {file_name}: {e}")
                continue
    
    if not numpy_data or not labels:
        print("No valid data was loaded. Check your files and slicing logic.")
        return np.array([]), np.array([])
    
    # Check if all arrays have the same shape before stacking
    feature_shapes = set(arr.shape for arr in numpy_data)
    label_shapes = set(arr.shape for arr in labels)
    
    print(f"Feature shapes found: {feature_shapes}")
    print(f"Label shapes found: {label_shapes}")
    
    if len(feature_shapes) > 1:
        print("WARNING: Features have inconsistent shapes, attempting to standardize...")
        # You might need custom logic here to handle different shapes
    
    try:
        numpy_data = np.array(numpy_data)
        labels = np.array(labels)
        print(f"Final data shapes - X: {numpy_data.shape}, y: {labels.shape}")
        return numpy_data, labels
    except ValueError as e:
        print(f"ERROR: Could not combine arrays: {e}")
        return np.array([]), np.array([])

# Load dataset
print("Loading data...")
X, y = load_data()
print("Loaded data shape:", X.shape, "Labels shape:", y.shape)

# Check if we have data before proceeding
if X.size == 0 or y.size == 0:
    print("ERROR: No data was loaded. Fix data loading issues before continuing.")
    exit(1)

# Normalize input data
X = X / np.max(X)  # More robust normalization
print("Data normalized. Range:", np.min(X), "to", np.max(X))

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Save processed datasets
try:
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    print("Dataset saved successfully!")
except Exception as e:
    print(f"ERROR saving datasets: {e}")