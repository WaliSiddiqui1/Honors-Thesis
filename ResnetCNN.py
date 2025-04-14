import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import zarr
import dask.array as da

# Directories
PROCESSED_DIR = "processed_data_optimized"
OUTPUT_DIR = "cloud_data1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clear"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cloudy"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metadata"), exist_ok=True)

# Load preprocessed data - handle both numpy and zarr formats
print("Loading preprocessed data...")

# Helper function to load data from either numpy or zarr
def load_array(base_path, name):
    # Try zarr first
    zarr_path = os.path.join(base_path, f"{name}.zarr")
    npy_path = os.path.join(base_path, f"{name}.npy")
    
    if os.path.exists(zarr_path):
        print(f"Loading {name} from zarr format...")
        # Load as dask array first, then compute to numpy for model training
        return da.from_zarr(zarr_path).compute()
    elif os.path.exists(npy_path):
        print(f"Loading {name} from numpy format...")
        return np.load(npy_path)
    else:
        raise FileNotFoundError(f"Could not find {name} data in {base_path}")

# Load data using the helper function
X_train = load_array(PROCESSED_DIR, "X_train")
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))  # Labels are always numpy
X_val = load_array(PROCESSED_DIR, "X_val")
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

# Load metadata
with open(os.path.join(PROCESSED_DIR, "train_metadata.json"), 'r') as f:
    train_metadata = json.load(f)
with open(os.path.join(PROCESSED_DIR, "val_metadata.json"), 'r') as f:
    val_metadata = json.load(f)

print(f"Loaded data: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Class distribution in training set: {np.bincount(y_train)}")

# Prepare data for ResNet
input_shape = (128, 128, 3)

# Check if we need to adjust the channel dimension
if X_train.shape[-1] < 3:
    print(f"Data has only {X_train.shape[-1]} channels, duplicating to create RGB...")
    X_train = np.repeat(X_train, 3 // X_train.shape[-1] + 1, axis=-1)[:, :, :, :3]
    X_val = np.repeat(X_val, 3 // X_val.shape[-1] + 1, axis=-1)[:, :, :, :3]
elif X_train.shape[-1] > 3:
    print(f"Data has {X_train.shape[-1]} channels, selecting first 3 for RGB model...")
    X_train = X_train[:, :, :, :3]
    X_val = X_val[:, :, :, :3]

print(f"Prepared data shape: {X_train.shape}")

# Build ResNet model
print("Building ResNet classifier...")
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False  # Freeze base model

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out = Dense(2, activation='softmax')(x)  # Binary: clear or cloudy

model = Model(inputs=base_model.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

# Memory optimization for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_classifier.h5"),
        save_best_only=True,
        monitor="val_accuracy")
]

# Train model - potentially in batches if data is large
print("Training cloud classifier...")
# For very large datasets, consider using a data generator approach
# For now, assuming the data fits in memory after loading from zarr

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    callbacks=callbacks
)

# Save model
model.save(os.path.join(OUTPUT_DIR, "cloud_classifier_model.h5"))
print("Cloud classifier model saved!")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
print("Training history saved")

# Save classified images with their metadata - process in batches to manage memory
print("Saving classified images with metadata...")

# Create dictionaries to track scenes
scenes = {}

# Determine batch size for processing (to avoid memory issues)
BATCH_SIZE = 500  # Adjust based on your system's memory

# Process training data in batches
total_samples = len(X_train)
for batch_start in range(0, total_samples, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_samples)
    batch_size = batch_end - batch_start
    
    print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1)//BATCH_SIZE}...")
    
    # Extract batch data
    X_batch = X_train[batch_start:batch_end]
    y_batch = y_train[batch_start:batch_end]
    meta_batch = train_metadata[batch_start:batch_end] if isinstance(train_metadata, list) else train_metadata
    
    for i in range(batch_size):
        idx = batch_start + i
        img = X_batch[i]
        label = y_batch[i]
        
        # Get metadata - handle both list and dictionary formats
        if isinstance(meta_batch, list):
            meta = meta_batch[i]
        else:
            meta = meta_batch.get(str(idx), {})
            
        # Create a unique key for this patch based on metadata
        location_id = meta.get('location_id', f"unknown_{idx}")
        category = "clear" if label == 0 else "cloudy"
        
        # Extract scene identifier (using file_id without full patch location)
        scene_id = meta.get('file_id', '').split('_')[0]
        if scene_id not in scenes:
            scenes[scene_id] = {'clear': [], 'cloudy': []}
        
        # Save image
        img_path = os.path.join(OUTPUT_DIR, category, f"{location_id}.npy")
        np.save(img_path, img)
        
        # Save metadata (include the path to the saved image)
        meta['img_path'] = img_path
        meta['predicted_label'] = int(label)
        meta['category'] = category
        meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{location_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        # Track by scene and category
        scenes[scene_id][category].append(location_id)
    
    # Progress update
    print(f"Processed {batch_end}/{total_samples} images")

# Save scene tracking information
with open(os.path.join(OUTPUT_DIR, "scenes.json"), 'w') as f:
    json.dump(scenes, f)

print("✓ Cloud classification complete!")

# Create potential pairing information for GAN training
print("Creating potential pairs for cloud removal training...")

# Find scenes with both clear and cloudy images
valid_scene_pairs = []
for scene_id, categories in scenes.items():
    if categories['clear'] and categories['cloudy']:
        # This scene has both clear and cloudy images
        for cloudy_id in categories['cloudy']:
            cloudy_meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{cloudy_id}.json")
            with open(cloudy_meta_path, 'r') as f:
                cloudy_meta = json.load(f)
                
            # Find the geographically closest clear image from the same scene
            closest_clear = None
            min_distance = float('inf')
            
            for clear_id in categories['clear']:
                clear_meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{clear_id}.json")
                with open(clear_meta_path, 'r') as f:
                    clear_meta = json.load(f)
                
                # Calculate geographic distance if coordinates available
                if ('center_lat' in cloudy_meta and 'center_lon' in cloudy_meta and
                    'center_lat' in clear_meta and 'center_lon' in clear_meta):
                    # Simple Euclidean distance in coordinate space
                    dist = ((cloudy_meta['center_lat'] - clear_meta['center_lat'])**2 + 
                            (cloudy_meta['center_lon'] - clear_meta['center_lon'])**2)**0.5
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_clear = clear_id
                else:
                    # If no coordinates, just pick the first clear image
                    closest_clear = clear_id
                    break
            
            if closest_clear:
                valid_scene_pairs.append({
                    'scene_id': scene_id,
                    'cloudy_id': cloudy_id,
                    'clear_id': closest_clear,
                    'distance': min_distance if min_distance != float('inf') else None
                })

# Save pairing information
with open(os.path.join(OUTPUT_DIR, "paired_data.json"), 'w') as f:
    json.dump(valid_scene_pairs, f)

print(f"Found {len(valid_scene_pairs)} potential paired images for GAN training")
print("Classification and pairing complete!")
