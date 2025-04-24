import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
from tqdm import tqdm

# Directory settings - points to the output from the classification step
OUTPUT_DIR = "cloud_data1"

def compute_distance(cloudy_meta, clear_meta):
    """Compute geographic distance between two points if coordinates available"""
    if ('center_lat' in cloudy_meta and 'center_lon' in cloudy_meta and
        'center_lat' in clear_meta and 'center_lon' in clear_meta):
        # Simple Euclidean distance in coordinate space
        # Note: For production environments, consider using haversine formula for proper geographic distance
        dist = ((cloudy_meta['center_lat'] - clear_meta['center_lat'])**2 + 
                (cloudy_meta['center_lon'] - clear_meta['center_lon'])**2)**0.5
        return dist
    return None

def find_best_pair(args):
    """Find the best clear/cloudy pair for a given scene"""
    scene_id, categories = args
    scene_pairs = []
    
    if not categories['clear'] or not categories['cloudy']:
        return scene_pairs  # Skip scenes without both categories
        
    # Limit the number of cloudy images to process per scene for efficiency
    # This can be adjusted based on your project needs and available computing resources
    cloudy_ids = categories['cloudy'][:y]  # Process at most y cloudy images per scene - 20 is a good number to use
    
    for cloudy_id in cloudy_ids:
        cloudy_meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{cloudy_id}.json")
        try:
            with open(cloudy_meta_path, 'r') as f:
                cloudy_meta = json.load(f)
        except FileNotFoundError:
            continue
                
        # Find the geographically closest clear image from the same scene
        # This maximizes the relevance of the pairing for training
        closest_clear = None
        min_distance = float('inf')
        
        for clear_id in categories['clear']:
            clear_meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{clear_id}.json")
            try:
                with open(clear_meta_path, 'r') as f:
                    clear_meta = json.load(f)
            except FileNotFoundError:
                continue
            
            # Calculate distance when possible to find optimal pairs
            dist = compute_distance(cloudy_meta, clear_meta)
            
            if dist is not None and dist < min_distance:
                min_distance = dist
                closest_clear = clear_id
            elif closest_clear is None:
                # Fallback: if no distance calculation possible
                skip
        
        if closest_clear:
            scene_pairs.append({
                'scene_id': scene_id,
                'cloudy_id': cloudy_id,
                'clear_id': closest_clear,
                'distance': min_distance if min_distance != float('inf') else None
            })
    
    return scene_pairs

def main():
    print("Creating potential pairs for cloud removal training (optimized version)...")
    
    # Load scenes data from the classification step
    scenes_path = os.path.join(OUTPUT_DIR, "scenes.json")
    if not os.path.exists(scenes_path):
        print(f"Error: Scenes file not found at {scenes_path}")
        return
        
    with open(scenes_path, 'r') as f:
        scenes = json.load(f)
    
    print(f"Loaded {len(scenes)} scenes with potential pairs")
    
    # Process in parallel to maximize efficiency
    max_workers = os.cpu_count() or 4  # Automatically use all available CPU cores
    print(f"Using {max_workers} workers for parallel processing")
    
    valid_scene_pairs = []
    
    # Filter for scenes that have both clear and cloudy images
    # This ensures we only process valuable data for pairing
    valid_scenes = [(scene_id, categories) for scene_id, categories in scenes.items() 
                    if categories['clear'] and categories['cloudy']]
    print(f"Found {len(valid_scenes)} scenes with both clear and cloudy images")
    
    # Shuffle to get a more diverse selection if we limit pairs
    # This helps ensure good coverage across different geographic regions
    # In many cases we can just comment this out if we want specificity
    random.shuffle(valid_scenes)
    
    # Process scenes in parallel for improved performance
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(find_best_pair, valid_scenes),
            total=len(valid_scenes),
            desc="Processing scenes"
        ))
    
    # Combine results from all parallel processes
    for scene_pairs in results:
        valid_scene_pairs.extend(scene_pairs)
    
    # Shuffle final pairs to avoid any systematic biases during training
    random.shuffle(valid_scene_pairs)
    
    # Save pairing information for use in the next pipeline stage
    pairs_path = os.path.join(OUTPUT_DIR, "paired_data_optimized1.json")
    with open(pairs_path, 'w') as f:
        json.dump(valid_scene_pairs, f)

    print(f" Created {len(valid_scene_pairs)} paired images")
    print(f"Pairs saved to {pairs_path}")

if __name__ == "__main__":
    main()
