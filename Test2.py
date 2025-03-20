import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyhdf.SD import SD, SDC

# Open the HDF4 file
file_path = "modis_data/MOD35_L2.A2023060.0320.061.2023131210321.hdf"
hdf = SD(file_path, SDC.READ)

# Extract Latitude, Longitude, and Cloud Mask
lat = hdf.select('Latitude')[:]
lon = hdf.select('Longitude')[:]
cloud_mask = hdf.select('Cloud_Mask')[:]

# Extract first bit of the first layer (Cloudy = 1, Clear = 0)
cloud_binary = (np.bitwise_and(cloud_mask[0], 1) > 0).astype(int)

# Step 1: First resize cloud_binary to match some base dimensions we can work with
# Let's choose dimensions that will work with pcolormesh
target_rows, target_cols = 2029, 1349  # One smaller than our target coordinate dimensions
cloud_binary_resized = cloud_binary[:target_rows, :target_cols]

# Step 2: Now create coordinate arrays that are ONE LARGER than the data array
target_lat_rows, target_lat_cols = target_rows + 1, target_cols + 1  # 2030, 1350
lat_resampled = np.linspace(np.min(lat), np.max(lat), target_lat_rows)
lon_resampled = np.linspace(np.min(lon), np.max(lon), target_lat_cols)

# Create 2D meshgrid for pcolormesh
lon_grid, lat_grid = np.meshgrid(lon_resampled, lat_resampled)

# Print shapes to verify
print(f"Cloud binary resized shape: {cloud_binary_resized.shape}")  # Should be (2029, 1349)
print(f"Coordinate grid shapes: {lon_grid.shape}")  # Should be (2030, 1350)

# Create the map
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")

# Use the correctly sized arrays with pcolormesh
sc = ax.pcolormesh(lon_grid, lat_grid, cloud_binary_resized, cmap='gray', 
                  transform=ccrs.PlateCarree())

# Colorbar
plt.colorbar(sc, label="Cloud Mask (1=Cloudy, 0=Clear)")

# Title
plt.title("MODIS Cloud Mask (MOD35_L2)")

plt.show()