from pyhdf.SD import SD, SDC

file_path = "modis_data/MOD35_L2.A2023060.0320.061.2023131210321.hdf"

# Open the file
hdf = SD(file_path, SDC.READ)

# Print all datasets available
print(hdf.datasets())

# Load a specific dataset (e.g., 'Cloud_Mask')
cloud_mask = hdf.select('Cloud_Mask')[:]
print(cloud_mask.shape)  # Check the dimensions