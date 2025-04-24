# Honors Thesis: Cloud Removal in MODIS Satellite Imagery

This repository implements a complete deep learning pipeline to enhance MODIS satellite images by detecting and removing clouds, and is specifically aimed at Arctic sea ice analysis. The pipeline includes data download, preprocessing, classification, and image restoration using a GAN (Generative adversarial networks).

---

# Environment Setup

**Clone the repository**:
git clone <your-repo-url>

Set up a Python environment (Python 3.9 or newer recommended):

conda create -n modis-env python=3.9

conda activate modis-env

**Install required packages**:

pip install -r requirements.txt

**Earthdata Credentials**:

Register at https://urs.earthdata.nasa.gov/

Generate a Bearer Token

Replace the placeholder EARTHDATA_TOKEN = "..." in Preprocessing.py with your token to enable downloading data from NASA

**Folder Structure**:

Honors-Thesis-main/
├── modis_data/                # Raw HDF satellite data
├── processed_data_optimized/  # Preprocessed numpy or zarr files
├── cloud_data1/               # Classified image patches and metadata
│   ├── clear/
│   ├── cloudy/
│   ├── metadata/
│   └── generated1/            # GAN-restored output
├── *.py                       # Python scripts
├── *.sh                       # SLURM job scripts (for GPU clusters)
└── requirements.txt

# Pipeline Overview:

**1. Download and Preprocess MODIS Data**

python Preprocessing.py

Downloads MOD021KM and MOD35_L2 datasets using NASA's CMR API

Output: modis_data/*.hdf

**2. Extract Features & Cloud Masks**

python CNN.py

Processes HDF files to extract cloud mask flags and features

Saves datasets as numpy/zarr arrays

Output: processed_data_optimized/

**3. Train ResNet-based Cloud Classifier**

python ResnetCNN.py

Classifies patches as “cloudy” or “clear” using a custom ResNet

Here, one must specify the spectral bands being used or just use the amount available in the given data (see file for more detail)

Saves image metadata and split images

Output: cloud_data1/ folders + metadata

**4. Match Cloudy/Clear Patch Pairs**

python Res2.py

Matches images by geolocation and time

Outputs a JSON file of paired cloudy and clear patches

Output: paired_data_optimized.json

**5. Cloud Removal using GAN**
   
python Cloudgen.py

Loads matched image pairs and trains a U-Net-style generator

Outputs restored clear-sky patches

Output: cloud_data1/generated1/

**6. Optional: Feature Importance Analysis**

python Features.py

Uses mutual information, random forest, and AUC to rank features

Output: feature_analysis/ (if directory is created)

# Full Pipeline

**Run the following in order**:

python Preprocessing.py

python CNN.py

python ResnetCNN.py

python Res2.py

python Cloudgen.py

Adjust max_pairs, batch sizes, and file limits based on available RAM.

GPU-based .sh scripts are included for high-performance training on SLURM clusters and will likely be needed for most scripts

To run these, once connected to an environment where you can tap into using a GPU, simply run "sbatch ____.sh"

Training may take several hours depending on system resources.
