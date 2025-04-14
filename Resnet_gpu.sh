#!/bin/bash
#SBATCH -p gpu                  # Use the GPU partition
#SBATCH --gres=gpu:1             # Request one GPU
#SBATCH -n 1                     # Use one core
#SBATCH -t 08:00:00              # Time limit for the job (8 hours)
#SBATCH --mem=100G               # Memory limit (100GB)
#SBATCH -J cloud_classifier      # Job name
#SBATCH -o logs/Res_classifier.out  # Standard output log file
#SBATCH -e logs/Res_classifier.err  # Standard error log file
#SBATCH --mail-type=END,FAIL     # Send email on job completion/failure
#SBATCH --mail-user=wali_siddiqui@brown.edu  # Your email address

# Clear any loaded modules and load required ones
module purge
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

# Create a virtual environment if it doesn't exist
if [ ! -d "$HOME/modis-env" ]; then
  echo "Creating virtual environment..."
  python -m venv $HOME/modis-env
fi

# Activate the virtual environment
source $HOME/modis-env/bin/activate

# Install required packages
pip list | grep -q tensorflow || pip install tensorflow
pip list | grep -q dask || pip install "dask[array]"
pip list | grep -q numpy || pip install numpy
pip list | grep -q scikit-learn || pip install scikit-learn
pip list | grep -q zarr || pip install zarr
pip list | grep -q matplotlib || pip install matplotlib
pip list | grep -q pandas || pip install pandas
pip list | grep -q tqdm || pip install tqdm
pip list | grep -q pyhdf || pip install pyhdf

# Create logs directory if it doesn't exist
mkdir -p logs

# Find the actual CUDA path based on the loaded module
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "Found CUDA at: $CUDA_PATH"

# Set environment variables for TensorFlow GPU usage
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Print information about the GPU
echo "CUDA Devices:"
nvidia-smi

# Create the output directories if they don't exist
mkdir -p processed_data_optimized
mkdir -p cloud_data1

# Print Python environment information
python -c "import numpy; print('NumPy version:', numpy.__version__); import dask; print('Dask version:', dask.__version__); import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Run the cloud classification script
echo "Starting cloud classification and model training..."
python -u Resnet1.py
