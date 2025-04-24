#!/bin/bash
#SBATCH -p gpu 
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH --mem=32G
#SBATCH -J cloudgen
#SBATCH -o logs/Zcloudgen2.out
#SBATCH -e logs/Zcloudgen2.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=_________________  # Your email address <-- sends you an email when job is finished

# Clear any modules and load required ones
module purge
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/tf-env" ]; then
  echo "Creating virtual environment..."
  python -m venv $HOME/tf-env
fi

# Activate your virtual environment
source $HOME/tf-env/bin/activate

# Install TensorFlow and essential data science packages if not already installed
pip list | grep -q tensorflow || pip install tensorflow==2.12.0
pip list | grep -q numpy || pip install numpy
pip list | grep -q matplotlib || pip install matplotlib
pip list | grep -q scikit-learn || pip install scikit-learn
pip list | grep -q pandas || pip install pandas
pip list | grep -q json || pip install json

# Find the actual CUDA path based on the loaded module
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "Found CUDA at: $CUDA_PATH"

# Set environment variables to fix the libdevice issue
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_PATH"
# Alternative approach: disable XLA JIT compilation
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
# Allow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# BACKUP 1: Search for the libdevice.10.bc file
echo "Searching for libdevice.10.bc..."
find /oscar/runtime/software -name "libdevice.10.bc" 2>/dev/null

# BACKUP 2: If libdevice is found at some path, create a link
LIBDEVICE_PATH=$(find /oscar/runtime/software -name "libdevice.10.bc" 2>/dev/null | head -1)
if [ -n "$LIBDEVICE_PATH" ]; then
  echo "Found libdevice at: $LIBDEVICE_PATH"
  ln -sf "$LIBDEVICE_PATH" ./libdevice.10.bc
else
  echo "Could not find libdevice.10.bc, trying to locate in NVVM directory..."
  
  # Try to find in standard CUDA locations
  if [ -d "$CUDA_PATH/nvvm/libdevice" ]; then
    echo "Found NVVM directory at $CUDA_PATH/nvvm/libdevice"
    if [ -f "$CUDA_PATH/nvvm/libdevice/libdevice.10.bc" ]; then
      ln -sf "$CUDA_PATH/nvvm/libdevice/libdevice.10.bc" ./libdevice.10.bc
      echo "Created symlink to libdevice.10.bc"
    fi
  fi
fi

# BACKUP 3: As a last resort, completely disable XLA
if [ ! -f "./libdevice.10.bc" ]; then
  echo "Could not find libdevice.10.bc, disabling XLA completely"
  export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
fi

# Print information about the GPU
echo "CUDA Devices:"
nvidia-smi

# Verify TensorFlow can see GPUs
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('TensorFlow version:', tf.__version__)"

# Run your script
python -u CloudGen1.py
