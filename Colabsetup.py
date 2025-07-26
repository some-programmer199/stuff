# LC0 build with XLA backend + PJRT plugin setup on Colab (for TPU use)

# Step 1: Install dependencies
!sudo apt update
!sudo apt install -y cmake ninja-build clang libomp-dev protobuf-compiler libprotobuf-dev
!pip install numpy onnx jax jaxlib

# Step 2: Clone and build LC0 with XLA backend
!git clone https://github.com/LeelaChessZero/lc0.git
%cd lc0
!./build.sh -Dxla

# Step 3: Locate PJRT plugin and copy it to LC0 build folder
import os
import shutil
import glob

matches = glob.glob("/usr/local/lib/python*/dist-packages/jaxlib/pjrt_c_api_*.so")
assert matches, "PJRT plugin not found"
pjrt_path = matches[0]
dest_path = "/content/lc0/build/release/pjrt_c_api_plugin.so"
shutil.copy(pjrt_path, dest_path)
print(f"Copied PJRT plugin to: {dest_path}")

# Step 4: Zip LC0 build folder for download
%cd /content/lc0/build/release
!zip -r lc0_xla_tpu.zip *

# Step 5: Download the zip archive
from google.colab import files
files.download('/content/lc0/build/release/lc0_xla_tpu.zip')
