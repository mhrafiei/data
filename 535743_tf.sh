#!/bin/bash

# Create a new conda environment with Python 3.10
conda create -n 535743_tf Python=3.10 -y

# Activate the conda environment
source activate 535743_tf

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with CUDA support
python -m pip install tensorflow[and-cuda]

# Install the IPython kernel package
conda install -y ipykernel

# Add the conda environment to Jupyter as a new kernel
python -m ipykernel install --user --name 535743_tf --display-name "Python (535743_tf)"

# Install additional Python packages
pip install pandas
pip install matplotlib 
pip install tqdm 
pip install opencv-python

# Add conda environment activation to .bashrc
echo "source activate 535743_tf" >> ~/.bashrc

echo "Environment setup complete. The environment will be activated automatically upon login."
