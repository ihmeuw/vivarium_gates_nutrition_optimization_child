#!/bin/bash

# This script creates a conda environment from a requirements.txt file from Github model repository.
# This sript should be run from the home directory of the repo. The script takes two arguments:
# The first argument is the environment name
# The second argument is the environment type. This will either be "simulation" or "artifact" e.g. 
# what the user will be using the environment to develop.

# Run this script by: source creator.sh <ENVIRONMENT_NAME> <ENVIRONMENT_TYPE>

env_name=$1
env_type=$2

# Determine which requirements.txt to install from
if [ $env_type == 'simulation' ]; then
  echo "Installing requirements for simulation environment..."
  install_file="requirements.txt"
elif [ $env_type == 'artifact' ]; then
  echo "Installing requirements for artifact environment"
  install_file="artifact_requirements.txt"
else
  echo "Invalid environment type. Valid argument types are simulation and artifact."
  return 
fi


# Create conda environment
conda create -n $env_name python=3.11 -y

# Activate new environment
conda activate $env_name

# Install requirements via Github
# NOTE: update branch name if you update requirements.txt in a branch
pip install -r https://raw.githubusercontent.com/ihmeuw/vivarium_gates_nutrition_optimization_child/main/$install_file 

# Editable install of repo
pip install -e .[dev]

# Install redis for simulation environments
if [ $env_type == 'simulation' ]; then
  conda install redis -y
fi  
