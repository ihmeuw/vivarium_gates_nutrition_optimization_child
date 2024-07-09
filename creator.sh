#!/bin/bash

# This script creates a conda environment from a requirements.txt file from Github.
# Conda environments should be created with the simulation science service user and 
# the credientials can be found at /mnt/team/simulation_science/priv/credentials/svc-simsci

# Three arguments must be provided to the script:
#   - The first argument is the name of the conda environment
#   - The second argument is the repo name
#   - The third argument is the name of the branch to fetch requirements.txt from for
#       repo specified in argument two.

env_name=$1
repo_name=$2
branch_name=$3

# Create conda environment
conda create -p -y /mnt/team/simulation_science/pub/envs/$env_name python=3.11

# Activate new environment
conda activate /mnt/team/simulation_science/pub/envs/$env_name

# Install requirements via Github
pip install -r https://raw.githubusercontent.com/ihmeuw/$repo_name/$branch_name/requirements.txt 

# Install redis for sims
conda install redis -y
    