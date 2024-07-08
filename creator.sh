#!/bin/bash

# This script creates a conda environment from a requirements.txt file from Github.
# Conda environments should be created with the simulation science service user and 
# the credientials can be found at /mnt/team/simulation_science/priv/credentials/svc-simsci

# Three arguments must be provided to the script:
#   - The first argument is the name of the conda environment
#   - The second argument is the repo name
#   - The third argument is the name of the branch to fetch requirements.txt from for
#       repo specified in argument two.

# Create conda environment
conda create -p /mnt/team/simulation_science/pub/envs/$1 python=3.11

# Activate new environment
conda activate /mnt/team/simulation_science/pub/envs/$1

# Install requirements via Github
pip install -r https://raw.githubusercontent.com/ihmeuw/$2/$3/requirements.txt 

# Install redis for sims
conda install redis
    