#!/bin/bash

# This script creates a conda environment from a requirements.txt file from Github.
# Conda environments should be created with the simulation science service user and 
# the credientials can be found at /mnt/team/simulation_science/priv/credentials/svc-simsci

# Two arguments are provided to the script:
#   - The first argument is the name of the conda environment
#   - The second argument is the url to the requirements.txt on Github. This should be of 
#       the format https://raw.githubusercontent.com/ihmeuw/<REPO_NAME>/<BRANCH_NAME>/requirements.txt 

# Create conda environment
conda create -p /mnt/team/simulation_science/pub/envs/$1 python=3.11

# Install requirements via Github
pip install -r $2
