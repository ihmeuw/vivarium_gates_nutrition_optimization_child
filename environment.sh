#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-h|t|v]"
   echo "options:"
   echo "h     Print this Help."
   echo "v     Verbose mode."
   echo "t     Type of conda environment. Either simulation (default) or artifact."
}

# Define variables
username=$(whoami)
env_type="simulation"
one_week_ago=$(date -d "7 days ago" '+%Y-%m-%d %H:%M:%S')

# Process input options
while getopts ":ht:" option; do
   case $option in
      h) # display help
         Help
         return;;
      t) # Type of conda environment to build
         env_type=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         return;;
   esac
done

# Parse environment name
env_name=$(basename "`pwd`")
env_name+="_$env_type"

# Check if environment exists already
env_exists=$(conda info --envs | grep $env_name | head -n 1)
if [[ $env_exists == '' ]]; then
  env_exists="no"
else
  env_exists="yes"
fi

if [ $env_exists == 'yes' ]; then
  creation_time="$(head -n1 /home/$username/miniconda3/envs/$env_name/conda-meta/history)"
  echo "Existing environment found for $env_name."
  # Check if existing environment is older than a week and remake it if so
  if [[ $one_week_ago > $creation_time ]]; then
    echo "Environment is older than one week old. Deleting and remaking environment...."
    conda remove -n $env_name --all -y
    env_exists="no"
  fi

if [ $env_exists == 'no' ]; then
# This cannot be an else about because we set env_exists to no in that block
  # Create new environment
  echo "Environment $env_name does not exist. Creating new environment $env_name..."
  # Create conda environment
  conda create -n $env_name python=3.11 -y
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
fi

# Activate new environment
conda activate $env_name

# Install requirements via Github for new environment
if [ $env_exists == 'no' ]; then
  # NOTE: update branch name if you update requirements.txt in a branch
  pip install -r https://raw.githubusercontent.com/ihmeuw/vivarium_gates_nutrition_optimization_child/main/$install_file 
  # Editable install of repo
  pip install -e .[dev]
  # Install redis for simulation environments
  if [ $env_type == 'simulation' ]; then
    conda install redis -y
  fi
fi