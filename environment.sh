#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Script to build conda environment. Run script from the home directory of a model repository."
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
creation_time="$(head -n1 /home/$username/miniconda3/envs/$env_name/conda-meta/history)"
branch_name=$(git rev-parse --abbrev-ref HEAD)
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
requirements_modification_time=$(date -r $install_file '+%Y-%m-%d %H:%M:%S')

# Pull repo to get latest changes from remote if remote exists
git fetch --all
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
if [[ $exit_code == '0' ]]; then
  echo "Git branch '$branch_name' exists in the remote repository"
  git pull -u origin $branch_name
fi

# Get location of conda installation used by default in an interactive shell,
# see https://github.com/conda/conda/issues/7980#issuecomment-472648567
CONDA_BASE=$($SHELL -ic 'conda info --base')
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Check if environment exists already
env_exists=$(conda info --envs | grep $env_name | head -n 1)
if [[ $env_exists == '' ]]; then
  env_exists="no"
else
  env_exists="yes"
fi

if [ $env_exists == 'yes' ]; then
  echo "Existing environment found for $env_name."
  # Check if existing environment is older than a week and remake it if so
  if [[ $one_week_ago > $creation_time ]]; then
    echo "Environment is older than one week old. Deleting and remaking environment...."
    conda remove -n $env_name --all -y
    env_exists="no"
  fi
  # Validate time requirements file was updated
  if [[ $creation_time < $requirements_modification_time ]]; then
    echo "Requirements file last modified after environment was created. Reinstalling packages..."
    env_exists="no"
  fi
fi

if [ $env_exists == 'no' ]; then
  # Create new environment
  echo "Environment $env_name does not exist. Creating new environment $env_name..."
  # Create conda environment
  conda create -n $env_name python=3.11 -y
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