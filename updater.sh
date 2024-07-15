# Script to update vivarium packages to versions specified in requirements.txt files
# Script takes one argument - environment type (env_type) which is either "artifact"
# or "simulation". If users run into environment issues, this is a quick way to check 
# all vivarium framework repositories are up to date.

# Run script with command: source updater.sh <ENV_TYPE>

env_type=$1
packages_to_update="vivarium vivarium_public_health vivarium_cluster_tools"

# Determine which requirements.txt to install from
if [ $env_type == 'simulation' ]; then
  echo "updating requirements for simulation environment..."
  install_file="requirements.txt"
elif [ $env_type == 'artifact' ]; then
  echo "Updating requirements for artifact environment"
  packages_to_update+=" gbd_mapping vivarium_inputs vivarium_gbd_access"
  install_file="artifact_requirements.txt"
else
  echo "Invalid environment type. Valid argument types are simulation (default) and artifact."
  sleep 
fi

# Update packages
echo "Updating packages: $packages_to_update..."
pip install $packages_to_update --constraint $install_file