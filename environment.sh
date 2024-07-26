#!/bin/bash
############################################################
# Help: 
# Run script by: source environment.sh                                         #
############################################################
Help()
{ 
  # Reset OPTIND so help can be invoked multiple times per shell session.
  OPTIND=1
   # Display Help
   echo "Script to automatically create and validate conda environments."
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
branch_name=$(git rev-parse --abbrev-ref HEAD)
# Determine which requirements.txt to install from
if [ $env_type == 'simulation' ]; then
  install_file="requirements.txt"
elif [ $env_type == 'artifact' ]; then
  install_file="artifact_requirements.txt"
else
  echo "Invalid environment type. Valid argument types are simulation and artifact."
  return 
fi
requirements_modification_time="$(date -r $install_file '+%Y-%m-%d %H:%M:%S')"

# Pull repo to get latest changes from remote if remote exists
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
if [[ $exit_code == '0' ]]; then
  git fetch --all
  echo "Git branch '$branch_name' exists in the remote repository"
  git pull -u origin $branch_name
fi

# Check if environment exists already
create_env=$(conda info --envs | grep $env_name | head -n 1)
if [[ $create_env == '' ]]; then
  # No environment exists with this name
  create_env="yes"
else
  create_env="no"
fi

if [ $create_env == 'no' ]; then
  # Check if existing environment needs to be updated
  echo "Existing environment found for $env_name."
  creation_time="$(head -n1 /home/$username/miniconda3/envs/$env_name/conda-meta/history)"
  # Check if existing environment is older than a week and delete it if so
  if [[ $one_week_ago > $creation_time ]]; then
    echo "Environment is older than one week old. Deleting and remaking environment...."
    conda remove -n $env_name --all -y
    create_env="yes"
  fi
fi

if [ $create_env == 'yes' ]; then
  # Create new environment
  echo "Environment $env_name does not exist. Creating new environment $env_name..."
  # Create conda environment
  conda create -n $env_name python=3.11 -y
fi

# Activate environment
conda activate $env_name
if [ $create_env == 'yes' ]; then
  # Install json parser
  conda install jq -y
fi
# At this point we have either a new environment or an environment that is less than a week old.
# We need to check if we need to reinstall packages
update_env=$create_env

if [ $update_env == 'no' ]; then
  # Check if environment was build before last modification to requirements file
  if [[ $creation_time < $requirements_modification_time ]]; then
    echo "Requirements file last modified after environment was created. Reinstalling packages..."
    update_env="yes"
  fi
  # Check if there has been an update to vivarium packages since last modification to requirements file
  grep @ $install_file 
  # TODO: can we make this grep output a variable?
  while read -r line ; do
    # Parse each line of grep output
    repo_info=(${line//@/ })
    repo=${repo_info[0]}
    repo_branch=${repo_info[2]}
    last_commit_time=$(curl -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/ihmeuw/$repo/commits?sha=$repo_branch | jq '.["0"].commit.committer.date')
    if [[ $requirements_modification_time < $last_commit_time ]]; then
      update_env="yes"
      break
    fi
  # This way of writing/exiting while loop is a here string so the process runs
  # in the main shell and not a subshell: https://www.gnu.org/software/bash/manual/bashref.html#Here-Strings
  # I put an arbitrary empty string here but this is so we can set create_env to yes if we hit that trigger
  done <<< "$(echo "")"
fi

# Install requirements via Github
if [ $update_env == 'yes' ]; then
  # NOTE: update branch name if you update requirements.txt in a branch
  echo "Installing packages for $env_type environment"
  pip install -r https://raw.githubusercontent.com/ihmeuw/vivarium_gates_nutrition_optimization_child/main/$install_file  
  # Editable install of repo
  pip install -e .[dev]
  # Install redis for simulation environments
  if [ $env_type == 'simulation' ]; then
    conda install redis -y
  fi
fi