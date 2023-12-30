#!/usr/bin/env bash

# Maintainer: Philip Chung
# Date: 08-30-2023

# The following script installs pyenv and poetry on an Azure Compute Instance.
# Pyenv gives us ability to control python version despite other python versions
# installed on this machine (e.g. system, conda, etc.).  
# Poetry is used for package dependency resolution and virtualenv creation.

### --- Script Defaults & Debug Settings --- ###
# If script fails, exit script
set -o errexit
# If access unset variable, fail script & exit (deactivate since we will access $CONDA_SHLVL)
# set -o nounset
# If pipe command fails, exit script
set -o pipefail
# Trace for debugging
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace;
fi

### --- Environment Variables --- ###
# Azure CloudFiles are mounted & symlinked as a dir within $HOME.  This path is referenced as CLOUDFILES_HOME
# Note: default user in Azure Compute Instance is `azureuser`, not USERNAME.
export USERNAME="chungph"   #fill in username
export PROJECT_NAME="llm-asa-los"
export CLOUDFILES_HOME="/home/azureuser/cloudfiles/code/Users/${USERNAME}"
export PROJECT_PATH="${CLOUDFILES_HOME}/${PROJECT_NAME}"

### --- Deactivate & Uninstall Conda --- ###
# Azure Compute Instances are docker containers that have preconfigured conda environments
# that are used by default.  We need to keep the default `azureml_py38` conda environment
# or else we can't install the VSCode server for remote connection.  However, we remove
# all other conda environments.

# Check if Conda is Installed
if { command -v conda &> /dev/null ; }
then
    # Deactivate all conda environments
    eval "$(conda shell.bash hook)"
    for i in $(seq ${CONDA_SHLVL}); do
        conda deactivate
    done

    # Modify .bashrc to avoid activation of default conda environment
    sudo sed -i '/conda activate azureml_py38/d' ~/.bashrc
    # Ensure we still load default profile, which will load all profile.d
    echo "source /etc/profile ;" >> ~/.bashrc
    # We also need to modify /etc/profile.d/conda.sh
    sudo sed -i '/conda activate jupyter_env/d' /etc/profile.d/conda.sh
    sudo sed -i '/conda activate azureml_py38/d' /etc/profile.d/conda.sh
fi

### --- Pyenv --- ###
# Install pyenv dependencies & additional linux packages
sudo apt update #&& sudo apt upgrade
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
llvm make wget git postgresql libpq-dev -y

# Install pyenv to install & manage different versions of python on machine
curl https://pyenv.run | bash

# Install a specific Python Version with pyenv & Use it Globally.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export GLOBAL_PYTHON_VERSION="3.10.11"
pyenv install ${GLOBAL_PYTHON_VERSION}
pyenv global ${GLOBAL_PYTHON_VERSION}

### --- Poetry --- ###
# Install Poetry 
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry installation to $PATH
export PATH="/home/azureuser/.local/bin:$PATH"

# Go to Project Directory
cd ${PROJECT_PATH}

# Use pyenv to install & specify local python version (only used in this Directory)
export LOCAL_PYTHON_VERSION="3.10.11"
if [[ "${LOCAL_PYTHON_VERSION}" != "${GLOBAL_PYTHON_VERSION}" ]]; then
    pyenv install ${LOCAL_PYTHON_VERSION};
fi
pyenv local ${LOCAL_PYTHON_VERSION}
# Have poetry pick-up local python version
poetry env use python

# Install Python Packages with Poetry
poetry install

# Export environment as standard requirements.txt using pip
poetry run pip freeze > requirements.txt

# Configure .bashrc to load deactivate conda, load pyenv, 
# add poetry to $PATH & go to project directory
echo '
### --- deactivate conda by default --- ###
export CONDA_AUTO_ACTIVATE_BASE=false
### --- deactivate conda by default --- ###

### --- force deactivate all conda environments --- ###
eval "$(conda shell.bash hook)"
for i in $(seq ${CONDA_SHLVL}); do
    conda deactivate
done
### --- force deactivate all conda environments --- ###

### --- pyenv --- ###
# Add pyenv to $PATH
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
### --- pyenv --- ###

### --- poetry custom config --- ###
# Add Poetry Installation to PATH
export PATH="/home/azureuser/.local/bin:$PATH"

# Navigate to project directory
export USERNAME='"${USERNAME}"'
export PROJECT_NAME='"${PROJECT_NAME}"'
export CLOUDFILES_HOME='"${CLOUDFILES_HOME}"'
export PROJECT_PATH='"${PROJECT_PATH}"'
cd ${PROJECT_PATH}
### --- poetry custom config --- ###
' >> ~/.bashrc

# Restart Shell
exec "$SHELL"