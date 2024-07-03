# #!/bin/bash

# Install model repository
pip install git+https://github.com/ihmeuw/vivarium_gates_nutrition_optimization_child.git

# Install at commits for framework repositories
# Vivarium
pip install git+https://github.com/ihmeuw/vivarium.git@release-candidate-spring#egg=vivarium
echo "Finished installing vivarium"

# Vivarium Public Health
pip install git+https://github.com/ihmeuw/vivarium_public_health.git@release-candidate-spring#egg=vivarium_public_health
echo "Finished installing vivarium public health"
