source ${SCRATCH}/overlay/miniconda3/etc/profile.d/conda.sh

export PATH=/scratch/${USER}/overlay/miniconda3/bin:$PATH
export PROJECT_ROOT="/scratch/${USER}/overlay/sweep_testing_pod"
export ZELDA_DATA_ROOT="${PROJECT_ROOT}/data/zelda"
export ZELDA_GOAL_MAPS_ROOT="${PROJECT_ROOT}/goal_maps/zelda"

alias python=${SCRATCH}/overlay/miniconda3/envs/sweep_testing_pod/bin/python
