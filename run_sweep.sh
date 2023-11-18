#!/bin/bash

source /scratch/${USER}/overlay/env.sh
conda activate sweep_testing_pod

source /scratch/${USER}/overlay/sweep_testing_pod/env.sh
cd ${PROJECT_ROOT}

python main.py experiments=exp_1 +mode=controllable +username=ms12010 +domain=zelda