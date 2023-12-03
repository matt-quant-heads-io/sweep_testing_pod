#!/bin/bash

source ${HOME}/sweep_testing_pod/env.sh
conda activate sweep_testing_pod

cd ${PROJECT_ROOT}

# TO run 'gen_train_data' process uncomment below
python /Users/matt/sweep_testing_pod/main.py experiments=exp_1 +mode=controllable +username=ms12010 +domain=zelda +process=gen_train_data


# TO run 'train' process uncomment below
# python main.py experiments=exp_1 +mode=controllable +username=ms12010 +domain=zelda +process=train

# python main.py experiments=exp_1 +mode=controllable +username=ms12010 +domain=zelda +process=inference
