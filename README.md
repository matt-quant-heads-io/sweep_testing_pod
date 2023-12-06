# sweep_testing_pod

# Pre-setup / Install

The following setup assumes that you have the following file structure on hpc,
```
   /scratch/<your username>/overlay
                              |
                               --  sweep_testing_pod/ (this repo)
                              |
                               --  miniconda3/ # To install follow: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started-amd-nodes?authuser=0
                              |
                               --  Miniconda3-latest-Linux-x86_64.sh # To install follow: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started-amd-nodes?authuser=0
```

Finally, cd into the cloned sweep_testing_pod repo.

# Setup / Install
## Install dependencies
```
   conda create -n sweep_test python=3.7 # Create conda environment
   conda activate sweep_test # Activate conda environment
   pip install -r requirements.txt # Install the requirements
```

# Usage


First, set a bunch of environment variables:
```
export PROJECT_ROOT='./'; export ZELDA_DATA_ROOT={$PROJECT_ROOT}/data/zelda; export ZELDA_GOAL_MAPS_ROOT={$PROJECT_ROOT}/goal_maps/zelda                                              ```
```


To generate training data (with a diffusion-like process that adds noise to a small dataset of target levels), run
```
python main.py experiments=exp_1 mode=controllable domain=zelda process=gen_train_data slurm=False
```

To train, set `process=train`

Running `main.py` will launch a sweep of experiments, either locally and sequentially, or by sending jobs to a SLURM cluster with `submitit`. To launch them locally, set `slurm=False`.

To launch on SLURM, set `slurm=True`, and set `+username={$USER}`

Hyperparameter sweeps are given in `constants.py`.


## Data generation, training, and inference for Zelda (uncomment the mode you want to run in)
You can also run the main.py script with:

```sh run_sweep.sh```

