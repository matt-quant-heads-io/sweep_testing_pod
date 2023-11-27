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

Once you have the files/dirs above properly installed source the required environment variables with,
```
source $SCRATCH/overlay/sweep_testing_pod/env.sh
```

Finally, cd into the cloned sweep_testing_pod repo.

# Setup / Install
## Install dependencies
```
   conda create -n sweep_test python=3.7 # Create conda environment
   conda activate sweep_test # Activate conda environment
   pip install -r requirements.txt # Install the requirements
```
## Test run (to fix identify the following error msg):

Run the main.py script with:

```
  python main.py --domain zelda --mode controllable --username ms12010 --debug
```

You will see the following error:

```
  Traceback (most recent call last):
  File "main.py", line 43, in <module>
    main(args.domain, args.mode, args.username, args.debug)
  File "main.py", line 16, in main
    generate_training_data_zelda(domain, mode, username, debug)
  File "/path/to/sweep_testing_pod/gen_training_data_zelda.py", line 1008, in generate_training_data_zelda
    render=False,
  File "/path/to/sweep_testing_pod/gen_training_data_zelda.py", line 553, in generate_controllable_pod_greedy
    new_map_stats_dict = env._prob.get_stats(string_map_for_map_stats)
  File "/path/to/minionda/envs/test_env/lib/python3.7/site-packages/gym/core.py", line 240, in __getattr__
    raise AttributeError(f"accessing private attribute '{name}' is prohibited")
AttributeError: accessing private attribute '_prob' is prohibited
```

To fix the issue comment out the following lines (239,240 from [gym]core.py):

```
  vi "/path/to/minionda/envs/test_env/lib/python3.7/site-packages/gym/core.py"

  237     def __getattr__(self, name):
  238         """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
  239         #comment out this line!: if name.startswith("_"):
  240         #comment out this line!:    raise AttributeError(f"accessing private attribute '{name}' is prohibited")
  241         return getattr(self.env, name)
  242 
  243     @property
  244     def spec(self):
  245         """Returns the environment specification."""
  :wq
  
```


# Running
## Data generation, training, and inference for Zelda (uncomment the mode you want to run in)

```sh run_sweep.sh```
