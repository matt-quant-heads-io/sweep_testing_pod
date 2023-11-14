import os

import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.utils import np_utils

import os
import re
import glob
import numpy as np
from gym_pcgrl import wrappers
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def int_map_from_onehot(map):
    int_map = []
    for y in map:
        for x in y:
            int_map.append(list(x).index(1))

    return np.array(int_map).reshape((19,19))
        




    """
Helper functions for train, infer, and eval modules.
"""


class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)

def get_action(obs, env, model, action_type=True):
    action = None
    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(a=list(range(len(action_prob))), size=1, p=action_prob)
    else:
        action = np.array([env.action_space.sample()])
    return action

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            print(f"kwargs is {kwargs}")
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
            print(f"kwargs is {kwargs}")
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)



