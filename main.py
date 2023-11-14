import argparse
from itertools import product
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from gen_training_data_zelda import generate_training_data_zelda
from gen_training_data_lr import generate_training_data_lr
from gen_training_data_lego import generate_training_data_lego

from train_zelda import train_zelda
# from .train import train_model
# from .inference import infer

from inference_zelda import inference_zelda



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig):
    params_combos_list = list(product(cfg.obs_sizes, cfg.experiments.goal_set_sizes, cfg.trajectory_lengths, cfg.training_dataset_sizes))

    if cfg.domain == "zelda":
        for combo_id, sweep_params in zip(cfg.experiments.combo_ids,params_combos_list):
            generate_training_data_zelda(combo_id, sweep_params, cfg.mode, cfg.username)
            trajectories_to_cleanup = train_zelda(combo_id, sweep_params, cfg.mode, cfg.username)
            for traj_path in trajectories_to_cleanup:
                os.remove(traj_path)
                print(f"Removed {traj_path}")

        for combo_id, sweep_params in zip(cfg.experiments.combo_ids,params_combos_list):
            inference_zelda(combo_id, sweep_params, cfg.domain, cfg.mode, cfg.username, debug)

        # TODO: inference_zelda here
    elif domain == "lr": 
        # TODO: generate_training_data_lr(combo_id, sweep_params, cfg.mode, cfg.username)
        # TODO: train_lr here
        # TODO: inference_lr here
        pass
    elif domain == "lego": 
        # TODO: generate_training_data_lego(domain, mode)
        # TODO: train_lego here
        # TODO: inference_lego here
        pass


if __name__ == '__main__':
    main()