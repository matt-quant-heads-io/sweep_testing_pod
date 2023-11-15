import argparse
from itertools import product
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import submitit

from gen_training_data_zelda import generate_training_data_zelda
from gen_training_data_lr import generate_training_data_lr
from gen_training_data_lego import generate_training_data_lego

from train_zelda import train_zelda
# from .train import train_model
# from .inference import infer

from inference_zelda import inference_zelda


def main(combo_id, sweep_params, domain, mode, username):
    if domain == "zelda":
        # for combo_id, sweep_params in combo_ids_params_combos:
        generate_training_data_zelda(combo_id, sweep_params, mode, username)
        trajectories_to_cleanup = train_zelda(combo_id, sweep_params, mode, username)
        for traj_path in trajectories_to_cleanup:
            os.remove(traj_path)
            print(f"Removed {traj_path}")

        # for combo_id, sweep_params in combo_ids_params_combos:
        inference_zelda(combo_id, sweep_params, mode, username)

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def get_sweep_params(cfg : DictConfig):
    params_combos_list = list(product(cfg.obs_sizes, cfg.experiments.goal_set_sizes, cfg.trajectory_lengths, cfg.training_dataset_sizes))
    params_list_lens = len(params_combos_list)
    combo_ids = cfg.experiments.combo_ids
    domain = cfg.domain #[cfg.domain]*params_list_lens
    mode = cfg.mode #[cfg.mode]*params_list_lens
    username = cfg.username #[cfg.username]*params_list_lens

    log_folder = "logs/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(slurm_array_parallelism=10, gpus_per_node=4, cpus_per_task=48, mem_gb=50)
    
    
    job = executor.submit(main, combo_ids, sweep_params, domain, mode, username)
    print(f"job: {job}")
    # assert sum(job.done() for job in jobs) == len(sweep_params)





if __name__ == '__main__':
    get_sweep_params()

