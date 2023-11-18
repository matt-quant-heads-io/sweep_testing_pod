from itertools import product
import os

import hydra
from omegaconf import DictConfig
import submitit

from gen_training_data_zelda import generate_training_data_zelda
from train_zelda import train_zelda
from inference_zelda import inference_zelda
import constants

RUN_PLAYBOOK = {"zelda": [generate_training_data_zelda, train_zelda, inference_zelda]}


def main(combo_ids, sweep_params, domain, mode):
    if domain not in RUN_PLAYBOOK.keys():
        raise KeyError(f"domain must be one of {list(RUN_PLAYBOOK.keys())}")

    gen_data_func, train_func, infer_func = RUN_PLAYBOOK[domain]
    print(f'{constants.DOMAIN_VARS_ZELDA["trajectories_to_generate"]}')

    # Gen training data
    obs_sz, goal_sz, traj_len, td_sz = [
        param_vals_list[1]
        for param_vals_list in constants.DOMAIN_VARS_ZELDA["trajectories_to_generate"]
    ]
    trajectories_to_generate = product(obs_sz, goal_sz, traj_len, td_sz)

    print(f"trajectories_to_generate: {trajectories_to_generate}")
    for trajectory_params in trajectories_to_generate:
        print(f"trajectory_params: {trajectory_params}")
        gen_data_func(trajectory_params, mode)

    # Traing models
    # obs_sz, goal_sz, traj_len, td_sz = [
    #     param_vals_list[1]
    #     for param_vals_list in constants.DOMAIN_VARS_ZELDA["sweep_params"]
    # ]
    # models_to_train = product(obs_sz, goal_sz, traj_len, td_sz)
    # for combo_id, params_combo in enumerate(models_to_train):
    #     train_func(combo_id, params_combo, mode)

    # # Infer
    # for combo_id, params_combo in enumerate(models_to_train):
    #     infer_func(combo_id, params_combo, mode)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    # cfg.experiments.exp_name
    sweep_params = list(
        product(
            cfg.obs_sizes,
            cfg.experiments.goal_set_sizes,
            cfg.trajectory_lengths,
            cfg.training_dataset_sizes,
        )
    )
    combo_ids = cfg.experiments.combo_ids
    domain = cfg.domain
    mode = cfg.mode

    log_folder = f"logs/{cfg.experiments.exp_name}"
    executor = submitit.AutoExecutor(folder=log_folder)
    # executor.update_parameters(
    #     gpus_per_node=1, cpus_per_task=1, mem_gb=50, timeout_min=1440
    # )

    executor.update_parameters(
        slurm_array_parallelism=1,
        gpus_per_node=4,
        cpus_per_task=48,
        mem_gb=50,
        timeout_min=1440,
    )

    job = executor.submit(main, combo_ids, sweep_params, domain, mode)
    # print(f"job: {job}")
    # assert sum(job.done() for job in jobs) == len(sweep_params)
    # TODO: Talk to HPC about fastest hardware combination


if __name__ == "__main__":
    run()
