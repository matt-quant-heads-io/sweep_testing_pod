import os
import sys
from itertools import product
import multiprocessing
import argparse
import logging
import pathlib
from timeit import default_timer as timer
from datetime import timedelta


import hydra
from omegaconf import DictConfig
import submitit

from gen_training_data_zelda import generate_training_data_zelda
from train_zelda import train_zelda
from inference_zelda import inference_zelda
import constants


RUN_PLAYBOOK = {"zelda": [generate_training_data_zelda, train_zelda, inference_zelda]}


# def init_logger(log_name):
#     log_root = pathlib.Path(__file__).parent.resolve()
#     logger = logging.getLogger(log_name)
#     log_file = f"{log_root}/{log_name}.log"

#     file_handler = logging.FileHandler(log_file)
#     file_handler.setFormatter("%(asctime)s %(levelname)s> %(funcName)s::%(filename)s::%(lineno)s - %(message)s")
#     logger.addHandler(file_handler)
#     std_handler = logging.StreamHandler(sys.stdout)
#     logger.addHandler(std_handler)
#     logger.setLevel(logging.INFO)

#     return logger


def init_logger(log_name="main"):
    log_root = pathlib.Path(__file__).parent.resolve()
    log_file = f"{log_root}/{log_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s> %(funcName)s::%(filename)s::%(lineno)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


init_logger()


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


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
    with multiprocessing.Manager() as manager:
        workers = []
        for trajectory_params in trajectories_to_generate:
            print(f"trajectory_params: {trajectory_params}")
            workers.append(
                multiprocessing.Process(
                    target=gen_data_func, args=(trajectory_params, mode)
                )
            )

        for wi, worker in enumerate(workers):
            worker.start()
            print(f"Started worker process {wi + 1}")
        for wi, worker in enumerate(workers):
            worker.join()
            print(f"Joined worker process {wi + 1}")


def run_gen_data(traj_chunk, domains):
    # logger.info(f"Calling run_gen_data with traj chunks from {traj_chunk[0]} to {traj_chunk[len(traj_chunk)-1]}")
    print(
        f"Calling run_gen_data with traj chunks from {traj_chunk[0]} to {traj_chunk[len(traj_chunk)-1]}"
    )
    gen_data_func, train_func, infer_func = RUN_PLAYBOOK[domains[0]]
    mode = "controllable"

    for traj_param_combo in traj_chunk:
        start = timer()
        gen_data_func(traj_param_combo, mode)
        end = timer()

        # logger.info(f"generate_training_data_zelda for params {traj_param_combo} took {timedelta(seconds=end-start)} seconds")
        print(
            f"generate_training_data_zelda for params {traj_param_combo} took {timedelta(seconds=end-start)} seconds"
        )


def run_train(train_chunk, domains):
    # logger.info(f"Calling run_train with train chunks from {train_chunk[0]} to {train_chunk[len(train_chunk)-1]}")
    print(
        f"Calling run_train with train chunks from {train_chunk[0]} to {train_chunk[len(train_chunk)-1]}"
    )
    # Traing models
    mode = "controllable"
    gen_data_func, train_func, infer_func = RUN_PLAYBOOK[domains[0]]
    for combo_id, params_combo in enumerate(train_chunk):
        start = timer()
        train_func(combo_id, params_combo, mode)
        end = timer()

        logging.info(
            f"train_zelda (3 models) for params {params_combo} took {timedelta(seconds=end-start)} seconds"
        )


def run_inference(infer_chunk, domains, username):
    gen_data_func, train_func, infer_func = RUN_PLAYBOOK[domains[0]]
    mode = "controllable"

    for infer_param_combo in infer_chunk:
        start = timer()
        infer_func(infer_param_combo, mode, username)
        end = timer()

        logging.info(
            f"inference_zelda for params {infer_param_combo} took {timedelta(seconds=end-start)} seconds"
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    log_folder = f"logs"
    executor = submitit.AutoExecutor(folder=log_folder)

    if cfg.process == "gen_train_data":
        init_logger("gen_train_data")

        print("Running process 'gen_train_data'")
        # Gen training data
        obs_sz, goal_sz, traj_len, td_sz = [
            param_vals_list[1]
            for param_vals_list in constants.DOMAIN_VARS_ZELDA[
                "trajectories_to_generate"
            ]
        ]
        trajectories_to_generate = product(obs_sz, goal_sz, traj_len, td_sz)
        # print(f"trajectories_to_generate: {len(list(trajectories_to_generate))}")
        traj_chunks = list(divide_chunks(list(trajectories_to_generate), 4))  # 4 chunks
        domains = ["zelda"] * len(traj_chunks)

        sum_runs = 0
        for traj_chunk in traj_chunks:
            sum_runs += len(traj_chunk)
        # print(f"sum_runs: {sum_runs}")

        if cfg.slurm:

            executor.update_parameters(
                slurm_array_parallelism=2,
                #gpus_per_node=1,
                tasks_per_node=1,
                cpus_per_task=4,
                mem_gb=50,
                timeout_min=1440,
                nodes=4
            )

            jobs = []
            with executor.batch():
                for traj_chunk in traj_chunks:
                    job = executor.submit(run_gen_data, traj_chunk, domains)
                    jobs.append(job)
        else:
            for traj_chunk in traj_chunks:
                run_gen_data(traj_chunk, domains)

    elif cfg.process == "train":
        init_logger("train")
        executor.update_parameters(
            slurm_array_parallelism=1,
            # gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=1,
            mem_gb=50,
            timeout_min=1440,
            nodes=1,
        )

        print("Running process 'train'")
        # Gen training data
        obs_sz, goal_sz, traj_len, td_sz = [
            param_vals_list[1]
            for param_vals_list in constants.DOMAIN_VARS_ZELDA["sweep_params"]
        ]
        trajectories_to_generate = product(obs_sz, goal_sz, traj_len, td_sz)
        train_chunks = list(
            divide_chunks(list(trajectories_to_generate), 4)
        )  # 11 chunks
        domains = ["zelda"] * len(train_chunks)

        sum_runs = 0
        for train_chunk in train_chunks:
            sum_runs += len(train_chunk)
        print(f"sum_runs: {sum_runs}")

        jobs = []

        if cfg.slurm:
            with executor.batch():
                for train_chunk in train_chunks:
                    job = executor.submit(run_train, train_chunk, domains)
                    jobs.append(job)
        else:
            for train_chunk in train_chunks:
                run_train(train_chunk, domains)

    elif cfg.process == "inference":
        executor.update_parameters(
            slurm_array_parallelism=2,
            # gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=1,
            mem_gb=50,
            timeout_min=1440,
            nodes=1,
        )

        print("Running process 'train'")
        # Gen training data
        obs_sz, goal_sz, traj_len, td_sz = [
            param_vals_list[1]
            for param_vals_list in constants.DOMAIN_VARS_ZELDA["sweep_params"]
        ]
        trajectories_to_generate = product(obs_sz, goal_sz, traj_len, td_sz)
        inference_chunks = list(
            divide_chunks(list(trajectories_to_generate), 4)
        )  # 11 chunks
        domains = ["zelda"] * len(inference_chunks)

        sum_runs = 0
        for inference_chunk in inference_chunks:
            sum_runs += len(inference_chunk)
        print(f"sum_runs: {sum_runs}")

        jobs = []
        with executor.batch():
            for inference_chunk in inference_chunks:
                job = executor.submit(
                    run_inference, inference_chunk, domains, cfg.username
                )
                jobs.append(job)
    
    else:
        raise Exception(f"Invalid process type {cfg.process}")


if __name__ == "__main__":
    # args = get_args()
    run()
