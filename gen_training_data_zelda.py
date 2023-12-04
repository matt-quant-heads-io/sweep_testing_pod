import os
import math
from collections import OrderedDict
import hashlib
import struct
import random
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import pandas as pd
from gym import error

from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper
from gym_pcgrl.envs.helper import (
    get_tile_locations,
    calc_num_regions,
    calc_certain_tile,
    run_dikjstra,
    get_string_map,
)
import constants


def generate_training_data_zelda(sweep_params, mode):
    start = timer()

    # logger.info(f"Calling generate_training_data_zelda with params={sweep_params}, mode={mode}")
    print(
        f"Calling generate_training_data_zelda with params={sweep_params}, mode={mode}"
    )
    if not os.path.exists(f"{constants.ZELDA_DATA_ROOT}/{mode}"):
        os.makedirs(f"{constants.ZELDA_DATA_ROOT}/{mode}")

    trajectories_to_skip_dir = (
        f"{constants.ZELDA_DATA_ROOT}/{mode}/trajectories_to_skip"
    )
    if not os.path.exists(trajectories_to_skip_dir):
        os.makedirs(trajectories_to_skip_dir)

    print(f"sweep_params: {sweep_params}")
    (obs_size, goal_set_size, trajectory_length, training_dataset_size) = sweep_params
    trajectory_skip_filename = f"goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}.done"
    trajectory_skip_filename_alt = f"goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}.csv"
    if os.path.exists(
        f"{trajectories_to_skip_dir}/{trajectory_skip_filename}"
    ) or os.path.exists(f"{trajectories_to_skip_dir}/{trajectory_skip_filename_alt}"):
        print(f"Skipping training data generation for {trajectory_skip_filename}.")
        return

    rep = NarrowRepresentation()

    # Reads in .txt playable map and converts it to string[][]
    def to_2d_array_level(file_name):
        level = []

        with open(file_name, "r") as f:
            rows = f.readlines()
            for row in rows:
                new_row = []
                for char in row:
                    if char != "\n":
                        new_row.append(constants.TILES_MAP_ZELDA[char])
                level.append(new_row)

        return level

    # Converts from string[][] to 2d int[][]
    def int_arr_from_str_arr(map):
        int_map = []
        for row_idx in range(len(map)):
            new_row = []
            for col_idx in range(len(map[0])):
                new_row.append(constants.INT_MAP_ZELDA[map[row_idx][col_idx]])
            int_map.append(new_row)
        return int_map

    def gen_random_map(random, width, height, prob):
        map = random.choice(
            list(prob.keys()), size=(height, width), p=list(prob.values())
        ).astype(np.uint8)
        return map

    def _int_list_from_bigint(bigint):
        # Special case 0
        if bigint < 0:
            raise error.Error("Seed must be non-negative, not {}".format(bigint))
        elif bigint == 0:
            return [0]

        ints = []
        while bigint > 0:
            bigint, mod = divmod(bigint, 2**32)
            ints.append(mod)
        return ints

    # TODO: don't hardcode sizeof_int here
    def _bigint_from_bytes(bytes):
        sizeof_int = 4
        padding = sizeof_int - len(bytes) % sizeof_int
        bytes += b"\0" * padding
        int_count = int(len(bytes) / sizeof_int)
        unpacked = struct.unpack("{}I".format(int_count), bytes)
        accum = 0
        for i, val in enumerate(unpacked):
            accum += 2 ** (sizeof_int * 8 * i) * val
        return accum

    def create_seed(a=None, max_bytes=8):
        """Create a strong random seed. Otherwise, Python 2 would seed using
        the system time, which might be non-robust especially in the
        presence of concurrency.

        Args:
            a (Optional[int, str]): None seeds from an operating system specific randomness source.
            max_bytes: Maximum number of bytes to use in the seed.
        """
        # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
        if a is None:
            a = _bigint_from_bytes(os.urandom(max_bytes))
        elif isinstance(a, str):
            a = a.encode("utf8")
            a += hashlib.sha512(a).digest()
            a = _bigint_from_bytes(a[:max_bytes])
        elif isinstance(a, int):
            a = a % 2 ** (8 * max_bytes)
        else:
            raise error.Error("Invalid type for seed: {} ({})".format(type(a), a))

        return a

    def hash_seed(seed=None, max_bytes=8):
        """Any given evaluation is likely to have many PRNG's active at
        once. (Most commonly, because the environment is running in
        multiple processes.) There's literature indicating that having
        linear correlations between seeds of multiple PRNG's can correlate
        the outputs:

        http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
        http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
        http://dl.acm.org/citation.cfm?id=1276928

        Thus, for sanity we hash the seeds before using them. (This scheme
        is likely not crypto-strength, but it should be good enough to get
        rid of simple correlations.)

        Args:
            seed (Optional[int]): None seeds from an operating system specific randomness source.
            max_bytes: Maximum number of bytes to use in the hashed seed.
        """
        if seed is None:
            seed = create_seed(max_bytes=max_bytes)
        hash = hashlib.sha512(str(seed).encode("utf8")).digest()
        return _bigint_from_bytes(hash[:max_bytes])

    def np_random(seed=None):
        if seed is not None and not (isinstance(seed, int) and 0 <= seed):
            raise error.Error(
                "Seed must be a non-negative integer or omitted, not {}".format(seed)
            )

        seed = create_seed(seed)

        rng = np.random.RandomState()
        rng.seed(_int_list_from_bigint(hash_seed(seed)))
        return rng, seed

    def find_closest_goal_map(random_map, goal_set_idxs):
        smallest_hamming_dist = math.inf
        filepath = constants.DOMAIN_VARS_ZELDA["goal_maps_filepath"]
        closest_map = curr_goal_map = int_arr_from_str_arr(
            to_2d_array_level(filepath.format(goal_set_idxs[0]))
        )

        for curr_idx in goal_set_idxs:
            temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
            if temp_hamm_distance < smallest_hamming_dist:
                closest_map = curr_goal_map
                smallest_hamming_dist = temp_hamm_distance

            curr_goal_map = int_arr_from_str_arr(
                to_2d_array_level(filepath.format(curr_idx))
            )
        return closest_map

    def compute_hamm_dist(random_map, goal):
        hamming_distance = 0.0
        for i in range(len(random_map)):
            for j in range(len(random_map[0])):
                if random_map[i][j] != goal[i][j]:
                    hamming_distance += 1
        return float(hamming_distance / (len(random_map) * len(random_map[0])))

    def transform(obs, x, y, crop_size):
        map = obs
        # View Centering
        size = crop_size
        pad = crop_size // 2
        padded = np.pad(map, pad, constant_values=1)
        cropped = padded[y : y + size, x : x + size]
        obs = cropped
        new_obs = []
        for i in range(len(obs)):
            for j in range(len(obs[0])):
                new_tile = [0] * 8
                new_tile[obs[i][j]] = 1
                new_obs.extend(new_tile)
        return new_obs

    def str_arr_from_int_arr(map):
        translation_map = {v: k for k, v in constants.INT_MAP_ZELDA.items()}
        str_map = []
        for row_idx in range(len(map)):
            new_row = []
            for col_idx in range(len(map[0])):
                new_row.append(translation_map[map[row_idx][col_idx]])
            str_map.append(new_row)

        return str_map

    def get_char_map(arr_map):
        str_arr_map = str_arr_from_int_arr(arr_map)
        str_to_char_map = constants.DOMAIN_VARS_ZELDA["char_map"]
        char_map = ""
        for row in str_arr_map:
            for str_tile in row:
                char_map += str_to_char_map[str_tile]

        return char_map

    def get_stats(map, width, height, list_of_tile_types):
        map_locations = get_tile_locations(map, list_of_tile_types)
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "key": calc_certain_tile(map_locations, ["key"]),
            "door": calc_certain_tile(map_locations, ["door"]),
            "enemies": calc_certain_tile(map_locations, ["bat", "spider", "scorpion"]),
            "regions": calc_num_regions(
                map,
                map_locations,
                ["empty", "player", "key", "bat", "spider", "scorpion"],
            ),
            "nearest-enemy": 0,
            "path-length": 0,
        }
        if map_stats["player"] == 1 and map_stats["regions"] == 1:
            p_x, p_y = map_locations["player"][0]
            enemies = []
            enemies.extend(map_locations["spider"])
            enemies.extend(map_locations["bat"])
            enemies.extend(map_locations["scorpion"])
            if len(enemies) > 0:
                dikjstra, _ = run_dikjstra(
                    p_x, p_y, map, ["empty", "player", "bat", "spider", "scorpion"]
                )
                min_dist = width * height
                for e_x, e_y in enemies:
                    if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
                        min_dist = dikjstra[e_y][e_x]
                map_stats["nearest-enemy"] = min_dist
            if map_stats["key"] == 1 and map_stats["door"] == 1:
                k_x, k_y = map_locations["key"][0]
                d_x, d_y = map_locations["door"][0]
                dikjstra, _ = run_dikjstra(
                    p_x,
                    p_y,
                    map,
                    ["empty", "key", "player", "bat", "spider", "scorpion"],
                )
                map_stats["path-length"] += dikjstra[k_y][k_x]
                dikjstra, _ = run_dikjstra(
                    k_x,
                    k_y,
                    map,
                    ["empty", "player", "key", "door", "bat", "spider", "scorpion"],
                )
                map_stats["path-length"] += dikjstra[d_y][d_x]

        return map_stats

    def generate_controllable_pod_greedy(
        env,
        random_target_map,
        goal_starting_map,
        total_steps,
        ep_len=77,
        crop_size=9,
    ):
        play_trace = []
        map_stat = []
        old_map = goal_starting_map.copy()
        random_map = random_target_map.copy()

        current_loc = [
            random.randint(0, len(random_target_map) - 1),
            random.randint(0, len(random_target_map[0]) - 1),
        ]
        env.rep._old_map = np.array([np.array(l) for l in goal_starting_map])
        env.rep._x = current_loc[1]
        env.rep._y = current_loc[0]
        row_idx, col_idx = current_loc[0], current_loc[1]

        hamm = compute_hamm_dist(random_target_map, goal_starting_map)
        curr_step = 0
        episode_len = ep_len
        env.reset()

        new_map = old_map.copy()
        string_map_for_map_stats = str_arr_from_int_arr(new_map)
        new_map_stats_dict = env.prob.get_stats(string_map_for_map_stats)

        # Targets
        num_regions = new_map_stats_dict["regions"]
        num_enemies = new_map_stats_dict["enemies"]
        nearest_enemy = new_map_stats_dict["nearest-enemy"]
        path_length = new_map_stats_dict["path-length"]

        actual_stats = []
        conditional_diffs = []

        while (
            hamm > 0.0
            and curr_step < episode_len
            and total_steps < training_dataset_size
        ):

            new_map = old_map.copy()
            transition_info_at_step = [None, None, None]
            rep._x = col_idx
            rep._y = row_idx

            new_map[row_idx] = old_map[row_idx].copy()

            old_tile_type = old_map[row_idx][col_idx]
            new_tile_type = random_target_map[row_idx][col_idx]

            string_map_for_map_stats = str_arr_from_int_arr(new_map)
            new_map_stats_dict = env.prob.get_stats(string_map_for_map_stats)

            enemies_diff = num_enemies - new_map_stats_dict["enemies"]
            if enemies_diff > 0:
                enemies_diff = 3
            elif enemies_diff < 0:
                enemies_diff = 1
            else:
                enemies_diff = 2

            regions_diff = num_regions - new_map_stats_dict["regions"]
            if regions_diff > 0:
                regions_diff = 3
            elif regions_diff < 0:
                regions_diff = 1
            else:
                regions_diff = 2

            nearest_enemies_diff = nearest_enemy - new_map_stats_dict["nearest-enemy"]
            if nearest_enemies_diff > 0:
                nearest_enemies_diff = 3
            elif nearest_enemies_diff < 0:
                nearest_enemies_diff = 1
            else:
                nearest_enemies_diff = 2

            path_diff = path_length - new_map_stats_dict["path-length"]
            if path_diff > 0:
                path_diff = 3
            elif path_diff < 0:
                path_diff = 1
            else:
                path_diff = 2

            conditional_diffs.append(
                (regions_diff, enemies_diff, nearest_enemies_diff, path_diff)
            )

            curr_map_stats = env.prob.get_stats(str_arr_from_int_arr(new_map))
            actual_stats.append(
                (
                    curr_map_stats["regions"],
                    curr_map_stats["enemies"],
                    curr_map_stats["nearest-enemy"],
                    curr_map_stats["path-length"],
                )
            )

            expert_action = [row_idx, col_idx, old_tile_type]
            destructive_action = [row_idx, col_idx, new_tile_type]
            transition_info_at_step[1] = destructive_action.copy()
            transition_info_at_step[2] = expert_action.copy()
            new_map[row_idx][col_idx] = new_tile_type

            map_stat.append((num_regions, num_enemies, nearest_enemy, path_length))
            play_trace.append(
                (
                    transform(random_map.copy(), col_idx, row_idx, crop_size),
                    expert_action.copy(),
                )
            )
            random_map[row_idx][col_idx] = old_tile_type

            curr_step += 1
            total_steps += 1

            old_map = new_map

            col_idx += 1
            if col_idx >= 11:
                col_idx = 0
                row_idx += 1
                if row_idx >= 7:
                    row_idx = 0

            hamm = compute_hamm_dist(random_target_map, old_map)
            if hamm == 0.0:
                play_trace.reverse()
                map_stat.reverse()
                return (
                    play_trace,
                    total_steps,
                    map_stat,
                    actual_stats,
                    conditional_diffs,
                )

        play_trace.reverse()
        map_stat.reverse()
        return play_trace, total_steps, map_stat, actual_stats, conditional_diffs

    rng, _ = np_random(None)

    trajectories_dir = f"{constants.ZELDA_DATA_ROOT}/{mode}/trajectories"
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)

    path_to_trajectory = f"{trajectories_dir}/goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}.csv"

    goal_maps_set = [i for i in range(len(os.listdir(constants.ZELDA_GOAL_MAPS_ROOT)))]
    random.shuffle(goal_maps_set)
    goal_set_idxs = goal_maps_set[:goal_set_size]

    dict_len = (obs_size**2) * constants.DOMAIN_VARS_ZELDA["action_space_size"]
    total_steps = 0
    exp_traj_dict = OrderedDict()
    exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}

    # TODO: Modify according to LR, Lego
    exp_traj_dict["actual_num_regions"] = []
    exp_traj_dict["actual_num_enemies"] = []
    exp_traj_dict["actual_nearest_enemy"] = []
    exp_traj_dict["actual_path_length"] = []

    exp_traj_dict["num_regions_targets"] = []
    exp_traj_dict["num_enemies_targets"] = []
    exp_traj_dict["nearest_enemy_targets"] = []
    exp_traj_dict["path_length_targets"] = []

    exp_traj_dict["num_regions_signed"] = []
    exp_traj_dict["num_enemies_signed"] = []
    exp_traj_dict["nearest_enemy_signed"] = []
    exp_traj_dict["path_length_signed"] = []

    exp_traj_dict["target"] = []
    num_start_maps = training_dataset_size // trajectory_length + 1
    start_maps = [
        gen_random_map(
            rng,
            constants.DOMAIN_VARS_ZELDA["env_x"],
            constants.DOMAIN_VARS_ZELDA["env_y"],
            constants.DOMAIN_VARS_ZELDA["action_pronbabilities_map"],
        )
        for _ in range(num_start_maps * 2)
    ]

    while total_steps < training_dataset_size:
        play_traces = []
        cropped_wrapper = CroppedImagePCGRLWrapper(
            constants.DOMAIN_VARS_ZELDA["gym_env_name"],
            obs_size,
            **{
                "change_percentage": 1,
                "trials": 1,
                "verbose": True,
                "cropped_size": obs_size,
                "render": False,
            },
        )
        pcgrl_env = cropped_wrapper.pcgrl_env
        start_map = start_maps.pop()
        goal_map = find_closest_goal_map(start_map, goal_set_idxs)
        map_stats = get_stats(
            get_string_map(
                np.array(goal_map),
                [
                    "empty",
                    "solid",
                    "player",
                    "key",
                    "door",
                    "bat",
                    "scorpion",
                    "spider",
                ],
            ),
            11,
            7,
            [
                "empty",
                "solid",
                "player",
                "key",
                "door",
                "bat",
                "scorpion",
                "spider",
            ],
        )

        (
            play_trace,
            temp_num_steps,
            map_stat,
            actual_stats,
            controllable_diffs,
        ) = generate_controllable_pod_greedy(
            pcgrl_env,
            start_map,
            goal_map,
            total_steps,
            ep_len=trajectory_length,
            crop_size=obs_size,
        )
        total_steps = temp_num_steps
        print(f"total_steps now: {total_steps}")

        play_traces.append(play_trace)
        for i, p_i in enumerate(play_trace):
            action = p_i[-1][-1]

            # Map characteristics here
            num_regions = map_stats["regions"]
            num_enemies = map_stats["enemies"]
            nearest_enemy = map_stats["nearest-enemy"]
            path_length = map_stats["path-length"]

            exp_traj_dict["num_regions_targets"].append(num_regions)
            exp_traj_dict["num_enemies_targets"].append(num_enemies)
            exp_traj_dict["nearest_enemy_targets"].append(nearest_enemy)
            exp_traj_dict["path_length_targets"].append(path_length)

            exp_traj_dict["num_regions_signed"].append(controllable_diffs[i][0])
            exp_traj_dict["num_enemies_signed"].append(controllable_diffs[i][1])
            exp_traj_dict["nearest_enemy_signed"].append(controllable_diffs[i][2])
            exp_traj_dict["path_length_signed"].append(controllable_diffs[i][3])

            exp_traj_dict["actual_num_regions"].append(actual_stats[i][0])
            exp_traj_dict["actual_num_enemies"].append(actual_stats[i][1])
            exp_traj_dict["actual_nearest_enemy"].append(actual_stats[i][2])
            exp_traj_dict["actual_path_length"].append(actual_stats[i][3])
            exp_traj_dict["target"].append(action)

            pt = p_i[0]
            assert dict_len == len(
                pt
            ), f"len(pt) is {len(pt)} and dict_len is {dict_len}"
            for i in range(len(pt)):
                exp_traj_dict[f"col_{i}"].append(pt[i])

    cols_to_keep = [f"col_{i}" for i in range(dict_len)] + [
        "num_regions_signed",
        "num_enemies_signed",
        "nearest_enemy_signed",
        "path_length_signed",
        "target",
    ]
    df = pd.DataFrame(data=exp_traj_dict)
    df[cols_to_keep].to_csv(path_to_trajectory, index=False)
    del df

    goal_maps_set = None

    with open(f"{trajectories_to_skip_dir}/{trajectory_skip_filename}", "w") as f:
        print(f"Finished generating training data for {trajectory_skip_filename}")
        f.write("")

    end = timer()

    # logger.info(f"generate_training_data_zelda for params {traj_param_combo} took {timedelta(seconds=end-start)} seconds")
    print(
        f"generate_training_data_zelda for params {sweep_params} took {timedelta(seconds=end-start)} seconds"
    )
