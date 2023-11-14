import os
import sys
import math
import time
from itertools import product
from collections import OrderedDict

from gym_pcgrl.envs.probs.loderunner_prob import LRProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
import csv

import hashlib
import numpy as np
import os
import struct
from gym import error
import random
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile, run_dikjstra, get_string_map
import pandas as pd


DOMAIN_SPEC_VARS = {
    "zelda": {
        "tiles_map": {
            "g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"
        },
        "int_map": {
            "empty": 0,
            "solid": 1,
            "player": 2,
            "key": 3,
            "door": 4,
            "bat": 5,
            "scorpion": 6,
            "spider": 7
        },
        "char_map": {
            "door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'
        },
        "goal_maps_filepath": "goal_maps/zelda/zelda_lvl{}.txt",
        "env_y": 7,
        "env_x": 11,
        "env_z": None,
        "action_space_size": 8,
        "action_pronbabilities_map": {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02},
        "gym_env_name": "zelda-narrow-v0",
        "prob": ZeldaProblem,
        "sweep_params": [
                ("obs_sizes", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
                ),
                ("goal_set_sizes", [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                ),
                ("trajectory_lengths", 
                [
                                    int(7*11*0.02),
                                    int(7*11*0.05),
                                    int(7*11*0.1),
                                    int(7*11*0.2),
                                    int(7*11*0.3),
                                    int(7*11*0.4),
                                    int(7*11*0.5),
                                    int(7*11*0.6),
                                    int(7*11*0.7),
                                    int(7*11*0.8),
                                    int(7*11*0.9),
                                    int(7*11*1.0)
                                    ]
                                    ),
                ("training_dataset_sizes", [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]
                ),
        ]
    },
    "lr": {
        "tiles_map": {
            ".": "empty",
            "b": "brick",
            "#": "ladder",
            "-": "rope",
            "B": "solid",
            "G": "gold",
            "E": "enemy",
            "M": "player",
        },
        "int_map": {
            "empty": 0,
            "brick": 1,
            "ladder": 2,
            "rope": 3,
            "solid": 4,
            "gold": 5,
            "enemy": 6,
            "player": 7,
        },
        "char_map": {
            "empty": ".",
            "brick": "b",
            "ladder": "#",
            "rope": "-",
            "solid": "B",
            "gold": "B",
            "enemy": "E",
            "player": "M",
        },
        "goal_maps_filepath": "goal_maps/lr/Level {}.txt",
        "env_y": 22,
        "env_x": 32,
        "env_z": None,
        "action_space_size": 8,
        "action_pronbabilities_map": {0: 0.55, 1: 0.24, 2: 0.1, 3: 0.04, 4: 0.03, 5: 0.03, 6: 0.005, 7: 0.005},
        "gym_env_name": "lr-narrow-v0",
        "prob": LRProblem,
        "sweep_params": [
                ("obs_sizes", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]),
                ("goal_set_sizes", [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9]),
                ("trajectory_lengths", [int(22 * 22*0.01), 
                                    int(22 * 32*0.02),
                                    int(22 * 32*0.05),
                                    int(22 * 32*0.1),
                                    int(22 * 32*0.2),
                                    int(22 * 32*0.3),
                                    int(22 * 32*0.4),
                                    int(22 * 32*0.5),
                                    int(22 * 32*0.6),
                                    int(22 * 32*0.7),
                                    int(22 * 32*0.8),
                                    int(22 * 32*0.9),
                                    int(22 * 32*1.0)]),
                ("training_dataset_sizes", [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000] ),
        ]
    },
    "lego": {
        "tiles_map": {
        },
        "int_map": {
        },
        "char_map": {
        },
        "goal_maps_filepath": "goal_maps/lego/Level {}.mpd",
        "env_y": 6,
        "env_x": 6,
        "env_z": 6,
        "action_space_size": 37, # TODO: check these numbers
        "action_pronbabilities_map": {0: 0.55, 1: 0.24, 2: 0.1, 3: 0.04, 4: 0.03, 5: 0.03, 6: 0.005, 7: 0.005},
        "gym_env_name": None,
        "sweep_params": [
                        ("obs_sizes", [1, 2, 3, 4, 5, 6]),
                        ("goal_set_sizes", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                        ("trajectory_lengths", [int(6*6*6*0.01), 
                                            int(6*6*6*0.02),
                                            int(6*6*6*0.05),
                                            int(6*6*6*0.1),
                                            int(6*6*6*0.2),
                                            int(6*6*6*0.3),
                                            int(6*6*6*0.4),
                                            int(6*6*6*0.5),
                                            int(6*6*6*0.6),
                                            int(6*6*6*0.7),
                                            int(6*6*6*0.8),
                                            int(6*6*6*0.9),
                                            int(6*6*6*1.0)]),
                        ("training_dataset_sizes", [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]),
        ]
    }
}


def generate_training_data_zelda(domain, mode, username, debug):
    # Reverse the k,v in TILES MAP for persisting back as char map .txt format
    TILES_MAP = DOMAIN_SPEC_VARS[domain]["tiles_map"]
    REV_TILES_MAP = {v: k for k, v in TILES_MAP.items()}
    INT_MAP = DOMAIN_SPEC_VARS[domain]["int_map"]
    REV_INT_MAP = {v: k for k, v in INT_MAP.items()}
    CHAR_MAP = DOMAIN_SPEC_VARS[domain]["int_map"]
    actions_list = [act for act in list(TILES_MAP.values())]
    prob = DOMAIN_SPEC_VARS[domain]["prob"]() if domain != 'lego' else None
    rep = NarrowRepresentation()
    epsilon = 0.1


    # Reads in .txt playable map and converts it to string[][]
    def to_2d_array_level(file_name):
        level = []

        with open(file_name, "r") as f:
            rows = f.readlines()
            for row in rows:
                new_row = []
                for char in row:
                    if char != "\n":
                        new_row.append(TILES_MAP[char])
                level.append(new_row)

        return level


    # Converts from string[][] to 2d int[][]
    def int_arr_from_str_arr(map):
        int_map = []
        for row_idx in range(len(map)):
            new_row = []
            for col_idx in range(len(map[0])):
                new_row.append(INT_MAP[map[row_idx][col_idx]])
            int_map.append(new_row)
        return int_map


    def to_char_level(map, dir=""):
        level = []

        for row in map:
            new_row = []
            for col in row:
                new_row.append(REV_TILES_MAP[col])
            level.append(new_row)
        level_as_str = []
        for row in level:
            level_as_str.append("".join(row) + "\n")

        with open(dir, "w") as f:
            for row in level_as_str:
                f.write(row)


    def act_seq_to_disk(act_seq, path):
        with open(path, "w") as f:
            wr = csv.writer(f)
            wr.writerows(act_seq)


    def act_seq_from_disk(path):
        act_seqs = []
        with open(path, "r") as f:
            data = f.readlines()
            for row in data:
                act_seq = [int(n) for n in row.split("\n")[0].split(",")]
                act_seqs.append(act_seq)
        return act_seqs


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


    def find_closest_goal_map(random_map, data_size, goal_set_idxs):
        smallest_hamming_dist = math.inf
        filepath = DOMAIN_SPEC_VARS[domain]["goal_maps_filepath"]
        closest_map = curr_goal_map = int_arr_from_str_arr(
                to_2d_array_level(filepath.format(goal_set_idxs[0]))
            )

        for next_id in goal_set_idxs:            
            temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
            if temp_hamm_distance < smallest_hamming_dist:
                closest_map = curr_goal_map
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
        translation_map = {v: k for k, v in INT_MAP.items()}
        str_map = []
        for row_idx in range(len(map)):
            new_row = []
            for col_idx in range(len(map[0])):
                new_row.append(translation_map[map[row_idx][col_idx]])
            str_map.append(new_row)

        return str_map


    def get_char_map(arr_map, domain):
        str_arr_map = str_arr_from_int_arr(arr_map)
        str_to_char_map = DOMAIN_SPEC_VARS[domain]["char_map"]
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
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "key", "bat", "spider", "scorpion"]),
            "nearest-enemy": 0,
            "path-length": 0
        }
        if map_stats["player"] == 1 and map_stats["regions"] == 1:
            p_x,p_y = map_locations["player"][0]
            enemies = []
            enemies.extend(map_locations["spider"])
            enemies.extend(map_locations["bat"])
            enemies.extend(map_locations["scorpion"])
            if len(enemies) > 0:
                dikjstra,_ = run_dikjstra(p_x, p_y, map, ["empty", "player", "bat", "spider", "scorpion"])
                min_dist = width * height
                for e_x,e_y in enemies:
                    if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
                        min_dist = dikjstra[e_y][e_x]
                map_stats["nearest-enemy"] = min_dist
            if map_stats["key"] == 1 and map_stats["door"] == 1:
                k_x,k_y = map_locations["key"][0]
                d_x,d_y = map_locations["door"][0]
                dikjstra,_ = run_dikjstra(p_x, p_y, map, ["empty", "key", "player", "bat", "spider", "scorpion"])
                map_stats["path-length"] += dikjstra[k_y][k_x]
                dikjstra,_ = run_dikjstra(k_x, k_y, map, ["empty", "player", "key", "door", "bat", "spider", "scorpion"])
                map_stats["path-length"] += dikjstra[d_y][d_x]

        return map_stats


    def generate_pod_greedy_tiles(
        env,
        random_target_map,
        goal_starting_map,
        total_steps,
        training_dataset_size,
        ep_len,
        crop_size,
        render=False,
        epsilon=0.02,
    ):
        play_trace = []
        old_map = goal_starting_map.copy()
        random_map = random_target_map.copy()
        current_loc = [random.randint(0, len(random_target_map)-1), random.randint(0, len(random_target_map[0])-1)]
        env.rep._old_map = np.array([np.array(l) for l in goal_starting_map])
        env.rep._x = current_loc[1]
        env.rep._y = current_loc[0]
        row_idx, col_idx = current_loc[0], current_loc[1]
        tile_count = 0

        hamm = compute_hamm_dist(random_target_map, goal_starting_map)
        curr_step = 0
        episode_len = ep_len
        env.reset()
        env.reset()
        while hamm > 0.0 and curr_step < episode_len and total_steps < training_dataset_size:
            curr_step += 1
            total_steps += 1

            new_map = old_map.copy()
            transition_info_at_step = [None, None, None]
            rep._x = col_idx
            rep._y = row_idx

            new_map[row_idx] = old_map[row_idx].copy()

            # existing tile type on the goal map
            old_tile_type = old_map[row_idx][col_idx]

            # new destructive tile
            new_tile_type = random_target_map[row_idx][col_idx]

            expert_action = [row_idx, col_idx, old_tile_type]
            destructive_action = [row_idx, col_idx, new_tile_type]
            transition_info_at_step[1] = destructive_action.copy()
            transition_info_at_step[2] = expert_action.copy()
            new_map[row_idx][col_idx] = new_tile_type

            play_trace.append(
                (
                    transform(random_map.copy(), col_idx, row_idx, crop_size),
                    expert_action.copy(),
                )
            )
            random_map[row_idx][col_idx] = old_tile_type
            

            old_map = new_map

            tile_count += 1
            col_idx = random.randint(0, len(random_target_map[0])-1)
            row_idx = random.randint(0, len(random_target_map)-1)

            hamm = compute_hamm_dist(random_target_map, old_map)
            if hamm == 0.0:
                play_trace.reverse()
                return play_trace, total_steps

        play_trace.reverse()
        return play_trace, total_steps

    def generate_controllable_pod_greedy(
        env,
        random_target_map,
        goal_starting_map,
        total_steps,
        ep_len=77,
        crop_size=9,
        render=True,
        epsilon=0.02,
    ):
        play_trace = []
        map_stat = []
        old_map = goal_starting_map.copy()
        random_map = random_target_map.copy()

        current_loc = [random.randint(0, len(random_target_map)-1), random.randint(0, len(random_target_map[0])-1)]
        env.rep._old_map = np.array([np.array(l) for l in goal_starting_map])
        env.rep._x = current_loc[1]
        env.rep._y = current_loc[0]
        row_idx, col_idx = current_loc[0], current_loc[1]
        tile_count = 0

        hamm = compute_hamm_dist(random_target_map, goal_starting_map)
        curr_step = 0
        episode_len = ep_len
        env.reset()
        env.reset()

        new_map = old_map.copy()
        string_map_for_map_stats = str_arr_from_int_arr(new_map)
        new_map_stats_dict = env._prob.get_stats(string_map_for_map_stats)

        # Targets
        num_regions = new_map_stats_dict["regions"]
        num_enemies = new_map_stats_dict["enemies"]
        nearest_enemy = new_map_stats_dict["nearest-enemy"]
        path_length = new_map_stats_dict["path-length"]

        actual_stats = []
        conditional_diffs = []

        while hamm > 0.0 and curr_step < episode_len and total_steps < training_dataset_size:
            curr_step += 1
            total_steps += 1

            new_map = old_map.copy()
            transition_info_at_step = [None, None, None]
            rep._x = col_idx
            rep._y = row_idx

            old_map_string_map_for_map_stats = str_arr_from_int_arr(new_map)
            old_map_stats_dict = env._prob.get_stats(old_map_string_map_for_map_stats)

            new_map[row_idx] = old_map[row_idx].copy()

            old_tile_type = old_map[row_idx][col_idx]
            new_tile_type = random_target_map[row_idx][col_idx]

            string_map_for_map_stats = str_arr_from_int_arr(new_map)
            new_map_stats_dict = env._prob.get_stats(string_map_for_map_stats)

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

            curr_map_stats = env._prob.get_stats(str_arr_from_int_arr(new_map))
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

            old_map = new_map

            col_idx = random.randint(0, len(random_target_map[0])-1)
            row_idx = random.randint(0, len(random_target_map)-1)

            hamm = compute_hamm_dist(random_target_map, old_map)
            if hamm == 0.0:
                play_trace.reverse()
                map_stat.reverse()
                return play_trace, total_steps, map_stat, actual_stats, conditional_diffs

        play_trace.reverse()
        map_stat.reverse()
        return play_trace, total_steps, map_stat, actual_stats, conditional_diffs


    # This code is for generating the maps
    def render_map(map, prob, rep, filename="", ret_image=False, pause=True):
        # format image of map for rendering
        if not filename:
            img = prob.render(map)
        else:
            img = to_2d_array_level(filename)
        img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
        img = np.array(img)
        if ret_image:
            return img
        else:
            ren = rendering.SimpleImageViewer()
            ren.imshow(img)

            if pause:
                input(f"")
            else:
                time.sleep(0.05)
            ren.close()


    def int_map_to_str_map(curr_map):
        new_level = []
        for idx, row in enumerate(curr_map):
            new_row = []
            for j_idx, col in enumerate(row):
                new_row.append(REV_INT_MAP[col])
            new_level.append(new_row)
        return new_level


    rng, seed = np_random(None)
    combo_id = 0
    sample_id = 0
    goal_maps_dir = f"/scratch/{username}/sweep_testing_pod/goal_maps/{domain}" # "/Users/matt/sweep_testing_pod/goal_maps" # TODO:  
    sweep_param_obs_size = None
    sweep_param_goal_set_size = None
    sweep_param_trajectory_length = None
    sweep_param_training_dataset_size = None
    sweep_param_random_start_map = None
    sweep_param_goal_map = None
    sweep_param_path_to_trajectories_dir_loc = None
    sweep_param_prefix_filename_of_model = None
    

    sweep_schema = {
        "combo_id": [],
        "sample_id": [],
        "sweep_param_obs_size": [],
        "sweep_param_goal_set_size": [],
        "sweep_param_trajectory_length": [],
        "sweep_param_training_dataset_size": [],
        "path_to_trajectories_dir_loc": [],
        "prefix_filename_of_model": [],
    }


    start_and_goal_shema_map = {
        "random_start_map": [],
        "goal_map": [],
        "sample_id": [],
        "combo_id": []
    }

    root_dir = f"/scratch/{username}/overlay/sweep_testing_pod" # testing on matt's laptop:  f"/Users/matt/sweep_testing_pod"
    
    obs_sizes, goal_set_sizes, trajectory_lengths, training_dataset_sizes = [p[1] for p in DOMAIN_SPEC_VARS[domain]["sweep_params"]]

    if mode == "non_controllable":
        for obs_size, goal_set_size, trajectory_length, training_dataset_size in product(obs_sizes, goal_set_sizes, trajectory_lengths, training_dataset_sizes):
            sweep_param_obs_size = obs_size
            print(f"debug: {debug}")
            if debug:
                print(f"Generating samples for obs_size={obs_size} goal_set_size={goal_set_size} trajectory_length={trajectory_length} training_dataset_size={training_dataset_size}")
            
            root_data_dir = f"{root_dir}/data/{domain}/{mode}"
            if not os.path.exists(root_data_dir):
                os.makedirs(root_data_dir)

            combo_id += 1
  

            for sample_num in range(1,11):
                sample_id += 1

                if debug:
                    print(f"SampleID: {sample_id}")
                
                training_trajectory_filepath = f"{root_data_dir}/comboID_{combo_id}_sampleID_{sample_id}"
                
                if not os.path.exists(training_trajectory_filepath):
                    os.makedirs(training_trajectory_filepath)

                training_trajectory_filepath =  f"{training_trajectory_filepath}/trajectories"
                if not os.path.exists(training_trajectory_filepath):
                    os.makedirs(training_trajectory_filepath)

                trained_models_dir = f"{root_data_dir}/comboID_{combo_id}_sampleID_{sample_id}/models"
                if not os.path.exists(trained_models_dir):
                    os.makedirs(trained_models_dir)

                sweep_schema["path_to_trajectories_dir_loc"].append(training_trajectory_filepath)
                sweep_schema["sample_id"].append(sample_id)
                sweep_schema["prefix_filename_of_model"].append(f"model_comboID_{combo_id}_sampleID_{sample_id}_")
                sweep_schema["combo_id"].append(combo_id)
                sweep_schema["sweep_param_obs_size"].append(sweep_param_obs_size)
                sweep_schema["sweep_param_goal_set_size"].append(goal_set_size)
                sweep_schema["sweep_param_trajectory_length"].append(trajectory_length)
                sweep_schema["sweep_param_training_dataset_size"].append(training_dataset_size) 


                goal_maps_set = [i for i in range(len(os.listdir(goal_maps_dir)))]

                
                random.shuffle(goal_maps_set)
                goal_set_idxs = goal_maps_set[:goal_set_size]

                # TODO: store goal_set_idxs in df
                dict_len = (obs_size**2) * DOMAIN_SPEC_VARS[domain]["action_space_size"]
                total_steps = 0
                exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
                exp_traj_dict["target"] = []
                save_count = 0
                while total_steps < training_dataset_size:
                    play_traces = []
                    cropped_wrapper = CroppedImagePCGRLWrapper(
                        DOMAIN_SPEC_VARS[domain]["gym_env_name"],
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
                    start_map = gen_random_map(
                        rng,
                        DOMAIN_SPEC_VARS[domain]["env_x"],
                        DOMAIN_SPEC_VARS[domain]["env_y"],
                        DOMAIN_SPEC_VARS[domain]["action_pronbabilities_map"],
                    )
                    goal_map = find_closest_goal_map(start_map, goal_set_size, goal_set_idxs)
                    start_map_str = get_char_map(start_map, domain)
                    goal_map_str =  get_char_map(goal_map, domain)

                    start_and_goal_shema_map["random_start_map"].append(start_map_str)
                    start_and_goal_shema_map["goal_map"].append(goal_map_str)
                    start_and_goal_shema_map["sample_id"].append(sample_id)
                    start_and_goal_shema_map["combo_id"].append(combo_id)

                    
                    # TODO: This is different between controllable/non-controllable
                    play_trace, temp_num_steps = generate_pod_greedy_tiles(
                        pcgrl_env,
                        start_map,
                        goal_map,
                        total_steps,
                        training_dataset_size,
                        trajectory_length,
                        obs_size,
                        render=False,
                    )
                    total_steps = temp_num_steps
                    play_traces.append(play_trace)

                    for p_i in play_trace:
                        action = p_i[1][-1]
                        exp_traj_dict["target"].append(action)
                        pt = p_i[0]
                        assert dict_len == len(
                            pt
                        ), f"len(pt) is {len(pt)} and dict_len is {dict_len}"
                        for i in range(len(pt)):
                            exp_traj_dict[f"col_{i}"].append(pt[i])

                    if total_steps > 0 and total_steps % min(training_dataset_size, 100000) == 0:
                        print(f"saving df at ts {total_steps}")
                        df = pd.DataFrame(data=exp_traj_dict)
                        df.to_csv(
                            f"{training_trajectory_filepath}/{save_count}.csv",
                            index=False,
                        )
                        save_count += 1

                if debug:
                    print(f"Total steps: {total_steps}")

                if total_steps > 0 and total_steps % min(training_dataset_size, 100000) != 0:
                    print(f"saving df at ts {total_steps}")
                    df = pd.DataFrame(data=exp_traj_dict)
                    df.to_csv(
                        f"{training_trajectory_filepath}/{save_count}.csv",
                        index=False,
                    )
                    exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
                    exp_traj_dict["target"] = []

                goal_maps_set = None

    elif mode == "controllable":
        goal_stats_set = {}
        for obs_size, goal_set_size, trajectory_length, training_dataset_size in product(obs_sizes, goal_set_sizes, trajectory_lengths, training_dataset_sizes):
            print(f"debug: {debug}")
            sweep_param_obs_size = obs_size
            if debug:
                print(f"Generating samples for obs_size={obs_size} goal_set_size={goal_set_size} trajectory_length={trajectory_length} training_dataset_size={training_dataset_size}")
            
            root_data_dir = f"{root_dir}/data/{domain}/{mode}"
            if not os.path.exists(root_data_dir):
                os.makedirs(root_data_dir)

            combo_id += 1
             

            for sample_num in range(1,2): #range(1,11):
                sample_id += 1

                if debug:
                    print(f"SampleID: {sample_id}")

                goal_stats_set[f"{combo_id}_{sample_id}"] = []
                training_trajectory_filepath = f"{root_data_dir}/comboID_{combo_id}_sampleID_{sample_id}"
                
                if not os.path.exists(training_trajectory_filepath):
                    os.makedirs(training_trajectory_filepath)

                training_trajectory_filepath =  f"{training_trajectory_filepath}/trajectories"
                if not os.path.exists(training_trajectory_filepath):
                    os.makedirs(training_trajectory_filepath)

                trained_models_dir = f"{root_data_dir}/comboID_{combo_id}_sampleID_{sample_id}/models"
                if not os.path.exists(trained_models_dir):
                    os.makedirs(trained_models_dir)

                goal_stats_map_root_path = f"{root_data_dir}/comboID_{combo_id}_sampleID_{sample_id}"
                
                sweep_schema["path_to_trajectories_dir_loc"].append(training_trajectory_filepath)
                sweep_schema["sample_id"].append(sample_id)
                sweep_schema["prefix_filename_of_model"].append(f"model_comboID_{combo_id}_sampleID_{sample_id}_")
                sweep_schema["combo_id"].append(combo_id)
                sweep_schema["sweep_param_obs_size"].append(sweep_param_obs_size)
                sweep_schema["sweep_param_goal_set_size"].append(goal_set_size)
                sweep_schema["sweep_param_trajectory_length"].append(trajectory_length)
                sweep_schema["sweep_param_training_dataset_size"].append(training_dataset_size)  


                goal_maps_set = [i for i in range(len(os.listdir(goal_maps_dir)))]

                
                random.shuffle(goal_maps_set)
                goal_set_idxs = goal_maps_set[:goal_set_size]

                # TODO: store goal_set_idxs in df
                dict_len = (obs_size**2) * DOMAIN_SPEC_VARS[domain]["action_space_size"]
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
                save_count = 0
                while total_steps < training_dataset_size:
                    play_traces = []
                    cropped_wrapper = CroppedImagePCGRLWrapper(
                        DOMAIN_SPEC_VARS[domain]["gym_env_name"],
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
                    start_map = gen_random_map(
                        rng,
                        DOMAIN_SPEC_VARS[domain]["env_x"],
                        DOMAIN_SPEC_VARS[domain]["env_y"],
                        DOMAIN_SPEC_VARS[domain]["action_pronbabilities_map"],
                    )
                    goal_map = find_closest_goal_map(start_map, goal_set_size, goal_set_idxs)
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
                        ["empty", "solid", "player", "key", "door", "bat", "scorpion", "spider"],
                    )
                    start_map_str = get_char_map(start_map, domain)
                    goal_map_str =  get_char_map(goal_map, domain)

                    start_and_goal_shema_map["random_start_map"].append(start_map_str)
                    start_and_goal_shema_map["goal_map"].append(goal_map_str)
                    start_and_goal_shema_map["sample_id"].append(sample_id)
                    start_and_goal_shema_map["combo_id"].append(combo_id)

                    
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
                        render=False,
                    )
                    total_steps = temp_num_steps

                    play_traces.append(play_trace)
                    goal_stats_set[f"{combo_id}_{sample_id}"].append(
                        (
                            map_stats["regions"],
                            map_stats["enemies"],
                            map_stats["nearest-enemy"],
                            map_stats["path-length"],
                        )
                    )

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

                    if total_steps > 0 and total_steps % min(training_dataset_size, 100000) == 0:
                        print(f"saving df at ts {total_steps}")
                        cols_to_keep = [f"col_{i}" for i in range(dict_len)] + ["num_regions_signed", "num_enemies_signed", "nearest_enemy_signed", "path_length_signed", "target"]
                        df = pd.DataFrame(data=exp_traj_dict)
                        df[cols_to_keep].to_csv(
                            f"{training_trajectory_filepath}/{save_count}.csv",
                            index=False,
                        )

                        exp_traj_dict = OrderedDict()
                        exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
                        exp_traj_dict["num_regions_targets"] = []
                        exp_traj_dict["num_enemies_targets"] = []
                        exp_traj_dict["nearest_enemy_targets"] = []
                        exp_traj_dict["path_length_targets"] = []

                        exp_traj_dict["actual_num_regions"] = []
                        exp_traj_dict["actual_num_enemies"] = []
                        exp_traj_dict["actual_nearest_enemy"] = []
                        exp_traj_dict["actual_path_length"] = []

                        exp_traj_dict["num_regions_signed"] = []
                        exp_traj_dict["num_enemies_signed"] = []
                        exp_traj_dict["nearest_enemy_signed"] = []
                        exp_traj_dict["path_length_signed"] = []
                    

                        exp_traj_dict["target"] = []
                        save_count += 1

                if total_steps > 0 and total_steps % min(training_dataset_size, 100000) != 0:
                    print(f"saving df at ts {total_steps}")
                    cols_to_keep = [f"col_{i}" for i in range(dict_len)] + ["num_regions_signed", "num_enemies_signed", "nearest_enemy_signed", "path_length_signed", "target"]
                    df = pd.DataFrame(data=exp_traj_dict)
                    df[cols_to_keep].to_csv(
                                f"{training_trajectory_filepath}/{save_count}.csv",
                                index=False,
                            )
                    exp_traj_dict = OrderedDict()
                    exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
                    exp_traj_dict["num_regions_targets"] = []
                    exp_traj_dict["num_enemies_targets"] = []
                    exp_traj_dict["nearest_enemy_targets"] = []
                    exp_traj_dict["path_length_targets"] = []

                    exp_traj_dict["actual_num_regions"] = []
                    exp_traj_dict["actual_num_enemies"] = []
                    exp_traj_dict["actual_nearest_enemy"] = []
                    exp_traj_dict["actual_path_length"] = []

                    exp_traj_dict["num_regions_signed"] = []
                    exp_traj_dict["num_enemies_signed"] = []
                    exp_traj_dict["nearest_enemy_signed"] = []
                    exp_traj_dict["path_length_signed"] = []

                    exp_traj_dict["target"] = []

                goal_maps_set = None

            df_goal_stats = pd.DataFrame(goal_stats_set[f"{combo_id}_{sample_id}"])
            df_goal_stats.to_csv(f"{goal_stats_map_root_path}/goal_stats_COMBO_ID_{combo_id}_SAMPLE_ID_{sample_id}.csv", index=False)

    df_sweep_schema = pd.DataFrame(sweep_schema)
    df_sweep_schema.to_csv(f"{root_dir}/data/{domain}/{mode}/sweep_schema.csv", index=False)

    df_map_start_and_goal_map_schema = pd.DataFrame(start_and_goal_shema_map)
    df_map_start_and_goal_map_schema.to_csv(f"{root_dir}/data/{domain}/{mode}/start_and_goal_shema_map.csv", index=False)

    



# NEXT/REMAINING STEPS DATA GENERATION
    # TODO: store goal_set_idxs in df (line 564)
    # Finalize what data gets written and make sure the organization (i.e. dataframe schema is correct); check lucid chart app to confirm we're not missing anything 
    #    (e.g. the starting map, the goal map, etc.)
    # Remove the print statements
    # Make the code controllable
    # Add the lego PoD code (controllable)


# NEXT/REMAINING STEPS TRAINING CODE


# NEXT/REMAINING STEPS INFERENCE CODE


