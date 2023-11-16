"""
Run a trained agent and get generated maps
"""
import os
import sys

# import model
# from stable_baselines import PPO2

import time
from utils import make_vec_envs, int_map_from_onehot
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.helper import get_string_map

from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

import pandas as pd
import numpy as np
from keras.utils import np_utils

from decimal import Decimal as D
from decimal import getcontext

getcontext().prec = 8


TILES_MAP = {
    "g": "door",
    "+": "key",
    "A": "player",
    "1": "bat",
    "2": "spider",
    "3": "scorpion",
    "w": "solid",
    ".": "empty",
}

INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7,
}

# For hashing maps to avoid duplicate goal states
CHAR_MAP = {
    "door": "a",
    "key": "b",
    "player": "c",
    "bat": "d",
    "spider": "e",
    "scorpion": "f",
    "solid": "g",
    "empty": "h",
}

REV_TILES_MAP = {
    "door": "g",
    "key": "+",
    "player": "A",
    "bat": "1",
    "spider": "2",
    "scorpion": "3",
    "solid": "w",
    "empty": ".",
}


def to_char_level(map, dir=""):
    level = []

    for row in map:
        new_row = []
        for col in row:
            new_row.append(REV_TILES_MAP[col])
        # add side borders
        new_row.insert(0, "w")
        new_row.append("w")
        level.append(new_row)
    top_bottom_border = ["w"] * len(level[0])
    level.insert(0, top_bottom_border)
    level.append(top_bottom_border)

    level_as_str = []
    for row in level:
        level_as_str.append("".join(row) + "\n")

    with open(dir, "w") as f:
        for row in level_as_str:
            f.write(row)
    f.close()


class Linear(tf.keras.layers.Layer):
    def __init__(self, units, name, **kwargs):
        print(**kwargs)
        super().__init__(**kwargs)
        self.units = units
        # self.name = name
        self.i = name

    def build(self, input_shape):
        self.i += 1
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name=f"weight_{self.i}",
        )
        self.i += 1
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            name=f"b_{self.i}",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config().copy()
        config.update({"units": self.units, "name": self.name})
        return config


# FOR SCALAR
class MLPCountingBlockScalar(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.linear_1 = Linear(32, 5)
        self.linear_2 = Linear(32, 5)
        self.dense_1 = Dense(1, activation="softmax")
        self.loss = tf.keras.losses.MeanSquaredError(name="counting_head_loss")

    def call(self, inputs):
        x = Concatenate()(inputs[:2])
        x = self.linear_1(x)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        output = self.dense_1(x)

        # cat_cross_entry = self.loss.update_state(inputs[2], output)
        # self.add_loss(cat_cross_entry)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"name": self.name})
        return config


# FOR SIGNED
class MLPCountingBlockSigned(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.linear_1 = Linear(32, 5)
        self.linear_2 = Linear(32, 5)
        self.dense_1 = Dense(3, activation="softmax")
        self.loss = tf.keras.losses.CategoricalCrossentropy(name="counting_head_loss")

    def call(self, inputs):
        x = Concatenate()(inputs[:2])
        x = self.linear_1(x)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        output = self.dense_1(x)

        cat_cross_entry = self.loss.update_state(inputs[2], output)
        self.add_loss(cat_cross_entry)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"name": self.name})
        return config


def int_map_from_onehot(map, obs_size):
    int_map = []
    for y in map:
        for x in y:
            int_map.append(list(x).index(1))

    return np.array(int_map).reshape((obs_size, obs_size))


def transform_narrow(obs, x, y, return_onehot=True, transform=True):
    pad = 11
    pad_value = 1
    size = 22
    map = obs  # obs is int
    # View Centering
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y : y + size, x : x + size]
    obs = cropped

    if return_onehot:
        obs = np.eye(8)[obs]
        if transform:
            new_obs = []
            for i in range(22):
                for j in range(22):
                    for z in range(8):
                        new_obs.append(obs[i][j][z])
            return new_obs
    return obs


def int_map_to_onehot(int_map):
    new_map = []
    for row_i in range(len(int_map)):
        new_row = []
        for col_i in range(len(int_map[0])):
            new_tile = [0] * 8
            new_tile[int_map[row_i][col_i]] = 1
            new_row.append(np.array(new_tile))
        new_map.append(np.array(new_row))
    return np.array(new_map)


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

    # Remove the border
    truncated_level = level[1 : len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1 : len(row) - 1]
        level.append(new_row)
    return level


def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    random_map = list(random_map[0])
    for i in range(len(goal)):
        for j in range(len(goal[0])):
            for k in range(8):
                if random_map[i][j][k] != goal[i][j][k]:
                    hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))


def transform_narrow(obs, x, y, obs_size=9, return_onehot=True, transform=True):
    pad = 22 - (22 - obs_size)
    pad_value = 1
    size = 22
    map = obs  # obs is int
    # View Centering
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y : y + size, x : x + size]
    obs = cropped

    # if return_onehot:
    #     obs = np.eye(8)[obs]
    #     if transform:
    #         new_obs = []
    #         for i in range(22):
    #             for j in range(22):
    #                 for z in range(8):
    #                     new_obs.append(obs[i][j][z])
    #         return new_obs
    return obs


# Converts from string[][] to 2d int[][]
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map


map_num_to_oh_dict = {}
root_dir = "playable_maps"


def infer(game, representation, model_abs_path, obs_size, model_num, mode, inference_results_path, combo_id, sample_id, **kwargs):
    """
    - max_trials: The number of trials per evaluation.
    - infer_kwargs: Args to pass to the environment.
    """
    env_name = "{}-{}-v0".format(game, representation)
    kwargs["cropped_size"] = obs_size
    kwargs["render"] = False

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    obs = env.reset()
    dones = False
    action_mode = "greedy"
    prob = ZeldaProblem()
    idx = 0

    agent = keras.models.load_model(
        model_abs_path,
    )
    start_level_str = ""

    if mode == "controllable":
        average_enemies = 2
        average_nearest_enemy = 4
        average_path_length = 21
        enemies_tests = [
            (min(i, 10), average_nearest_enemy, average_path_length)
            for i in range(1, 11)
        ]
        nearest_enemy_tests = [
            (average_enemies, min(i, 20), average_path_length) for i in range(1, 21)
        ]
        path_length_tests = [
            (average_enemies, average_nearest_enemy, i) for i in range(20, 41)
        ]
        results_dict = {"chg_pct":[], "enemies_input":[], "nearest_enemy_input": [], "path_length_input": [], "start_map":[], "final_map":[], "result":[], "target_enemies":[], "target_nearest_enemy":[], "target_path_length":[], "comboID":[], "sampleID":[], "model_num": []}

        for target_enemies, target_nearest_enemy, target_path_length in (
            enemies_tests + nearest_enemy_tests + path_length_tests
        ):
            idx += 1
            enemies, nearest_enemy, path_length = (
                target_enemies,
                target_nearest_enemy,
                target_path_length,
            )

            for i in range(kwargs.get("trials", 1)):
                j = 0
                is_start_map = True
                while not dones:
                    j += 1
                    print(f"j: {j}")
                    if is_start_map:
                        start_map = get_string_map(
                                int_map_from_onehot(obs[0], obs_size),
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
                        start_level_str = ""

                        for row in start_map:
                            for col in row:
                                start_level_str += REV_TILES_MAP[col]

                        

                        

                        is_start_map = False

                    try:
                        map_stats = prob.get_stats(
                            get_string_map(
                                int_map_from_onehot(obs[0], obs_size),
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
                        )
                        regions_input = map_stats["regions"]
                        enemies_input = map_stats["enemies"]
                        nearest_enemy_input = map_stats["nearest-enemy"]
                        path_length_input = map_stats["path-length"]
                        

                        # Enemies
                        if enemies_input > target_enemies:
                            enemies_diff = [1, 0, 0]
                        elif enemies_input == target_enemies:
                            enemies_diff = [0, 1, 0]
                        else:
                            enemies_diff = [0, 0, 1]

                        # Nearest enemy
                        if nearest_enemy_input > target_nearest_enemy:
                            nearest_diff = [1, 0, 0]
                        elif nearest_enemy_input == target_nearest_enemy:
                            nearest_diff = [0, 1, 0]
                        else:
                            nearest_diff = [0, 0, 1]

                        # Path length
                        if path_length_input > target_path_length:
                            path_length_diff = [1, 0, 0]
                        elif path_length_input == target_path_length:
                            path_length_diff = [0, 1, 0]
                        else:
                            path_length_diff = [0, 0, 1]

                        signed_inputs = enemies_diff + nearest_diff + path_length_diff
                        prediction = agent.predict(
                            x=[np.array([obs[0]]),
                                np.array([signed_inputs])],
                                steps=1
                            )
                        
                        if action_mode == "greedy":

                            # prediction = agent.predict(x={'input_1':np.array([obs[0]]), 'input_2': stats}, steps=1)
                            action = np.argmax(prediction) + 1
                        else:

                            sum_pred_probs = sum(prediction)
                            a = D(1 / sum_pred_probs)
                            pred_probs_scaled = [
                                D(float(e)) * a for e in prediction
                            ]
                            action = np.random.choice(8, 1, p=pred_probs_scaled) + 1
                    except Exception:
                        map_stats = prob.get_stats(
                            get_string_map(
                                int_map_from_onehot(obs[0], obs_size),
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
                        )
                        enemies_input = map_stats["enemies"]
                        nearest_enemy_input = map_stats["nearest-enemy"]
                        path_length_input = map_stats["path-length"]

                        # Enemies
                        if enemies_input > target_enemies:
                            enemies_diff = [1, 0, 0]
                        elif enemies_input == target_enemies:
                            enemies_diff = [0, 1, 0]
                        else:
                            enemies_diff = [0, 0, 1]

                        # Nearest enemy
                        if nearest_enemy_input > target_nearest_enemy:
                            nearest_diff = [1, 0, 0]
                        elif nearest_enemy_input == target_nearest_enemy:
                            nearest_diff = [0, 1, 0]
                        else:
                            nearest_diff = [0, 0, 1]

                        # Path length
                        if path_length_input > target_path_length:
                            path_length_diff = [1, 0, 0]
                        elif path_length_input == target_path_length:
                            path_length_diff = [0, 1, 0]
                        else:
                            path_length_diff = [0, 0, 1]

                        signed_inputs = enemies_diff + nearest_diff + path_length_diff
                        prediction = agent.predict(
                            x=[np.array([obs[0]]),
                                np.array([signed_inputs])],
                                steps=1
                            )

                        action = np.argmax(prediction) + 1

                    obs, _, dones, info = env.step([action])
                    map_stats = prob.get_stats(info[0]["final_map"])
                    img = prob.render(info[0]["final_map"])

                    if info[0]["solved"]:
                        if (abs(target_enemies - map_stats["enemies"]) <= 3 and abs(target_nearest_enemy - map_stats["nearest-enemy"]) <= 4 and abs(target_path_length - map_stats["path-length"]) <= 6):
                            results_dict["result"].append("sucess")
                        else:
                            results_dict["result"].append("partial")

                        enemies_input = map_stats["enemies"]
                        nearest_enemy_input = map_stats["nearest-enemy"]
                        path_length_input = map_stats["path-length"]

                        results_dict["enemies_input"].append(enemies_input)
                        results_dict["nearest_enemy_input"].append(nearest_enemy_input)
                        results_dict["path_length_input"].append(path_length_input)

                        final_map = info[0]["final_map"]
                        level_str = ""

                        for row in final_map:
                            for col in row:
                                level_str += REV_TILES_MAP[col]

                        results_dict["final_map"].append(level_str)
                        results_dict["comboID"].append(combo_id)
                        results_dict["sampleID"].append(sample_id)
                        results_dict["model_num"].append(model_num)
                        results_dict["start_map"].append(start_level_str)
                        results_dict["target_enemies"].append(target_enemies)
                        results_dict["target_nearest_enemy"].append(target_nearest_enemy)
                        results_dict["target_path_length"].append(target_path_length)
                        results_dict["chg_pct"].append(kwargs["change_percentage"])

                        dones = True

                    elif dones:
                        pass
                        
                    
                dones = False
                obs = env.reset()
                obs = env.reset()

        results_df = pd.DataFrame(results_dict).to_csv(inference_results_path, mode="a", header=not os.path.exists(inference_results_path), index=False)

    elif mode == "non_controllable":
        results_dict = {"chg_pct":[], "start_map":[], "final_map":[], "result":[], "comboID":[], "sampleID":[], "model_num": []}

        idx += 1      
        for i in range(kwargs.get("trials", 1)):
            j = 0
            is_start_map = True
            while not dones:
                j += 1
                print(f"j: {j}")
                if is_start_map:
                    start_map = get_string_map(
                            int_map_from_onehot(obs[0], obs_size),
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
                    level_str = ""

                    for row in start_map:
                        for col in row:
                            start_level_str += REV_TILES_MAP[col]


                    is_start_map = False

                try:
                    prediction = agent.predict(np.array([obs[0]]))
                    if action_mode == "greedy":
                        action = np.argmax(prediction) + 1
                    else:

                        sum_pred_probs = sum(prediction)
                        a = D(1 / sum_pred_probs)
                        pred_probs_scaled = [
                            D(float(e)) * a for e in prediction
                        ]
                        action = np.random.choice(8, 1, p=pred_probs_scaled) + 1
                except Exception:
                    
                    print(f"np.array([obs[0]]): {np.array([obs[0]]).shape}")
                    prediction = agent.predict(np.array([obs[0]]))
                    action = np.argmax(prediction) + 1

                obs, _, dones, info = env.step([action])
                map_stats = prob.get_stats(info[0]["final_map"])
                img = prob.render(info[0]["final_map"])
                # print(f'info[0]["final_map"]: {info[0]["final_map"]}')

                if info[0]["solved"]:
                    results_dict["result"].append("sucess")

                    final_map = info[0]["final_map"]
                    level_str = ""

                    for row in final_map:
                        for col in row:
                            level_str += REV_TILES_MAP[col]

                    results_dict["final_map"].append(level_str)
                    results_dict["comboID"].append(combo_id)
                    results_dict["sampleID"].append(sample_id)
                    results_dict["model_num"].append(model_num)
                    results_dict["start_map"].append(start_level_str)

                    dones = True

                elif dones:
                    pass
                
            dones = False
            obs = env.reset()
            obs = env.reset()

        results_df = pd.DataFrame(results_dict).to_csv(inference_results_path, mode="a", header=not os.path.exists(inference_results_path), index=False)


################################## MAIN ########################################
game = "zelda"
representation = "narrow"
model_path = "models/{}/{}/model_1.pkl".format(game, representation)
kwargs = {
    "change_percentage": 1.0,  # 0.4,
    "trials": 5,  # This was either 50 or 200 for the ID inference, For OOD inference this was 20
    "verbose": True,
}


def inference_zelda(combo_id, sweep_params, mode, username):
    root_path = f"/scratch/{username}/overlay/sweep_testing_pod/data/zelda/{mode}" # For testing: 
    obs_size, goal_set_size, trajectory_length, training_dataset_size = sweep_params

    combo_id_path = f"{root_path}/comboID_{combo_id}"
    
    for sample_id in range(1, 4):
        sample_id_path = f"{combo_id_path}/sampleID_{sample_id}"
        model_count = sample_id
        model_path = f"{sample_id_path}/models/{model_count}.h5"
        inference_results_path = f"{sample_id_path}/inference_results.csv"

        if os.path.isfile(f"{sample_id_path}/results.done"):
            continue

        for model_count in range(1,4):
            for chg_pct in range(1, 11):
                kwargs["change_percentage"] = chg_pct / 10.0
                infer(game, representation, model_path, obs_size, model_count, mode, inference_results_path, combo_id, sample_id, **kwargs)

            with open(f"{sample_id_path}/results.done", "w") as f:
                f.write("")


