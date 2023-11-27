import os

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem


PROJECT_ROOT = os.environ["PROJECT_ROOT"]

# ###################   ZELDA CONSTANTS   ##############################

ZELDA_DATA_ROOT = os.environ["ZELDA_DATA_ROOT"]
ZELDA_GOAL_MAPS_ROOT = os.environ["ZELDA_GOAL_MAPS_ROOT"]
DOMAIN_VARS_ZELDA = {
    "tiles_map": {
        "g": "door",
        "+": "key",
        "A": "player",
        "1": "bat",
        "2": "spider",
        "3": "scorpion",
        "w": "solid",
        ".": "empty",
    },
    "int_map": {
        "empty": 0,
        "solid": 1,
        "player": 2,
        "key": 3,
        "door": 4,
        "bat": 5,
        "scorpion": 6,
        "spider": 7,
    },
    "char_map": {
        "door": "a",
        "key": "b",
        "player": "c",
        "bat": "d",
        "spider": "e",
        "scorpion": "f",
        "solid": "g",
        "empty": "h",
    },
    "goal_maps_filepath": "goal_maps/zelda/zelda_lvl{}.txt",
    "env_y": 7,
    "env_x": 11,
    "env_z": None,
    "action_space_size": 8,
    "action_pronbabilities_map": {
        0: 0.58,
        1: 0.3,
        2: 0.02,
        3: 0.02,
        4: 0.02,
        5: 0.02,
        6: 0.02,
        7: 0.02,
    },
    "gym_env_name": "zelda-narrow-v0",
    "prob": ZeldaProblem,
    "sweep_params": [
        ("obs_sizes", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
        ("goal_set_sizes", [1, 10, 20, 30, 40, 50]),
        (
            "trajectory_lengths",
            [
                int(7 * 11 * 0.02),
                int(7 * 11 * 0.1),
                int(7 * 11 * 0.2),
                int(7 * 11 * 0.3),
                int(7 * 11 * 0.4),
                int(7 * 11 * 0.5),
                int(7 * 11 * 0.6),
                int(7 * 11 * 0.7),
                int(7 * 11 * 0.8),
                int(7 * 11 * 0.9),
                int(7 * 11 * 1.0),
            ],
        ),
        (
            "training_dataset_sizes",
            [
                100_000,
                200_000,
                300_000,
                400_000,
                500_000,
                600_000,
                700_000,
                800_000,
                900_000,
                1_000_000,
            ],
        ),
    ],
    "trajectories_to_generate": [
        ("obs_size", [21]),
        ("goal_set_sizes", [1, 10, 20, 30, 40, 50]),
        (
            "trajectory_lengths",
            [
                int(7 * 11 * 0.02),
                int(7 * 11 * 0.1),
                int(7 * 11 * 0.2),
                int(7 * 11 * 0.3),
                int(7 * 11 * 0.4),
                int(7 * 11 * 0.5),
                int(7 * 11 * 0.6),
                int(7 * 11 * 0.7),
                int(7 * 11 * 0.8),
                int(7 * 11 * 0.9),
                int(7 * 11 * 1.0),
            ],
        ),
        ("training_dataset_sizes", [400_000]),
    ],
}
TILES_MAP_ZELDA = DOMAIN_VARS_ZELDA["tiles_map"]
INT_MAP_ZELDA = DOMAIN_VARS_ZELDA["int_map"]