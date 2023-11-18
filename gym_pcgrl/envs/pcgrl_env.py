from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
from gym import spaces
import PIL

"""
The PCGRL GYM Environment
"""


class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """

    def __init__(self, prob="binary", rep="narrow"):
        self.prob = PROBLEMS[prob]()
        self.rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self.prob._width * self.prob._height), 1)
        self._max_iterations = self._max_changes * self.prob._width * self.prob._height
        self._heatmap = np.zeros((self.prob._height, self.prob._width))

        self.seed()
        self.viewer = None

        self.action_space = self.rep.get_action_space(
            self.prob._width, self.prob._height, self.get_num_tiles()
        )
        self.observation_space = self.rep.get_observation_space(
            self.prob._width, self.prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces["heatmap"] = spaces.Box(
            low=0,
            high=self._max_changes,
            dtype=np.uint8,
            shape=(self.prob._height, self.prob._width),
        )

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """

    def seed(self, seed=None):
        seed = self.rep.seed(seed)
        self.prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """

    def reset(self):
        self._changes = 0
        self._iteration = 0
        self.rep.reset(
            self.prob._width,
            self.prob._height,
            get_int_prob(self.prob._prob, self.prob.get_tile_types()),
        )
        self._rep_stats = self.prob.get_stats(
            get_string_map(self.rep._map, self.prob.get_tile_types())
        )
        self.prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self.prob._height, self.prob._width))

        observation = self.rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """

    def get_border_tile(self):
        return self.prob.get_tile_types().index(self.prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """

    def get_num_tiles(self):
        return len(self.prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """

    def adjust_param(self, **kwargs):
        if "change_percentage" in kwargs:
            percentage = min(1, max(0, kwargs.get("change_percentage")))
            self._max_changes = max(
                int(percentage * self.prob._width * self.prob._height), 1
            )
        self._max_iterations = self._max_changes * self.prob._width * self.prob._height
        self.prob.adjust_param(**kwargs)
        self.rep.adjust_param(**kwargs)
        self.action_space = self.rep.get_action_space(
            self.prob._width, self.prob._height, self.get_num_tiles()
        )
        self.observation_space = self.rep.get_observation_space(
            self.prob._width, self.prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces["heatmap"] = spaces.Box(
            low=0,
            high=self._max_changes,
            dtype=np.uint8,
            shape=(self.prob._height, self.prob._width),
        )

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """

    # def step(self, action):
    #     self._iteration += 1
    #     # save copy of the old stats to calculate the reward
    #     old_stats = self._rep_stats
    #     # update the current state to the new state based on the taken action
    #     change, x, y = self.rep.update(action)
    #     if change > 0:
    #         self._changes += change
    #         self._heatmap[y][x] += 1.0
    #         self._rep_stats = self.prob.get_stats(
    #             get_string_map(self.rep._map, self.prob.get_tile_types())
    #         )
    #     # calculate the values
    #     observation = self.rep.get_observation()
    #     observation["heatmap"] = self._heatmap.copy()
    #     reward = self.prob.get_reward(self._rep_stats, old_stats)
    #     done = (
    #         self.prob.get_episode_over(self._rep_stats, old_stats)
    #         or self._changes >= self._max_changes
    #         or self._iteration >= self._max_iterations
    #     )
    #     # similarity reward at the end of episode
    #     if done:
    #         sim_reward = self.prob.get_similarity_rewrad(self._rep_stats)
    #         reward = -sim_reward

    #     info = self.prob.get_debug_info(self._rep_stats, old_stats)
    #     info["iterations"] = self._iteration
    #     info["changes"] = self._changes
    #     info["max_iterations"] = self._max_iterations
    #     info["max_changes"] = self._max_changes
    #     info["solved"] = self.prob.get_episode_over(self._rep_stats, old_stats)
    #     info["final_map"] = get_string_map(self.rep._map, self.prob.get_tile_types())
    #     info["x"] = self.rep._x
    #     info["y"] = self.rep._y
    #     # return the values
    #     return observation, reward, done, info

    def step(self, action):
        self._iteration += 1
        # save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self.rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self.prob.get_stats(
                get_string_map(self.rep._map, self.prob.get_tile_types())
            )
        # calculate the values
        observation = self.rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self.prob.get_reward(self._rep_stats, old_stats)
        done = (
            self.prob.get_episode_over(self._rep_stats, old_stats)
            or self._changes >= self._max_changes
            or self._iteration >= self._max_iterations
        )
        info = self.prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        info["solved"] = self.prob.get_episode_over(self._rep_stats, old_stats)
        info["final_map"] = get_string_map(self.rep._map, self.prob.get_tile_types())
        info["x"] = self.rep._x
        info["y"] = self.rep._y
        # return the values
        return observation, reward, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """

    def render(self, mode="human"):
        tile_size = 16
        img = self.prob.render(
            get_string_map(self.rep._map, self.prob.get_tile_types())
        )
        img = self.rep.render(
            img, self.prob._tile_size, self.prob._border_size
        ).convert("RGB")
        if mode == "rgb_array":
            return img
        # elif mode == "human":
        #     from gym.envs.classic_control import rendering

        #     if self.viewer is None:
        #         self.viewer = rendering.SimpleImageViewer()
        #     if not hasattr(img, "shape"):
        #         img = np.array(img)
        #     self.viewer.imshow(img)
        #     return self.viewer.isopen

    """
    Close the environment
    """

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
