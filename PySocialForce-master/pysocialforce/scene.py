"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils

# 歩行者の状態を管理
class PedState:
    """歩行者の位置、速度、目標地点、およびグループ情報を追跡"""
    
    def __init__(self, state, types, groups, scene_configs):
        self.types = types
        self.scene_configs = scene_configs

        self.ped_states = []
        self.group_states = []

        # タイプごとの初期化処理
        self.state = state
        self.groups = groups
        self.agent_settings = self.initialize_agent_settings()

    def initialize_agent_settings(self):
        """タイプごとにシーン設定を適用"""
        settings = []
        for i, ped_type in enumerate(self.types):
            config = self.scene_configs[ped_type]
            settings.append({
                "agent_radius": config("agent_radius", 0.35),
                "step_width": config("step_width", 1.0),
                "max_speed_multiplier": config("max_speed_multiplier", 1.3),
                "tau": config("tau", 0.5)
            })
        return settings

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        """タイプ別に `tau` を拡張"""
        if state.shape[1] < 7:
            taus = np.full((state.shape[0], 1), 0.5)    # stateの行列にtauの列を追加する   
            #taus = np.expand_dims(np.array(self.default_tau), -1)

            self._state = np.concatenate((state, taus), axis=-1)
        else:
            self._state = state
        self.ped_states.append(self._state.copy())
        
    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def tau(self):
        return self.state[:, 6:7]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """Move peds according to forces"""
        # desired velocity
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        # update state
        next_state = self.state
        next_state[:, 0:2] += desired_velocity * self.step_width
        next_state[:, 2:4] = desired_velocity
        next_groups = self.groups
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups)

    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1

# 環境状態、障害物を管理
class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)
