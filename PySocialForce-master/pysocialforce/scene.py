"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils

# 歩行者の状態を管理
class PedState:
    """歩行者の位置、速度、目標地点、およびグループ情報を追跡"""

    def __init__(self, state, types, groups, scene_configs):
        """
        state: 初期状態（各歩行者の位置、速度、目標地点など）を保持するnumpy配列。
        types: 各歩行者のタイプ（0: 成人、1: 老人、2: 子供）。
        groups: 歩行者のグループ情報（例: [[0, 1], [2]]）。
        scene_configs: タイプ別のシーン設定を格納した辞書。
        """
        self.types = types
        self.scene_configs = scene_configs

        agent_settings = self.initialize_agent_settings()  # 各歩行者の設定を適応

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []
        self.update(state, groups, agent_settings)

    def initialize_agent_settings(self):
        """タイプごとにシーン設定を適用"""
        settings = []
        for i, ped_type in enumerate(self.types):
            config = self.scene_configs[ped_type]
            self.type_tau = config("tau", 0.5)
            self.step_width = config("step_width", 1.0)
            self.agent_radius = config("agent_radius", 0.35)
            self.max_speed_multiplier = config("max_speed_multiplier", 1.3)
            
            settings.append({
                self.type_tau,
                self.step_width,
                self.agent_radius,
                self.max_speed_multiplier,
            })  # 辞書形式でリストに保存
        return settings

    def update(self, state, groups, agent_setting):
        # タイプごとの初期化処理
        self.state = state
        self.groups = groups
        self.agent_settings = agent_setting  # 各歩行者の設定を適応

    @property
    def state(self):
        """歩行者の状態を取得・設定"""
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
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds

        self.ped_states.append(self._state.copy())  # 状態履歴（ped_states）に現在の状態を追加して保存
        
    def get_states(self):
        """現在までの全ての状態履歴（ped_states）とグループ履歴（group_states）を取得"""
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        """各歩行者の現在の位置（px, py）を返す"""
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        """各歩行者の現在の速度（vx, vy）を返す"""
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        """各歩行者の目標地点（gx, gy）を返す"""
        return self.state[:, 4:6]

    def tau(self):
        """各歩行者のリラクゼーション時間（tau）を返す"""
        return self.state[:, 6:7]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """外部から与えられる力（force）に基づいて、歩行者の次の位置と速度を計算"""
        desired_velocity = self.vel() + self.step_width * force  # 理想速度の計算
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)  # 速度の制限(最大速度を超えないようcapped_velocityを使用して調整)
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]   # 目標到達時の速度制限

        # 状態更新
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
        """各歩行者が目標地点に向かう方向ベクトルを返す"""
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """速度が最大速度を超えないようにスケーリング"""
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

    def __init__(self, obstacles, resolution=50):
        self.resolution = resolution   # 線分のサンプリング分解能
        self.obstacles = obstacles  # 障害物（線分）のリスト

    @property
    def obstacles(self) -> List[np.ndarray]:
        """障害物のリストを取得・設定"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """各障害物線分を細分化し、サンプル点のリストとして保存"""
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
