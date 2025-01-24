"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils

# 歩行者の状態を管理
class PedState:
    """歩行者の位置、速度、目標地点、およびグループ情報を追跡"""

    def __init__(self, state, waypoints, types, groups, obs_area, sgn_area, scene_configs):
        """
        state: 初期状態（各歩行者の位置、速度、目標地点など）を保持するnumpy配列。
        types: 各歩行者のタイプ（0: 成人、1: 老人、2: 子供）。
        groups: 歩行者のグループ情報（例: [[0, 1], [2]]）。
        scene_configs: タイプ別のシーン設定を格納した辞書。
        """
        self.waypoints = waypoints
        self.types = types
        self.obs_area = obs_area
        self.sgn_area = sgn_area
        
        self.scene_configs = scene_configs

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []

        # TODO:タイプによるmax_speed_multiplierの変化の実装
        self.max_speed_multiplier = self.scene_configs.get('0', {}).get('scene', {}).get('max_speed_multiplier', 1.3)

        self.update(state, groups)

        self.original_goals = np.copy(self.state[:, 4:6])
        self.temporary_goals = np.full_like(self.original_goals, np.nan)  # 一時ゴール（初期値は無効）

    def update(self, state, groups):
        # タイプごとの初期化処理
        self.state = state
        self.groups = groups
        
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
        
        self.max_speeds = self.get_max_speeds()
        self.ped_states.append(self._state.copy())  # 状態履歴（ped_states）に現在の状態を追加して保存
        
    def get_states(self):
        """現在までの全ての状態履歴（ped_states）とグループ履歴（group_states）を取得"""
        return np.stack(self.ped_states), self.group_states
    
    def get_agent_radius(self, ped_type):
        return self.scene_configs[str(ped_type)]["scene"]["agent_radius"]
    
    def get_max_speeds(self):
        max_speeds = np.zeros((self.size(),))
        for i, ped_type in enumerate(self.types):  # 各歩行者の種類に基づいて計算
            config = self.scene_configs.get(str(ped_type))
            max_speed_multiplier = config.get("scene").get("max_speed_multiplier")
            max_speeds[i] = max_speed_multiplier * self.initial_speeds[i]
        return max_speeds

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
        speed = stateutils.speeds(self.state)
        return speed

    def step(self, force, groups=None):
        """外部から与えられる力（force）に基づいて、歩行者の次の位置と速度を計算"""
        desired_velocity = np.zeros_like(self.vel())  # 歩行者ごとの速度を保持
        step_width = np.zeros((self.size(),))

        for i, ped_type in enumerate(self.types):  # 各歩行者の種類に基づいて計算
            # 種類ごとの設定を取得
            pos = self.pos()[i]
            config = self.scene_configs.get(str(ped_type))
            step_width[i] = config.get("scene").get("step_width")

            # 特定エリア内での力調整
            if self.obs_area is not None:
                step_width[i] = self.obstruction_area(step_width[i], pos, ped_type, self.obs_area)

            # 理想速度を計算
            velocity = self.vel()[i] + step_width[i] * force[i]
            
            desired_velocity[i] = velocity

        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # 目標到達時の速度制限
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]
        
        step_width_expanded = step_width[:, None]
        # 状態更新
        next_state = self.state
        next_state[:, 0:2] += desired_velocity * step_width_expanded
        next_state[:, 2:4] = desired_velocity

        # ゴール到達判定と更新
        if self.waypoints is not None:
            for i, waypoints in enumerate(self.waypoints):
                # ゴールに到達したか判定
                current_pos = next_state[i, 0:2]
                current_goal = next_state[i, 4:6]
                distance_to_goal = np.linalg.norm(current_pos - current_goal)

                if distance_to_goal < 0.5:  # ゴールに到達した場合
                    if waypoints:  # 中継地点が残っている場合
                        next_state[i, 4:6] = waypoints.pop(0)  # 次の目的地を設定
                    else:  # 中継地点がない場合、スタート地点に戻す
                        next_state[i, 0:2] = self.original_goals[i]  # スタート地点に戻す
                        next_state[i, 4:6] = self.original_goals[i]  # 次のゴールをスタート地点に設定
            
        next_groups = self.groups
        if groups is not None:
            next_groups = groups
        
        self.update(next_state, next_groups)

    # 特定エリアによって、速度を制限
    def obstruction_area(self, step_width, pos, ped_type, obs_area):
        reduction_steps = [0.2, 0.1, 0.15]
        step_width = step_width
        x, y = pos
        for i, area in enumerate(obs_area):
            x_min, x_max, y_min, y_max, scalr = area

            if x_min <= x <= x_max and y_min <= y <= y_max:
                step_width *= scalr
                break

        return step_width
    
    def is_temporary_goal(self, idx):
        """一時ゴールが設定されているか確認"""
        return not np.isnan(self.temporary_goals[idx, 0])

    def set_temporary_goal(self, idx, goal):
        """一時ゴールを設定"""
        self.temporary_goals[idx] = goal
        self.state[idx, 4:6] = goal  # ゴールを一時的に変更

    def restore_original_goal(self, idx):
        """元のゴールに戻す"""
        self.state[idx, 4:6] = self.original_goals[idx]
        self.temporary_goals[idx] = np.nan  # 一時ゴールを無効化

    def reached_temporary_goal(self, idx):
        """一時ゴールに到達しているか確認"""
        goal = self.temporary_goals[idx]
        if np.isnan(goal[0]):
            return False  # 一時ゴールが設定されていない場合
        return np.linalg.norm(self.pos()[idx] - goal) < 0.5  # 到達判定（距離が0.5未満）
    
    def initial_speeds(self):
        return stateutils.speeds(self.ped_states[0])

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

    def __init__(self, obstacles, resolution):
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
