"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod

import numpy as np

from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config, stateutils, logger

def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self): # initより先に呼び出す
        super().__init__()
        self.scene = None  # シーン情報（歩行者や障害物の情報など）
        self.peds = None  # 歩行者情報
        self.default_config = Config()
        self.config = {}  # 種類ごとの設定を保持
    
    def init(self, scene, config, factor_list):
        """Load config and scene"""
        # 設定ファイルから各力のパラメータを取得
        self.scene = scene
        self.peds = self.scene.peds
        self.config = config
        self.factor_list = factor_list
        
    @abstractmethod
    def _get_force(self) -> np.ndarray:
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    def get_force(self, debug=False):
        force = self._get_force()
        if debug:
            logger.debug(f"{camel_to_snake(type(self).__name__)}:\n {repr(force)}")
        return force


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def _initialize(self):
        """初回呼び出し時に初期化"""
        self.goal_factor = np.zeros((self.peds.size(),))
        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.goal_factor[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)
        self.initialized = True

    def _get_force(self):
        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行

        F0 = (
            1.0
            / self.peds.tau()
            * (
                np.expand_dims(self.peds.initial_speeds, -1) * self.peds.desired_directions()
                - self.peds.vel()
            )
        )
        return F0 * self.goal_factor


class PedRepulsiveForce(Force):
    """Ped to ped repulsive force"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def _initialize(self):
        """初回呼び出し時に初期化"""
        self.v0 = np.zeros((self.peds.size(),))
        self.sigma = np.zeros((self.peds.size(),))
        self.fov_phi = np.zeros((self.peds.size(),))
        self.fov_factor = np.zeros((self.peds.size(),))
        self.step_width = np.zeros((self.peds.size(),))
        self.ped_factors = np.zeros((self.peds.size(),))

        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.v0[i] = config.get("v0", 2.1)
            self.sigma[i] = config.get("sigma", 0.3)
            self.fov_phi[i] = config.get("fov_phi", 100.0)
            self.fov_factor[i] = config.get("fov_factor", 0.5)
            self.step_width[i] = self.peds.scene_configs.get(str(ped_type), {}).get("scene", {}).get("step_width")
            self.ped_factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        self.initialized = True

    def _get_force(self):
        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行

        potential_func = PedPedPotential(self.step_width, self.v0, self.sigma)
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)

        fov = FieldOfView(self.fov_phi, self.fov_factor)
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)

        F_ab = w * f_ab

        forces = np.sum(F_ab, axis=1) * self.ped_factors[:, None]

        return forces


class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def _initialize(self):
        """初回呼び出し時に初期化"""
        self.u0 = np.zeros((self.peds.size(),))
        self.r = np.zeros((self.peds.size(),))
        self.spa_factors = np.zeros((self.peds.size(),))

        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.u0[i] = config.get("u0", 10)
            self.r[i] = config.get("r", 0.2)
            self.spa_factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        self.initialized = True

    def _get_force(self):
        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行

        if self.scene.get_obstacles() is None:
            F_aB = np.zeros((self.peds.size(), 0, 2))
        else:
            potential_func = PedSpacePotential(self.scene.get_obstacles(), self.u0, self.r)
            F_aB = -1.0 * potential_func.grad_r_aB(self.peds.state)

        forces = np.sum(F_aB, axis=1) * self.spa_factors[:, None]

        return forces


class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""
    

    def _get_force(self):
        factors = np.zeros((self.peds.size(),))
        forces = np.zeros((self.peds.size(), 2))

        for i, ped_type in enumerate(self.peds.types):
            factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                vectors, norms = stateutils.normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        
        factors_expanded = factors[:, None]
        forces = forces * factors_expanded

        return forces


class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def _get_force(self):
        factors = np.zeros((self.peds.size(),))
        forces = np.zeros((self.peds.size(), 2))

        for i, ped_type in enumerate(self.peds.types):
            factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T

        factors_expanded = factors[:, None]
        forces = forces * factors_expanded

        return forces


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))

        config = self.default_config.sub_config(camel_to_snake(type(self).__name__))
        threshold = config("threshold", 0.5)
        factor = config("factor", 1.0)

        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        forces = forces * factor

        return forces


class GroupGazeForce(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))

        config = self.default_config.sub_config(camel_to_snake(type(self).__name__))
        vision_angle = config("fov_phi", 100.0)
        factor = config("factor", 4.0)
        directions, _ = stateutils.desired_directions(self.peds.state)

        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, _ = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                com_angles = np.degrees(np.arccos(element_prod))
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -rotation.reshape(-1, 1) * member_directions
                forces[group, :] += force

        return forces * factor


class GroupGazeForceAlt(Force):
    """Group gaze force"""
    
    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))

        config = self.default_config.sub_config(camel_to_snake(type(self).__name__))
        factor = config("factor", 4.0)
        directions, dist = stateutils.desired_directions(self.peds.state)
        
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                member_dist = dist[group]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, com_dist = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                force = (
                    com_dist.reshape(-1, 1)
                    * element_prod.reshape(-1, 1)
                    / member_dist.reshape(-1, 1)
                    * member_directions
                )
                forces[group, :] += force

        return forces * factor


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint."""

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.relaxation_time = None
        self.goal_threshold = None
        self.des_factors = None

    def _initialize(self):
        """初回呼び出し時に初期化"""
        self.relaxation_time = np.zeros((self.peds.size(),))
        self.goal_threshold = np.zeros((self.peds.size(),))
        self.des_factors = np.zeros((self.peds.size(),))

        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.relaxation_time[i] = config.get("relaxation_time", 0.5)
            self.goal_threshold[i] = config.get("goal_threshold", 0.2)
            self.des_factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        self.initialized = True  # 初期化フラグをセット

    def _get_force(self):
        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行

        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()

        direction, dist = stateutils.normalize(goal - pos)
        forces = np.zeros((self.peds.size(), 2))

        for i in range(self.peds.size()):
            if dist[i] > self.goal_threshold[i]:
                force = (direction[i] * self.peds.max_speeds[i] - vel[i]) / self.relaxation_time[i]
            else:
                force = -1.0 * vel[i] / self.relaxation_time[i]

            forces[i] = force * self.des_factors[i]

        return forces

class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents."""

    def __init__(self):
        super().__init__()
        self.initialized = False  # 初期化フラグ
        self.lambda_importance = None
        self.gamma = None
        self.n = None
    
    def _initialize(self):
        # 配列を初期化
        self.lambda_importance = np.zeros((self.peds.size(),))
        self.gamma = np.zeros((self.peds.size(),))
        self.n = np.zeros((self.peds.size(),))
        self.n_prime = np.zeros((self.peds.size(),))
        self.soc_factors = np.zeros((self.peds.size(),))

        # 種類に応じた設定を取得
        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.lambda_importance[i] = config.get("lambda_importance", 2.0)
            self.gamma[i] = config.get("gamma", 0.35)
            self.n[i] = config.get("n", 2)
            self.n_prime[i] = config.get("n_prime", 3)
            self.soc_factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        self.initialized = True  # 初期化完了

    def _get_force(self):

        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行
            
        pos = self.peds.pos()
        vel = self.peds.vel()
        
        # 歩行者間の相互作用を計算
        pos_diff = stateutils.each_diff(pos)  # 位置差
        diff_direction, diff_length = stateutils.normalize(pos_diff)  # 正規化方向と距離
        vel_diff = -1.0 * stateutils.each_diff(vel)  # 速度差

        # lambda_importance を vel_diff の形状に拡張
        lambda_expanded = np.repeat(self.lambda_importance, vel_diff.shape[0] // self.lambda_importance.shape[0])
        lambda_expanded = lambda_expanded.reshape(-1, 1)  # (6, 1)

        # vel_diff の形状に合わせる
        lambda_expanded = lambda_expanded * np.ones_like(vel_diff)

        # 相互作用方向を計算
        interaction_vec = lambda_expanded * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # 角度とモデルパラメータを計算
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        gamma_expanded = np.repeat(self.gamma, len(interaction_length) // len(self.gamma))
        B = gamma_expanded * interaction_length

        n_prime_expanded = np.repeat(self.n_prime, len(B) // len(self.n_prime))
        n_expanded = np.repeat(self.n, len(B) // len(self.n))
  
        # 力の計算
        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime_expanded * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n_expanded * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        # 個々の歩行者に作用する力を抽出
        force = force_velocity + force_angle
        force = np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1)
        
        factors_expanded = self.soc_factors[:, None]
        forces = force * factors_expanded

        #print(f"Forces[{camel_to_snake(type(self).__name__)}]: ", forces)

        return forces

class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this scene."""

    def __init__(self):
        super().__init__()
        self.initialized = False  # 初期化フラグ
        self.sigma = None
        self.threshold = None
        self.agent_radius = None
        self.factors = None

    def _initialize(self):
        """初回呼び出し時に初期化"""
        self.sigma = np.zeros((self.peds.size(),))
        self.threshold = np.zeros((self.peds.size(),))
        self.agent_radius = np.zeros((self.peds.size(),))
        self.factors = np.zeros((self.peds.size(),))

        for i, ped_type in enumerate(self.peds.types):
            config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), {})
            self.sigma[i] = config.get("sigma", 0.2)
            self.threshold[i] = config.get("threshold", 3.0)
            self.agent_radius[i] = self.peds.get_agent_radius(ped_type)
            self.factors[i] = self.factor_list.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__), 1.0)

        self.initialized = True  # 初期化完了

    def _get_force(self):
        if not self.initialized:
            self._initialize()  # 初回のみ初期化を実行

        pos = self.peds.pos()
        force = np.zeros((self.peds.size(), 2))

        if len(self.scene.get_obstacles()) == 0:
            return force

        # 障害物と歩行者の位置を計算
        obstacles = np.vstack(self.scene.get_obstacles())  # 障害物の全座標

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.agent_radius[i]
            if np.all(dist >= self.threshold[i]):
                continue
            dist_mask = dist < self.threshold[i]
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / self.sigma[i])
            force[i] = np.sum(directions[dist_mask], axis=0)

        factors_expanded = self.factors[:, None]
        forces = force * factors_expanded

        return forces
