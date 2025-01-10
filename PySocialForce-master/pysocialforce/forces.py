"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod

import numpy as np

from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config, stateutils, logger

import json

def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self): # initより先に呼び出す
        super().__init__()
        self.scene = None  # シーン情報（歩行者や障害物の情報など）
        self.peds = None  # 歩行者情報
        self.config = {}  # 種類ごとの設定を保持
    
    def init(self, scene, config, factor_list):
        """Load config and scene"""
        # 設定ファイルから各力のパラメータを取得
        self.scene = scene
        self.peds = self.scene.peds
        self.config = config
        self.factor_list = factor_list
        
        # 種類ごとの設定を保存
        #for ped_type, config in configs_by_type.items():
            #self.configs[ped_type] = config  # 各種類の設定を保持
            #self.factors[ped_type] = config.get("factor", 1.0)  # 各種類の係数を保存
            #print("ped_type, config: ", ped_type," 、", self.configs)

    @abstractmethod
    def _get_force(self) -> np.ndarray:
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    def get_force(self, debug=False):
        print("name: ", camel_to_snake(type(self).__name__))
        forces = np.zeros((self.peds.size(), 2))
        for i, ped_type in enumerate(self.peds.types):
            forces[i] = self._get_force(ped_type, i) * self.factor_list.get(str(ped_type), 1.0).get(camel_to_snake(type(self).__name__))
            print("i , ped_type : ", i," , ", ped_type)
        if debug:
            logger.debug(f"{camel_to_snake(type(self).__name__)}:\n {repr(forces)}")
        print("value: ", self.factor_list.get(str(ped_type), 1.0).get(camel_to_snake(type(self).__name__)))
        print(f"{type(self).__name__} Forces:\n{forces}")
        return forces


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def _get_force(self):
        F0 = (
            1.0
            / self.peds.tau()
            * (
                np.expand_dims(self.peds.initial_speeds, -1) * self.peds.desired_directions()
                - self.peds.vel()
            )
        )
        return F0 * self.factor


class PedRepulsiveForce(Force):
    """Ped to ped repulsive force"""
    def __init__(self, config):
        super().__init__()
        self.v0 = config("v0", 2.1)
        self.sigma = config("sigma", 0.3)
        self.fov_phi = config("fov_phi", 100.0)
        self.fov_factor = config("fov_factor", 0.5)

    def _get_force(self):
        potential_func = PedPedPotential(self.peds.step_width, v0=self.v0, sigma=self.sigma)
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)

        fov = FieldOfView(phi=self.fov_phi, out_of_view_factor=self.fov_factor)
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)
        F_ab = w * f_ab

        return np.sum(F_ab, axis=1) * self.factor


class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""
    def __init__(self, config):
        super().__init__()
        self.v0 = config("v0", 2.1)
        self.sigma = config("sigma", 0.3)
        self.fov_phi = config("fov_phi", 100.0)
        self.fov_factor = config("fov_factor", 0.5)

    def _get_force(self):
        potential_func = PedPedPotential(self.peds.step_width, v0=self.v0, sigma=self.sigma)
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)

        fov = FieldOfView(phi=self.fov_phi, out_of_view_factor=self.fov_factor)
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)
        F_ab = w * f_ab

        return np.sum(F_ab, axis=1) * self.factor


class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                vectors, norms = stateutils.normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        return forces * self.factor


class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def _get_force(self):
        threshold = self.config("threshold", 0.5)
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.factor


class GroupGazeForce(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        vision_angle = self.config("fov_phi", 100.0)
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

        return forces * self.factor


class GroupGazeForceAlt(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
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

        return forces * self.factor


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint."""

    def _get_force(self, ped_type, index):
        config = self.config.get(str(ped_type)).get(camel_to_snake(type(self).__name__))
        relaxation_time = config.get("relaxation_time", 0.5)
        goal_threshold = config.get("goal_threshold", 0.2)

        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)

        # 力を計算
        force = np.zeros((2,))
        if dist[index] > goal_threshold:
            force = (
                direction[index] * self.peds.max_speeds[index] - vel[index]
            ) / relaxation_time
        else:
            force = -vel[index] / relaxation_time

        return force

class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents."""

    def _get_force(self, ped_type, index):
        # 種類に応じた設定を取得
        config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__))
        lambda_importance = config.get("lambda_importance", 2.0)
        gamma = config.get("gamma", 0.35)
        n = config.get("n", 2)
        n_prime = config.get("n_prime", 3)

        # 歩行者間の相互作用を計算
        pos_diff = stateutils.each_diff(self.peds.pos())  # 位置差
        diff_direction, diff_length = stateutils.normalize(pos_diff)  # 正規化方向と距離
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())  # 速度差

        # 相互作用方向を計算
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # 角度とモデルパラメータを計算
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        B = gamma * interaction_length

        # 力の計算
        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        # 個々の歩行者に作用する力を抽出
        force = force_velocity + force_angle
        force = np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1)

        return force[index]

class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this scene."""

    def _get_force(self, ped_type, index):
        # 種類に応じた設定を取得
        config = self.config.get(str(ped_type), {}).get(camel_to_snake(type(self).__name__))
        sigma = config.get("sigma", 0.2)
        threshold = config.get("threshold", 3.0)

        # 初期化
        force = np.zeros((2,))
        if len(self.scene.get_obstacles()) == 0:
            return force

        # 障害物と歩行者の位置を計算
        obstacles = np.vstack(self.scene.get_obstacles())  # 障害物の全座標
        pos = self.peds.pos()[index]  # 現在の歩行者の位置

        # 各障害物への距離と方向を計算
        diff = pos - obstacles
        directions, dist = stateutils.normalize(diff)
        agent_radius = self.peds.get_agent_radius(ped_type)
        dist -= agent_radius  # 障害物との距離からエージェントの半径を引く

        # 距離が閾値より小さい障害物にのみ影響を与える
        mask = dist < threshold
        if np.any(mask):
            directions[mask] *= np.exp(-dist[mask].reshape(-1, 1) / sigma)
            force = np.sum(directions[mask], axis=0)
        
        return force