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
        self.scene = None   # シーン情報
        self.peds = None    # 歩行者状態
        self.factor = 1.0   # 力の係数(設定ファイルから取得)
        self.config = Config()  # 設定を格納

    def init(self, scene, config):
        """Load config and scene"""
        # 設定ファイルから各力のパラメータを取得
        self.config = config.sub_config(camel_to_snake(type(self).__name__))
        if self.config:
            self.factor = self.config("factor", 1.0)

        self.scene = scene  # シーン情報
        self.peds = self.scene.peds # 歩行者情報

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
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """
    def __init__(self, config):
        super().__init__()
        self.relaxation_time = config("relaxation_time", 0.5)
        self.goal_threshold = config("goal_threshold", 0.2)

    def _get_force(self):
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)
        force = np.zeros((self.peds.size(), 2))
        # 目標地点に向かう力を計算
        force[dist > self.goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > self.goal_threshold, :]
        force[dist <= self.goal_threshold] = -1.0 * vel[dist <= self.goal_threshold]
        
        return force / self.relaxation_time * self.factor


class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """
    def __init__(self, config):
        super().__init__()
        self.lambda_importance = config("lambda_importance", 2.0)
        self.gamma = config("gamma", 0.35)
        self.n = config("n", 2)
        self.n_prime = config("n_prime", 3)

    def _get_force(self):
        pos_diff = stateutils.each_diff(self.peds.pos())
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())

        # 相互作用方向
        interaction_vec = self.lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(diff_direction)
        B = self.gamma * interaction_length

        # 力の計算
        force_velocity = np.exp(-diff_length / B - (self.n_prime * B * theta) ** 2).reshape(-1, 1) * interaction_direction
        force_angle = -np.sign(theta).reshape(-1, 1) * np.exp(-diff_length / B - (self.n * B * theta) ** 2) * stateutils.left_normal(interaction_direction)
        force = force_velocity + force_angle

        return np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1) * self.factor


class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def __init__(self, config):
        super().__init__()
        self.sigma = config("sigma", 0.2)
        self.threshold = config("threshold", 3.0)

    def _get_force(self):
        force = np.zeros((self.peds.size(), 2))
        if len(self.scene.get_obstacles()) == 0:
            return force

        obstacles = np.vstack(self.scene.get_obstacles())
        pos = self.peds.pos()

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius
            mask = dist < self.threshold
            directions[mask] *= np.exp(-dist[mask].reshape(-1, 1) / self.sigma)
            force[i] = np.sum(directions[mask], axis=0)

        return force * self.factor
