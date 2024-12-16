# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces


class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, types=None, groups=None, obstacles=None, config_file=None):
        # 設定を読み込む
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)
        #self.scene_config = self.config.sub_config("scene")

        # typeごとの設定を保持
        self.types = types # 0: adult, 1: elderly, 2: child

        # シーン設定の読み込み
        # setで重複を削除しユニークなタイプの集合を作成([0,1,0,1]→[0,1])
        self.scene_configs = {t: self.config.sub_config(f"types.{t}.scene") for t in set(self.types)}
        self.force_configs = {t: self.config.sub_config(f"types.{t}.forces") for t in set(self.types)}

        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))

        # initiate agents
        self.peds = PedState(state, types, groups, self.scene_configs)

        # construct forces
        self.forces = self.make_forces()

    def make_forces(self):
        """Construct forces"""
        force_list = []
        for i, ped_type in enumerate(self.types):
            type_force_config = self.force_configs[ped_type]
            force_list.append(forces.DesiredForce(type_force_config))
            force_list.append(forces.PedRepulsiveForce(type_force_config))
            force_list.append(forces.SpaceRepulsiveForce(type_force_config))

        return force_list

    def compute_forces(self):
        """compute forces"""
        return sum(map(lambda x: x.get_force(), self.forces))

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self.step_once()
        return self
