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
        self.group_config = self.config.sub_config("scene")

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

        print("peds:", self.peds)  # pedsオブジェクト全体
        print("Current State:", self.peds.state)  # 歩行者の状態（位置、速度、目標など）
        print("Positions:", self.peds.pos())  # 歩行者の位置
        print("Velocities:", self.peds.vel())  # 歩行者の速度
        print("Goals:", self.peds.goal())  # 歩行者の目標地点
        print("Tau:", self.peds.tau())  # 歩行者のリラクゼーション時間
        print("Groups:", self.peds.groups)  # 歩行者のグループ情報

        # construct forces
        self.forces = self.make_forces()

    def make_forces(self):
        """Construct forces for each pedestrian type and include group forces if enabled."""
        force_list = []

        # 歩行者ごとの力を追加
        for i, ped_type in enumerate(self.types):
            type_force_config = self.force_configs[ped_type]

            # 個別の力 (5つ) をタイプ別設定で初期化
            force_list += [
                forces.DesiredForce(type_force_config),
                forces.SocialForce(type_force_config),
                forces.ObstacleForce(type_force_config),
                forces.PedRepulsiveForce(type_force_config),
                forces.SpaceRepulsiveForce(type_force_config),
            ]
        print("Force list after adding individual forces:", force_list)

        # グループ関連の力を有効化する場合
        if self.config("scene", {}).get("enable_group", False):
            group_force_config = self.config  # グループ力は全体設定を利用
            group_forces = [
                forces.GroupCoherenceForceAlt(),
                forces.GroupRepulsiveForce(),
                forces.GroupGazeForceAlt(),
            ]
            # グループ力に対して初期化を行う
            for group_force in group_forces:
                group_force.init(self, group_force_config)
            force_list += group_forces  # グループ力を全体の力に追加

        print("Force list after adding group forces:", force_list)

        # 全ての力を初期化 (シーン情報と設定を渡す)
        for force in force_list:
            force.init(self, self.config)

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
