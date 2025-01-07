# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces

import json

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

        #print("Config内容:", vars(self.config))  # オブジェクトの属性を表示
            
        # typeごとの設定を保持
        self.types = types # 0: adult, 1: elderly, 2: child

        self.scene_configs = self.get_scene_configs(self.types, self.config)
        #print("\nscene_configs: ", json.dumps(self.scene_configs, indent=4))

        self.force_configs = self.get_force_configs(self.types, self.config)
        #print("\nforce_configs: ",json.dumps(self.force_configs, indent=4))

        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))

        # initiate agents
        self.peds = PedState(state, types, groups, self.scene_configs)

        #print("peds:", self.peds)  # pedsオブジェクト全体
        #print("Current State:", self.peds.state)  # 歩行者の状態（位置、速度、目標など）
        #print("Positions:", self.peds.pos())  # 歩行者の位置
        #print("Velocities:", self.peds.vel())  # 歩行者の速度
        #print("Goals:", self.peds.goal())  # 歩行者の目標地点
        #print("Tau:", self.peds.tau())  # 歩行者のリラクゼーション時間
        #print("Groups:", self.peds.groups)  # 歩行者のグループ情報

        # construct forces
        self.forces = self.make_forces(self.force_configs)

    # configからtypeごとのsceneを抜き出す
    def get_scene_configs(self, types, config):
        scene_configs = {}
        # setで種類の数だけ実行([0,1,0,2] → [0,1,2])
        for i in set(types):  # typeを一意にしてループ
            # typeごとのscene設定を抽出
            scene_config = (
                config.sub_config("types")
                .sub_config(f"{i}")
                .sub_config("scene")
                .config
            )
            if scene_config:  # scene設定が存在する場合のみ追加
                scene_configs[str(i)] = {"scene": scene_config}

        return scene_configs
    
    # configからtypeごとのforceを抜き出す
    def get_force_configs(self, types, config):
        force_list = [  # forceの名称リスト
            "desired_force",
            "social_force",
            "obstacle_force",
            "ped_repulsive_force",
            "space_repulsive_force",
        ]
        force_configs = {}  # 空のコンフィグ

        # setで種類の数だけ実行([0,1,0,2] → [0,1,2])
        for i in set(types):  # typeを一意にしてループ
            type_forces = {}  # typeごとのforce設定を格納する辞書
            for force in force_list:
                # force設定を抽出
                force_config = (
                    config.sub_config("types")
                    .sub_config(f"{i}")
                    .sub_config("forces")
                    .sub_config(force)
                    .config
                )
                if force_config:  # force_configが存在する場合のみ追加
                    type_forces[force] = force_config
            # typeをキーとしてforce設定を格納
            force_configs[str(i)] = type_forces

        return force_configs


    def make_forces(self, force_configs):
        """Construct forces for each pedestrian type and include group forces if enabled."""
        forces_list = []

        # 各種類の力を生成
        for ped_type, type_config in force_configs.items():
            print("ped_type , type_config: ",ped_type," , ",type_config)
            # 種類別に力を生成
            desired_force = forces.DesiredForce()
            social_force = forces.SocialForce()
            obstacle_force = forces.ObstacleForce()

            # 力オブジェクトを初期化
            desired_force.init(self, {ped_type: type_config["desired_force"]})
            social_force.init(self, {ped_type: type_config["social_force"]})
            obstacle_force.init(self, {ped_type: type_config["obstacle_force"]})

            # 力をリストに追加
            forces_list.extend([desired_force, social_force, obstacle_force])

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

            forces_list["group"] = group_forces  # グループ力を全体の力に追加

        print("Final Force List:", forces_list)

        return forces_list

    # self.forcesに格納されたすべての力(Desired,social,...)を呼び出して計算
    def compute_forces(self):
        """compute forces"""
        # self.forcesの各要素に指定された関数(get_force)を適用
        return sum(map(lambda x: x.get_force(), self.forces))   # 各力を合計して計算

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    # 歩行者に作用する合力を計算
    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())   # 計算した力をPedState.stepに渡し、歩行者の状態を更新

    # 指定回数文ループを回す
    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self.step_once()
        return self
