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

    def __init__(self, state, waypoints=None, types=None, groups=None, obstacles=None, obs_area = None, sgn_area=None, sgn_goal=None, config_file=None):
        # 設定を読み込む
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)
        
        # typeごとの設定を保持
        self.types = types # 0: adult, 1: elderly, 2: child

        if waypoints is not None:
            state, waypoints = self.load_waypoints(state, waypoints)

        self.sgn_area = sgn_area

        self.scene_configs = self.get_scene_configs(self.types, self.config)
        
        self.force_configs, self.factor_list = self.get_force_configs(self.types, self.config)

        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 50.0))

        # initiate agents
        self.peds = PedState(state, waypoints, types, groups, obs_area, sgn_area, self.scene_configs)

        # construct forces
        self.forces = self.make_forces()

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
    
    def load_waypoints(self, state, waypoints):
        for i, ped_type in enumerate(self.types):  # 各歩行者について処理
            goal = state[i, 4:6].tolist()  # 現在のゴール地点を取得
            
            first_goal = waypoints[i].pop(0)
            state[i, 4:6] = first_goal  # 最初の中継地点を次のゴールに設定
            waypoints[i].append(goal)  # 現在のゴール地点をリストの末尾に追加

        return state, waypoints
    
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

        # forceごとのfactorをリスト化
        factor_list = {}
        for ped_type, forces in force_configs.items():
            factor_list[ped_type] = {force_name: details.get('factor') for force_name, details in forces.items()}
        
        return force_configs, factor_list

    def make_forces(self):
        """Construct forces for each pedestrian type and include group forces if enabled."""
        forces_list = []

        # 種類別に力を生成し初期化
        forces_to_initialize = [
            forces.DesiredForce,
            forces.SocialForce,
            forces.ObstacleForce,
            forces.PedRepulsiveForce,
            forces.SpaceRepulsiveForce,
        ]

        for force_class in forces_to_initialize:
            force = force_class()
            force.init(self, self.force_configs, self.factor_list)
            forces_list.append(force)

        # グループ関連の力を有効化する場合
        if self.config("scene", {}).get("enable_group", False):
            group_forces_to_initialize = [
                forces.GroupCoherenceForceAlt,
                forces.GroupRepulsiveForce,
                forces.GroupGazeForceAlt,
            ]

            for group_force_class in group_forces_to_initialize:
                group_force = group_force_class()
                group_force.init(self, self.force_configs, self.factor_list)
                forces_list.append(group_force)

        #print("Force_list: ", forces_list)
        return forces_list


    # self.forcesに格納されたすべての力(Desired,social,...)を呼び出して計算
    def compute_forces(self):
        """compute forces"""
        if self.sgn_area is not None:
            self.sign_area()
        # self.forcesの各要素に指定された関数(get_force)を適用
        total_force = sum(map(lambda x: x.get_force(), self.forces))    # 各力を合計して計算
        #print(f"Total Force: {total_force}")
        return total_force
    
    # 特定エリアによって、ゴールを変更
    def sign_area(self):
        """signエリア内で一時的なゴールを設定"""
        for i, ped_pos in enumerate(self.peds.pos()):
            for sign in self.sgn_area:
                x_min, x_max, y_min, y_max, x_temp, y_temp = sign

                # 歩行者が指定されたエリア内にいる場合
                if x_min <= ped_pos[0] <= x_max and y_min <= ped_pos[1] <= y_max:
                    if not self.peds.is_temporary_goal(i):  # 一時ゴールが設定されていない場合
                        self.peds.set_temporary_goal(i, [x_temp, y_temp])  # 一時ゴールを設定
                        break  # 一つのsignエリアだけ処理する
            else:
                # 一時ゴールに到達している場合は元のゴールに戻す
                if self.peds.reached_temporary_goal(i):
                    self.peds.restore_original_goal(i)

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
        for i in range(n):
            self.step_once()
            print(f"step{i}")
        return self
