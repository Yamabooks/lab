from pathlib import Path
import numpy as np
import pysocialforce as psf

import random
from random import uniform, randint, choice

def initialize_pedestrians(num_pedestrians=300):
    """
    歩行者の初期状態を生成する関数。
    スタート位置、ゴール位置、初速度、歩行者タイプを設定。

    :param num_pedestrians: 歩行者の数
    :return: 初期状態 (state), 歩行者タイプ (types)
    """
    # スタート位置候補
    start_positions = [[0, 8], [7, 50], [12.5, 0], [25, 0],[40, 0], [50, 31], [50, 12]]
    
    # 初速度候補
    initial_velocities = [[0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.5, 0.5],[0, 0.5], [-0.5, -0.5], [-0.5, -0.5]]

    # 歩行者タイプの確率 (例: adult: 0.6, elderly: 0.3, child: 0.1)
    type_probabilities = [0.6, 0.3, 0.1]

    # 初期状態とタイプを格納するリスト
    initial_state = []
    types = []

    # スタートと異なるゴール位置を決定するヘルパー関数
    def select_different_goal(start, positions):
        """
        スタート位置と異なるゴール位置をランダムに選択
        :param start: スタート位置 [x, y]
        :param positions: 候補となる位置リスト
        :return: ゴール位置 [x, y]
        """
        possible_goals = [pos for pos in positions if pos != start]
        return random.choice(possible_goals)

    # 各歩行者を設定
    for i in range(num_pedestrians):
        # スタート位置をランダムに選択
        start = random.choice(start_positions)
        start_x, start_y = start

        # スタート位置にランダムなずれを追加 (±0.3 の範囲でランダムに変更)
        start_x += random.uniform(-0.3, 0.3)
        start_y += random.uniform(-0.3, 0.3)

        # スタート位置とは異なるゴール位置を選択
        goal_x, goal_y = select_different_goal(start, start_positions)

        # 初速度をランダムに選択
        vx, vy = random.choice(initial_velocities)

        # 状態をリストに追加 (x, y, vx, vy, gx, gy)
        initial_state.append([start_x, start_y, vx, vy, goal_x, goal_y])

        # 歩行者のタイプをランダムで設定 (0: adult, 1: elderly, 2: child)
        ped_type = np.random.choice([0, 1, 2], p=type_probabilities)
        types.append(ped_type)

    # 初期状態を NumPy 配列に変換
    initial_state = np.array(initial_state)

    return initial_state, types


if __name__ == "__main__":
    
    # 歩行者の位置、速度、目標を次の形式で表します (px, py, vx, vy, gx, gy)
    # 歩行者の人数を指定
    num_pedestrians = 10

    # タイプの出現割合を指定 (例: 0: 60%, 1: 30%, 2: 10%)
    type_probabilities = [0.6, 0.3, 0.1]  # 必ず合計が1になるようにする
    initial_state = []
    types = []
    groups = []  # グループリスト
    groups = None
    
    """for i in range(num_pedestrians):
        # スタートのy座標を10または-10に設定（ランダムに選択）
        start_y = uniform(0, 20)
        start_x = uniform(0, 60)

        # ゴールのy座標はスタートと反対側
        goal_y = uniform(0, 20)
        goal_x = -4 if uniform(0, 1) > 0.5 else 64

        # 初期速度を0.5に固定
        vx = 0.5 * (-1 if start_x > goal_x else 1)  # ゴール方向に向かうよう符号を調整
        vy = 0.5 * (-1 if start_y > goal_y else 1)  # ゴール方向に向かうよう符号を調整
      
        # 状態をリストに追加
        initial_state.append([start_x, start_y, vx, vy, goal_x, goal_y])

        # ランダムに歩行者タイプを設定 (0: adult, 1: elderly, 2: child)
        types.append(np.random.choice([0, 1, 2], p=type_probabilities))

    # グループがNoneでない場合のみグループ分けを行う
    if groups is not None:
        grouped_pedestrians = set()  # すでにグループ化された歩行者を追跡
        groups = []  # グループリスト

        for i in range(num_pedestrians):
            if i not in grouped_pedestrians:
                # グループにする人数をランダムで決定（1～3人）
                group_size = randint(1, 3)

                # グループに追加する候補をランダムに選択
                group_members = [i]
                while len(group_members) < group_size and len(group_members) + len(grouped_pedestrians) < num_pedestrians:
                    candidate = randint(0, num_pedestrians - 1)
                    if candidate not in group_members and candidate not in grouped_pedestrians:
                        group_members.append(candidate)

                # グループをリストに追加
                groups.append(group_members)

                # グループ化された歩行者を記録
                grouped_pedestrians.update(group_members)

    # NumPy配列に変換
    initial_state = np.array(initial_state)"""

    # px, py, vx, vy, gx, gy 
    """initial_state = np.array(
        [
            #[0.5, 2.5, 0.5, 0, 20.0, 2.5],
            #[0.5, 1.0, 0.5, 0, 20.0, 1.0],
            #[0.5, 2.5, 0.5, 0, 20.0, 2.5],
            #[0.5, 4.0, 0.5, 0, 20.0, 4.0],
            
            [0,8 , 0.5, 0.5, 7, 50],
            #[3.0, 0.0, 0.5, 0.5, 4.0, 10.0],
        ]
    )"""

    # 中継地点をリストで保持
    waypoints = [
        [[12.5, 0], [20.0, 5.0], [12.5, 10.0]],  # 歩行者1の中継地点
        #[[5.0, 0.0], [-5.0, 5.0], [-10.0, 10.0]],  # 歩行者2の中継地点
        #[[-5.0, -5.0], [0.0, -10.0], [-10.0, -15.0]], # 歩行者3の中継地点
    ]
    waypoints = None
    
    types = [0]    # 0: 成人, 1: 老人, 2: 子供
    # social groups informoation is represented as lists of indices of the state array
    groups = [[0]]
    groups = None
    
    """obs = [
        [0, 0, 0, 5],

        [5, 5, 0, 2.5],
        [5, 5, 4.5, 5],

        [10, 10, 0, 0.5],
        [10, 10, 2.5, 5],

        [14, 17, 1.5, 3.5],

        [0, 10, 0, 0],
        [0, 10, 5, 5],
    ]"""
    """obs = [
        [0, 60, 0, 0],
        [0, 60, 20, 20],
        ]   
    """
    obs = [
        # 中央
        [15, 18, 0, 5],
        [35, 38, 5, 0],
        [18, 22, 5, 5],
        [28, 35, 5, 5],
        [22, 22, 0, 5],
        [28, 28, 0, 5],
        [15, 15, -5, 0],
        [22, 22, -5, 0],
        [28, 28, -5, 0],
        [38, 38, -5, 0],

        # 左
        [0, 5, 10, 30],
        [5, 5, 30, 55],
        [-5, 0, 10, 10],

        # 上
        [10, 10, 38, 55],
        [10, 44, 38, 30],
        [44, 50, 30, 33],
        [50, 55, 33, 33],

        # 右
        [47, 50, 27, 29],
        [47, 50, 27, 18],
        [50, 55, 18, 18],
        [50, 55, 29, 29],

        # 中央obs
        [25, 39, 30, 27],
        [22, 25, 10, 30],
        [22, 36, 10, 10],
        [36, 39, 10, 27],

        # 左obs
        [10, 20, 33, 31],
        [5, 10, 9, 33],
        [5, 12, 9, 4],
        [17, 20, 10, 31],
        [12, 17, 4, 10],

        # 右obs
        [40, 43, 5, 0],
        [40, 43, 5, 23],
        [43, 50, 23, 8],
        [43, 43, -5, 0],
        [50, 55, 8, 8],

        # 左下obs
        [0, 10, 5, 0],
        [-5, 0, 5, 5],
        [10, 10, 0, -5],

    ]
    #obs = None

    obstruction_area = [
        [2.5, 5, 0, 5, 0.5],
        [10, 12.5, 0, 5, 1.5],
        ]
    obstruction_area = None

    # x_min,x_max,y_min,y_max,x_temp,y_temp
    sign_area = [
        [10, 11, 0.5, 2.5, 14, 0.5],
        ]
    sign_area = None

    sign_goal = {
        "goal0": [[1, 2, 3], [4, 5, 6]],
        #"goal1": [[7, 8, 9], [10, 11, 12]],
        #"goal2": [[13, 14, 15], [16, 17, 18]],
    }

    initial_state, types = initialize_pedestrians()
    groups = None

    # 確認用の出力
    print("Initial State: ", initial_state)
    print("Types: ", types)
    print("Groups: ", groups)
    
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        waypoints,
        types=types,
        groups=groups,
        obstacles=obs,
        obs_area = obstruction_area,
        sgn_area = sign_area,
        sgn_goal = sign_goal,
        config_file=Path(__file__).resolve().parent.joinpath("main.toml"),
    )
    
    s.step(500)

    with psf.utils.plot.SceneVisualizer(s, "output/animation") as sv:
        sv.animate()
        sv.plot()
