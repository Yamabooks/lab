from pathlib import Path
import numpy as np
import pysocialforce as psf

from random import uniform, randint, choice

if __name__ == "__main__":
    
    # 歩行者の位置、速度、目標を次の形式で表します (px, py, vx, vy, gx, gy)
    # 歩行者の人数を指定
    num_pedestrians = 50

    # タイプの出現割合を指定 (例: 0: 50%, 1: 30%, 2: 20%)
    type_probabilities = [0.7, 0.2, 0.1]  # 必ず合計が1になるようにする
    initial_state = []
    types = []
    groups = []  # グループリスト
    groups = None
    """
    for i in range(num_pedestrians):
        # スタートのy座標を10または-10に設定（ランダムに選択）
        start_y = uniform(0, 20)
        start_x = uniform(-30, 30)

        # ゴールのy座標はスタートと反対側
        goal_y = uniform(0, 20)
        goal_x = -32 if uniform(0, 1) > 0.5 else 32

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
    initial_state = np.array(initial_state)
"""
    # px, py, vx, vy, gx, gy 
    initial_state = np.array(
        [
            [0.5, 2.5, 0.5, 0, 20.0, 2.5],
            #[1.0, 10, -0.5, -0.5, -1.0, -10.0],
            #[0.0, 0.0, 0.5, 0.5, 1.0, 10.0],
            #[1.0, 0.0, 0.5, 0.5, 2.0, 10.0],
            #[2.0, 10, -0.5, -0.5, 3.0, 0.0],
            #[3.0, 0.0, 0.5, 0.5, 4.0, 10.0],
        ]
    )

    # 中継地点をリストで保持
    """waypoints = [
        [[0.0, 5.0], [5.0, -5.0], [10.0, -10.0]],  # 歩行者1の中継地点
        [[5.0, 0.0], [-5.0, 5.0], [-10.0, 10.0]],  # 歩行者2の中継地点
        [[-5.0, -5.0], [0.0, -10.0], [-10.0, -15.0]], # 歩行者3の中継地点
    ]"""
    waypoints = None

    types = [2,]    # 0: 成人, 1: 老人, 2: 子供
    # social groups informoation is represented as lists of indices of the state array
    groups = [[0],]
    #groups = None
    obs = [
        [0, 0, 0, 5],

        [5, 5, 0, 2.5],
        [5, 5, 4.5, 5],

        [10, 10, 0, 0.5],
        [10, 10, 2.5, 5],

        [14, 17, 1.5, 3.5],

        [0, 10, 0, 0],
        [0, 10, 5, 5],
    ]
    #obs = None

    obstruction_area = [ 2, 3, 2, 3]

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
        area = obstruction_area,
        config_file=Path(__file__).resolve().parent.joinpath("main.toml"),
    )
    
    s.step(200)

    with psf.utils.plot.SceneVisualizer(s, "output/animation") as sv:
        sv.animate()
        sv.plot()
