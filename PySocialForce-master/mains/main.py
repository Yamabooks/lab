from pathlib import Path
import numpy as np
import pysocialforce as psf

from random import uniform

if __name__ == "__main__":
    # 歩行者の位置、速度、目標を次の形式で表します (px, py, vx, vy, gx, gy)
    # 歩行者の人数を指定
    num_pedestrians = 10

    # タイプの出現割合を指定 (例: 0: 50%, 1: 30%, 2: 20%)
    type_probabilities = [0.7, 0.2, 0.1]  # 必ず合計が1になるようにする
    initial_state = []
    types = []
    #groups = []  # グループリスト
    groups = None

    """for i in range(num_pedestrians):
        # スタートのy座標を10または-10に設定（ランダムに選択）
        start_y = 10 if uniform(0, 1) > 0.5 else -10
        start_x = uniform(-5, 5)  # x座標を-5～5の範囲でランダムに生成

        # ゴールのy座標はスタートと反対側
        goal_y = -10 if start_y == 10 else 10
        goal_x = uniform(-5, 5)  # x座標を-5～5の範囲でランダムに生成

        # 初期速度を0に設定
        vx, vy = 0.0, -0.5 if start_y == 10 else 0.5

        # 状態をリストに追加
        initial_state.append([start_x, start_y, vx, vy, goal_x, goal_y])

        # ランダムに歩行者タイプを設定 (0: adult, 1: elderly, 2: child)
        types.append(np.random.choice([0, 1, 2], p=type_probabilities))

        # グループを1人ずつ設定
        #groups.append([i])

    # NumPy配列に変換
    initial_state = np.array(initial_state)"""

    initial_state = np.array(
        [
            [-1.0, 10, -0.5, -0.5, 1.0, -10],
            [1.0, 10, -0.5, -0.5, -1.0, -10.0],
            #[0.0, 0.0, 0.5, 0.5, 1.0, 10.0],
            #[1.0, 0.0, 0.5, 0.5, 2.0, 10.0],
            #[2.0, 10, -0.5, -0.5, 3.0, 0.0],
            #[3.0, 0.0, 0.5, 0.5, 4.0, 10.0],
        ]
    )
    types = [0,1]    # 0: 成人, 1: 老人, 2: 子供
    # social groups informoation is represented as lists of indices of the state array
    groups = [[0], [1],]

    # 確認用の出力
    print("Initial State: ", initial_state)
    print("Types: ", types)
    print("Groups: ", groups)
    obs = [
        [-6, -6, -11, 11],  # 左の壁
        [6, 6, -11, 11],    # 右の壁
        [-1.5, 0, 5, 6],    # 中上辺
        [0, 1.5, 6, 5],
        [-1.5, 0, -5, -6],    # 中下辺
        [0, 1.5, -6, -5],  
        [-1.5, -1.5, 5, -5],
        [1.5, 1.5, 5, -5],

    ]
    #obs = None
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        types=types,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("main.toml"),
    )
    
    s.step(10)

    with psf.plot.SceneVisualizer(s, "output/animation") as sv:
        sv.animate()
        sv.plot()
