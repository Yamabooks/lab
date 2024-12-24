from pathlib import Path
import numpy as np
import pysocialforce as psf


if __name__ == "__main__":
    # 歩行者の位置、速度、目標を次の形式で表します (px, py, vx, vy, gx, gy)
    initial_state = np.array(
        [
            [0.0, 10, 0.0, -0.5, 0.0, 0.0],
            [0.5, 10, 0.0, -0.5, 0.5, 0.0],
            #[0.0, 0.0, 0.0, 0.5, 0.0, 10.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )
    # age type
    #types = [0,1,0,]    # 0: adult, 1: elderly, 2: child(未実装)
    types = [0,1]
    # social groups informoation is represented as lists of indices of the state array
    #groups = [[0, 1], [2]] # 括弧内がグループ
    groups = [[0],[1]]
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    obs = [[0, 1, 5, 5]]
    # obs = None
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        types=types,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("main.toml"),
    )
    
    s.step(120)

    with psf.plot.SceneVisualizer(s, "output/animation") as sv:
        sv.animate()
        sv.plot()
