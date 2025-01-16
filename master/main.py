from pathlib import Path
import numpy as np
from pysocialforce.simulator import Simulator
from pysocialforce.utils.plot import SceneVisualizer


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
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
    type = [0,1]    # 0: 成人, 1: 老人, 2: 子供
    # social groups informoation is represented as lists of indices of the state array
    groups = [[0], [1],]
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
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
    s = Simulator(
        initial_state,
        type = type,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("main.toml"),
    )
    # update 80 steps
    s.step(1)
    #s.step(120)

    output_folder = Path("outexample")
    output_folder.mkdir(parents=True, exist_ok=True)  # フォルダが存在しない場合は作成


    with SceneVisualizer(s, str(output_folder / "example")) as sv:
        sv.animate()
        sv.plot()
