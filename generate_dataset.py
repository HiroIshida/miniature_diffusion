import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tyro
from matplotlib.patches import Rectangle

from lib.rrt import RRTConnect2D
from lib.task import BOUNDS, sample_obstacles, sample_point
from lib.trajectory import Trajectory


def generate_dataset(n: int = 10000, debug: bool = False):
    r = random.Random()
    traj_arrs = []
    # temporary fix obstacles
    obstacles = sample_obstacles(BOUNDS, (16.0, 16.0), 10, rng=random.Random(0))
    pbar = tqdm.tqdm(total=n, desc="Generating trajectories")
    seed = 0

    # sample goal randomly and plan a trajectory
    while len(traj_arrs) < n:
        start = (2, 2)
        r = random.Random(seed)
        goal = sample_point(BOUNDS, obstacles, r, min_dist_to=start, min_sep=50)
        planner = RRTConnect2D(
            BOUNDS,
            obstacles=obstacles,
            step_size=5.0,
            max_iters=30000,
            goal_sample_rate=0.2,
            collision_step=1.0,
            rng=r,
        )
        seed += 1
        path = planner.plan(start, goal)
        if path is None:
            continue
        path = planner.shortcut(path, iterations=10)
        if path is None:
            continue
        traj = Trajectory(np.array(path))
        traj_arr = traj.resample(100).numpy()
        traj_arrs.append(traj_arr)

        pbar.update(1)

        if debug:
            # visualize the world
            fig, ax = plt.subplots()
            ax.set_aspect("equal", "box")
            ax.set_xlim(BOUNDS[0], BOUNDS[2])
            ax.set_ylim(BOUNDS[1], BOUNDS[3])
            for oxmin, oymin, oxmax, oymax in obstacles:
                ax.add_patch(
                    Rectangle(
                        (oxmin, oymin),
                        oxmax - oxmin,
                        oymax - oymin,
                        facecolor="tab:gray",
                        alpha=0.6,
                        edgecolor="black",
                    )
                )
            # add solution path
            pts = np.array(path, dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], linewidth=2)
            ax.plot(pts[:, 0], pts[:, 1], "o", markersize=3)
            # add start and goal
            ax.plot([start[0], goal[0]], [start[1], goal[1]], "o", markersize=6)
            debug_base = Path("debug")
            debug_base.mkdir(exist_ok=True, parents=True)
            plt.savefig(debug_base / f"trajectory_{len(traj_arrs)}.png")
            plt.close(fig)

    trajs_arr = np.array(traj_arrs, dtype=np.float32)
    dataset_path = Path("workspace") / "dataset.npy"
    with dataset_path.open("wb") as f:
        np.save(f, trajs_arr)


if __name__ == "__main__":
    tyro.cli(generate_dataset)
