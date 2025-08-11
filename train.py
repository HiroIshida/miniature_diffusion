import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.patches import Rectangle

from rrt import RRTConnect2D
from task import BOUNDS, sample_obstacles, sample_point
from trajectory import Trajectory


def generate_dataset(n: int = 300):
    r = random.Random()
    trajs = []
    # temporary fix obstacles
    obstacles = sample_obstacles(BOUNDS, (16.0, 16.0), 10, rng=random.Random(0))
    pbar = tqdm.tqdm(total=n, desc="Generating trajectories")
    seed = 0
    while len(trajs) < n:
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
        trajs.append(traj)

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
        plt.savefig(f"dataset/trajectory_{len(trajs)}.png")
        pbar.update(1)


if __name__ == "__main__":
    generate_dataset()
