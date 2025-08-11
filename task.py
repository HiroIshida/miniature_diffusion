import matplotlib.pyplot as plt
import random
import math
from rrt import RRTConnect2D
from matplotlib.patches import Rectangle

BOUNDS = (0.0, 0.0, 100.0, 100.0)

def _overlap(a, b):
    return (min(a[2], b[2]) - max(a[0], b[0]) > 0) and (min(a[3], b[3]) - max(a[1], b[1]) > 0)

def _point_in_rect(x, y, r):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]

def _sample_obstacles(bounds, size, n, rng, max_tries=10000):
    xmin, ymin, xmax, ymax = bounds
    w, h = size
    obs = []
    tries = 0
    while len(obs) < n and tries < max_tries:
        tries += 1
        x0 = rng.uniform(xmin, xmax - w)
        y0 = rng.uniform(ymin, ymax - h)
        r = (x0, y0, x0 + w, y0 + h)
        bad = False
        for o in obs:
            if _overlap(r, o):
                bad = True
                break
        if not bad:
            obs.append(r)
    return obs

def _sample_point(bounds, obstacles, rng, min_dist_to=None, min_sep=0.0, max_tries=10000):
    xmin, ymin, xmax, ymax = bounds
    for _ in range(max_tries):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if any(_point_in_rect(x, y, r) for r in obstacles):
            continue
        if min_dist_to is not None:
            if math.hypot(x - min_dist_to[0], y - min_dist_to[1]) < min_sep:
                continue
        return (x, y)
    raise RuntimeError("failed to sample a free point")

def sample_problem(bounds=BOUNDS, obstacle_size=(16.0, 16.0), n_obstacles=None, rng=None):
    rng = rng or random.Random()
    n = n_obstacles if n_obstacles is not None else rng.randint(5, 10)
    start = (2, 2)
    obstacles = _sample_obstacles(bounds, obstacle_size, n, rng)
    diag = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
    goal = _sample_point(bounds, obstacles, rng, min_dist_to=start, min_sep= 0.8*diag)
    return start, goal, obstacles

if __name__ == "__main__":
    rng = None
    start, goal, obstacles = sample_problem(rng=rng)
    planner = RRTConnect2D(BOUNDS, obstacles=obstacles, step_size=5.0, max_iters=30000, goal_sample_rate=0.2, collision_step=1.0, rng=rng)
    path = planner.plan(start, goal)
    path = planner.shortcut(path)
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    for oxmin, oymin, oxmax, oymax in obstacles:
        ax.add_patch(Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin, facecolor='tab:gray', alpha=0.6, edgecolor='black'))
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 'o', markersize=6)
    if path is not None:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2)
        ax.plot(xs, ys, 'o', markersize=3)
    plt.show()
