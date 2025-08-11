# The file is sorely written by ChatGPT
import math
import random

class Node:
    def __init__(self, x, y, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent

class RRTConnect2D:
    def __init__(self, bounds, obstacles=None, step_size=10.0, max_iters=10000, goal_sample_rate=0.1, collision_step=1.0, rng=None):
        self.bounds = bounds
        self.obstacles = obstacles or []
        self.step_size = float(step_size)
        self.max_iters = int(max_iters)
        self.goal_sample_rate = float(goal_sample_rate)
        self.collision_step = float(collision_step)
        self.rng = rng or random.Random()

    def plan(self, start, goal):
        if not self.point_free(start[0], start[1]) or not self.point_free(goal[0], goal[1]):
            return None
        T_start = [Node(start[0], start[1], None)]
        T_goal = [Node(goal[0], goal[1], None)]
        flip = False
        for _ in range(self.max_iters):
            q_rand = goal if self.rng.random() < self.goal_sample_rate else self.sample_free()
            if not flip:
                s, ia = self.extend(T_start, q_rand)
                if s != "Trapped":
                    s2, ib = self.connect(T_goal, (T_start[ia].x, T_start[ia].y))
                    if s2 == "Reached":
                        return self.reconstruct(T_start, ia, T_goal, ib)
            else:
                s, ia = self.extend(T_goal, q_rand)
                if s != "Trapped":
                    s2, ib = self.connect(T_start, (T_goal[ia].x, T_goal[ia].y))
                    if s2 == "Reached":
                        return self.reconstruct(T_start, ib, T_goal, ia)
            flip = not flip
        return None

    def shortcut(self, path, iterations=200):
        if not path or len(path) < 3:
            return path
        pts = [tuple(map(float, p)) for p in path]
        for _ in range(int(iterations)):
            i = self.rng.randint(0, len(pts) - 3)
            j = self.rng.randint(i + 2, len(pts) - 1)
            if self.segment_free(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
                pts = pts[:i + 1] + pts[j:]
        return pts

    def sample_free(self):
        xmin, ymin, xmax, ymax = self.bounds
        for _ in range(100):
            x = self.rng.uniform(xmin, xmax)
            y = self.rng.uniform(ymin, ymax)
            if self.point_free(x, y):
                return (x, y)
        return (self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax))

    def nearest(self, tree, q):
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(tree):
            d = (n.x - q[0]) ** 2 + (n.y - q[1]) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def steer(self, from_node, q):
        dx = q[0] - from_node.x
        dy = q[1] - from_node.y
        d = math.hypot(dx, dy)
        if d <= self.step_size:
            return (q[0], q[1])
        r = self.step_size / d
        return (from_node.x + dx * r, from_node.y + dy * r)

    def extend(self, tree, q):
        i_near = self.nearest(tree, q)
        new_xy = self.steer(tree[i_near], q)
        if self.segment_free(tree[i_near].x, tree[i_near].y, new_xy[0], new_xy[1]):
            tree.append(Node(new_xy[0], new_xy[1], i_near))
            if math.hypot(new_xy[0] - q[0], new_xy[1] - q[1]) <= self.step_size:
                return "Reached", len(tree) - 1
            return "Advanced", len(tree) - 1
        return "Trapped", None

    def connect(self, tree, q):
        status = "Advanced"
        last_idx = None
        while status == "Advanced":
            status, last_idx = self.extend(tree, q)
            if status == "Trapped":
                return "Trapped", last_idx
        return status, last_idx

    def reconstruct(self, tree_start, idx_start, tree_goal, idx_goal):
        p1 = self.path_to_root(tree_start, idx_start)
        p2 = self.path_to_root(tree_goal, idx_goal)
        return p1 + p2[::-1][1:]

    def path_to_root(self, tree, idx):
        out = []
        i = idx
        while i is not None:
            n = tree[i]
            out.append((n.x, n.y))
            i = n.parent
        return out[::-1]

    def point_free(self, x, y):
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        for oxmin, oymin, oxmax, oymax in self.obstacles:
            if oxmin <= x <= oxmax and oymin <= y <= oymax:
                return False
        return True

    def segment_free(self, x0, y0, x1, y1):
        if not self.point_free(x0, y0) or not self.point_free(x1, y1):
            return False
        dx = x1 - x0
        dy = y1 - y0
        L = math.hypot(dx, dy)
        if L == 0:
            return self.point_free(x0, y0)
        steps = max(1, int(math.ceil(L / self.collision_step)))
        for k in range(1, steps + 1):
            t = k / steps
            x = x0 + dx * t
            y = y0 + dy * t
            if not self.point_free(x, y):
                return False
        return True
