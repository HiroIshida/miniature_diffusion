import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro
from matplotlib.patches import Rectangle

from lib.task import BOUNDS, sample_obstacles
from model import SimpleMLP
from train import Normalizer


def main(step: int | None = None):
    workspace_path = Path("./workspace")
    ckpt_path = workspace_path / "best_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model = SimpleMLP(50)
    model.load_state_dict(state["model_state_dict"])

    norm_min_max_path = workspace_path / "normalizer_min_max.pt"
    if not norm_min_max_path.exists():
        raise FileNotFoundError(f"Normalizer min/max file not found: {norm_min_max_path}")
    norm_min_max = torch.load(norm_min_max_path, map_location=torch.device("cpu"))
    normalizer_min = norm_min_max["min"]
    normalizer_max = norm_min_max["max"]
    normalizer_context = Normalizer(normalizer_min[-2:], normalizer_max[-2:])
    normalizer_action = Normalizer(normalizer_min, normalizer_max)

    obstacles = sample_obstacles(BOUNDS, (16.0, 16.0), 10, rng=random.Random(0))
    goal = (60, 98)

    context = torch.Tensor(goal).unsqueeze(0).float()
    context_norm = normalizer_context.normalize(context)

    actions_np_list = []
    for _ in range(20):
        actions_norm = model.sample(context_norm, n_step=step)
        actions_torch = normalizer_action.denormalize(actions_norm)
        actions_np = actions_torch.numpy().reshape(-1, 2)
        actions_np_list.append(actions_np)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", "box")
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
    ax.set_aspect("equal")
    for actions_np in actions_np_list:
        ax.plot(
            actions_np[:, 0],
            actions_np[:, 1],
            marker="o",
            markersize=3,
            label="Actions",
            alpha=0.5,
            color="orange",
        )
    ax.plot(goal[0], goal[1], marker="*", markersize=30, color="red", label="Goal")
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"step size {step}" if step is not None else "step size 100")
    ax.set_title(ax.get_title(), fontsize=20)
    file_name = f"step_{step:03d}.png" if step is not None else "step_100.png"
    plt.savefig(workspace_path / file_name, bbox_inches="tight")


if __name__ == "__main__":
    tyro.cli(main, description="Evaluate the model and visualize the results.")
