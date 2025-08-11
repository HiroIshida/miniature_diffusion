from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from model import SimpleMLP

DOF = 2


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path: Path):
        trajectories = np.load(dataset_path)
        n, T, _ = trajectories.shape
        flattened_trajectories = trajectories.reshape(n, T * DOF)  # flatten
        self.data = torch.from_numpy(flattened_trajectories).float()
        print(f"data shape: {self.data.shape}")

        # compute min and max for normalization
        self.min = self.data.min(dim=0).values
        self.max = self.data.max(dim=0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx][-DOF:]


class Normalizer:
    def __init__(self, min_values, max_values):
        # Diffusion models usually normalize to [-1, 1]
        self.min = min_values
        self.max = max_values

    def normalize(self, x):
        tmp = (x - self.min) / (self.max - self.min + 1e-8)
        return tmp * 2 - 1

    def denormalize(self, x):
        tmp = (x + 1) / 2
        return tmp * (self.max - self.min) + self.min

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self


@torch.no_grad()
def evaluate(model, loader, normalizer, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    for features, contexts in loader:
        features = features.to(device)
        features_normalized = normalizer.normalize(features)
        loss_out = model.compute_loss(features_normalized, features_normalized[:, -DOF:])
        loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out
        bsz = features.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz
    model.train()
    return total_loss / max(total_count, 1)


if __name__ == "__main__":
    workspace = Path("workspace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = 1.0
    batch_size = 512
    num_epochs = 10000
    val_ratio = 0.2  # split train/validation

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        workspace / "runs" / f"log-{timestamp}.tb"
    )  # directory name with an extension-like suffix
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir}")

    dataset = TrajectoryDataset(dataset_path=workspace / "dataset.npy")
    torch.save(
        {"min": dataset.min, "max": dataset.max},
        workspace / "normalizer_min_max.pt",
    )

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    normalizer = Normalizer(dataset.min, dataset.max).to(device)

    model = SimpleMLP(50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_ckpt_path = workspace / "best_ckpt.pt"
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        running_count = 0

        for features, contexts in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False
        ):
            features = features.to(device)
            features_normalized = normalizer.normalize(features)
            loss_out = model.compute_loss(features_normalized, features_normalized[:, -DOF:])
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bsz = features.size(0)
            running_loss += loss.detach().item() * bsz
            running_count += bsz

        train_loss = running_loss / max(running_count, 1)

        val_loss = evaluate(model, val_loader, normalizer, device)

        print(f"Epoch {epoch:04d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "normalizer_min": normalizer.min.detach().cpu(),
                "normalizer_max": normalizer.max.detach().cpu(),
                "timestamp": timestamp,
            }
            torch.save(state, best_ckpt_path)
            print(f"✅ New best! Saved checkpoint to: {best_ckpt_path}")

    writer.close()
