from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from model import SimpleMLP


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path: Path):
        trajectories = np.load(dataset_path)
        n, T, _ = trajectories.shape
        flattened_trajectories = trajectories.reshape(n, T * 2)  # flatten
        self.data = torch.from_numpy(flattened_trajectories).float()
        print(f"data shape: {self.data.shape}")

        # compute min and max for normalization
        self.min = self.data.min(dim=0).values
        self.max = self.data.max(dim=0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx][-2:]


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


if __name__ == "__main__":
    dataset = TrajectoryDataset(dataset_path=Path("workspace/dataset.npy"))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SimpleMLP(50)

    normalizer = Normalizer(dataset.min, dataset.max)
    for features, contexts in tqdm.tqdm(dataloader):
        features_normalized = normalizer.normalize(features)
        model.compute_loss(features_normalized, contexts)

if __name__ == "__main__":
    workspace = Path("workspace")
    device = "cuda"
    grad_clip = 1.0

    dataset = TrajectoryDataset(dataset_path=workspace / "dataset.npy")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    normalizer = Normalizer(dataset.min, dataset.max)
    normalizer = normalizer.to(device)

    model = SimpleMLP(50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    global_step = 0
    model.train()

    for epoch in range(10000):
        for features, contexts in dataloader:
            features = features.to(device)
            contexts = contexts.to(device)

            features_normalized = normalizer.normalize(features)

            loss_out = model.compute_loss(features_normalized, contexts)
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_val = loss.detach().item()
            print(loss_val)
