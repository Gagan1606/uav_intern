import numpy as np
import os
import csv
import open3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models

cache = np.load("cached_data_fixed.npz", allow_pickle=True)

all_distance_grids = cache["dist"]
all_gray_grids = cache["gray"]
all_mean_curv_grids = cache["curv"]
all_mos_list = cache["mos"]

assert len(all_distance_grids) == len(all_gray_grids) == len(all_mean_curv_grids) == len(all_mos_list)
print("Loaded cached data:", len(all_mos_list), "samples")


class PatchDataset(Dataset):
    def __init__(self, d, g, c, y):
        self.d, self.g, self.c, self.y = d, g, c, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = np.stack([self.d[i], self.g[i], self.c[i]], axis=0)
        x = torch.tensor(x).float().unsqueeze(0)
        x = F.interpolate(x, size=(32,32), mode="bilinear", align_corners=False).squeeze(0)
        return x, torch.tensor(self.y[i]).float()

dataset = PatchDataset(all_distance_grids, all_gray_grids, all_mean_curv_grids, all_mos_list)
print("Dataset size:", len(dataset))


n = len(dataset)
train_ds, val_ds, test_ds = random_split(dataset, [int(0.7*n), int(0.15*n), n-int(0.85*n)])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True


opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

best_val = float("inf")
best_state = None
epochs = 25


for epoch in range(epochs):
    print("Starting training...")
    model.train()
    train_loss = 0
    for b,(x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device).unsqueeze(1)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        if b % 10 == 0:
            print(f"Epoch {epoch+1} Batch {b} Train Loss {loss.item():.4f}")

    train_loss /= len(train_loader)

    model.eval()
    print("Starting val...")

    val_loss = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device).unsqueeze(1)
            val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} Train {train_loss:.4f} Val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_state = model.state_dict()
        torch.save(best_state, "best_model.pth")

if best_state is not None:
    model.load_state_dict(best_state)
else:
    print("Warning: best_state is None, using last epoch model")
model.eval()
print("Starting testing...")


test_loss = 0
scores = []
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device).unsqueeze(1)
        pred = model(x)
        test_loss += loss_fn(pred, y).item()
        scores.append(pred.cpu())

test_loss /= len(test_loader)
global_quality_index = torch.cat(scores).mean().item()

print(f"Test Loss: {test_loss:.4f}")
print(f"Global Quality Index: {global_quality_index:.4f}")
