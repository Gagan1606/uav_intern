import os
import csv
import open3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models



ply_dir = r"/home/nandhini/Downloads/Gagan/samples_with_MOS"
csv_path = r"/home/nandhini/Downloads/Gagan/subjectiveMOS.csv"
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))

# mos_map = {}
# with open(csv_path, "r") as f:
#     reader = csv.DictReader(f)
#     print("CSV loaded. Total MOS entries:", len(mos_map))
#     for row in reader:
#         mos_map[row["name"]] = float(row["MOS"])
mos_map = {}
with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mos_map[row["name"]] = float(row["MOS"])

print("Total MOS entries found:", len(mos_map))
print("First 3 MOS entries:", list(mos_map.items())[:3])

all_distance_grids = []
all_gray_grids = []
all_mean_curv_grids = []
all_mos_list = []
all_cloud_ids = []
cloud_id = 0


for fname in os.listdir(ply_dir):
    print(f"\nProcessing file: {fname}")

    if not fname.endswith(".ply"):
        continue

    mos_value = mos_map[fname]
    pc = open3d.io.read_point_cloud(os.path.join(ply_dir, fname))

    all_pts_arr = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    gray = (0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]).reshape(-1,1)
    all_pts_arr = np.hstack((all_pts_arr, gray))

    num_select = 50
    idx = np.random.choice(len(all_pts_arr), num_select, replace=False)
    selected_points = all_pts_arr[idx]
    selection_radius = 10

    all_patches = []
    for i in selected_points:
        patch = []
        for j in all_pts_arr:
            if np.linalg.norm(i[:3] - j[:3]) <= selection_radius:
                patch.append(j)
        all_patches.append(patch)
    print("Number of patches:", len(all_patches))


    patch_2d_list = []
    patch_centres_2d = []

    for patch in all_patches:
        patch_arr = np.asarray(patch)
        centroid = patch_arr[:,:3].mean(axis=0)
        centered = patch_arr[:,:3] - centroid
        C = centered.T @ centered
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        u = centered @ vecs[:,order[0]]
        v = centered @ vecs[:,order[1]]
        uv = np.stack([u,v], axis=1)
        uvg = np.hstack([uv, patch_arr[:,3:4]])
        patch_2d_list.append(uvg)

    for patch in patch_2d_list:
        patch[:,0] = (patch[:,0]-patch[:,0].min())/(patch[:,0].max()-patch[:,0].min()+1e-9)
        patch[:,1] = (patch[:,1]-patch[:,1].min())/(patch[:,1].max()-patch[:,1].min()+1e-9)
        patch_centres_2d.append([patch[0,0], patch[0,1]])

    H = W = 3
    grids = []

    for patch in patch_2d_list:
        grid = [[[] for _ in range(W)] for _ in range(H)]
        for k in range(len(patch)):
            i = min(int(patch[k,1]*H), H-1)
            j = min(int(patch[k,0]*W), W-1)
            grid[i][j].append(patch[k])
        for i in range(H):
            for j in range(W):
                pts = np.asarray(grid[i][j])
                if pts.size == 0:
                    grid[i][j] = [0.0,0.0,0.0]
                else:
                    grid[i][j] = [pts[:,0].mean(), pts[:,1].mean(), pts[:,2].mean()]
        grids.append(np.asarray(grid))

    distance_grids = []
    gray_grids = []

    for grid, centre in zip(grids, patch_centres_2d):
        diff = grid[:,:,:2] - np.asarray(centre)
        distance_grids.append(np.linalg.norm(diff, axis=2))
        gray_grids.append(grid[:,:,2])

    all_grid_3d = []
    for d3, d2 in zip(all_patches, patch_2d_list):
        grid = [[[] for _ in range(W)] for _ in range(H)]
        for k in range(len(d2)):
            i = min(int(d2[k,1]*H), H-1)
            j = min(int(d2[k,0]*W), W-1)
            grid[i][j].append(d3[k])
        all_grid_3d.append(grid)

    mean_curv_grids = []

    for grid in all_grid_3d:
        curv = [[0.0 for _ in range(W)] for _ in range(H)]
        for i in range(H):
            for j in range(W):
                pts = np.asarray(grid[i][j])
                if pts.shape[0] < 3:
                    continue
                centered = pts[:,:3] - pts[0,:3]
                C = centered.T @ centered
                vals,_ = np.linalg.eigh(C)
                curv[i][j] = vals[0]/vals.sum()
        mean_curv_grids.append(curv)

    all_distance_grids.extend(distance_grids)
    all_gray_grids.extend(gray_grids)
    all_mean_curv_grids.extend(mean_curv_grids)
    all_mos_list.extend([mos_value] * len(distance_grids))

    num_patches = len(distance_grids)
    all_cloud_ids.extend([cloud_id] * num_patches)
    cloud_id += 1


    np.savez("cached_data.npz",
         dist=all_distance_grids,
         gray=all_gray_grids,
         curv=all_mean_curv_grids,
         mos=all_mos_list,
         cloud_ids=all_cloud_ids)



# class PatchDataset(Dataset):
#     def __init__(self, d, g, c, y):
#         self.d, self.g, self.c, self.y = d, g, c, y
#     def __len__(self):
#         return len(self.y)
#     def __getitem__(self, i):
#         x = np.stack([self.d[i], self.g[i], self.c[i]], axis=0)
#         x = torch.tensor(x).float().unsqueeze(0)
#         x = F.interpolate(x, size=(32,32), mode="bilinear", align_corners=False).squeeze(0)
#         return x, torch.tensor(self.y[i]).float()

# dataset = PatchDataset(all_distance_grids, all_gray_grids, all_mean_curv_grids, all_mos_list)
# print("Dataset size:", len(dataset))


# n = len(dataset)
# train_ds, val_ds, test_ds = random_split(dataset, [int(0.7*n), int(0.15*n), n-int(0.85*n)])

# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=16)
# test_loader = DataLoader(test_ds, batch_size=16)



# model = models.resnet18(weights="IMAGENET1K_V1")
# model.fc = nn.Linear(model.fc.in_features, 1)
# model = model.to(device)

# opt = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = nn.MSELoss()

# best_val = float("inf")
# best_state = None
# epochs = 25


# for epoch in range(epochs):
#     print("Starting training...")
#     model.train()
#     train_loss = 0
#     for b,(x,y) in enumerate(train_loader):
#         x,y = x.to(device), y.to(device).unsqueeze(1)
#         opt.zero_grad()
#         pred = model(x)
#         loss = loss_fn(pred, y)
#         loss.backward()
#         opt.step()
#         train_loss += loss.item()
#         if b % 10 == 0:
#             print(f"Epoch {epoch+1} Batch {b} Train Loss {loss.item():.4f}")

#     train_loss /= len(train_loader)

#     model.eval()
#     print("Starting val...")

#     val_loss = 0
#     with torch.no_grad():
#         for x,y in val_loader:
#             x,y = x.to(device), y.to(device).unsqueeze(1)
#             val_loss += loss_fn(model(x), y).item()
#     val_loss /= len(val_loader)

#     print(f"Epoch {epoch+1} Train {train_loss:.4f} Val {val_loss:.4f}")

#     if val_loss < best_val:
#         best_val = val_loss
#         best_state = model.state_dict()

# model.load_state_dict(best_state)
# model.eval()
# print("Starting testing...")


# test_loss = 0
# scores = []
# with torch.no_grad():
#     for x,y in test_loader:
#         x,y = x.to(device), y.to(device).unsqueeze(1)
#         pred = model(x)
#         test_loss += loss_fn(pred, y).item()
#         scores.append(pred.cpu())

# test_loss /= len(test_loader)
# global_quality_index = torch.cat(scores).mean().item()

# print(f"Test Loss: {test_loss:.4f}")
# print(f"Global Quality Index: {global_quality_index:.4f}")
# # 