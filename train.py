import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
DATASETS = { # позначка: (файл, класів K, каналів C)
"Д1": ("dermamnist", 7, 3),
"Д2": ("bloodmnist", 8, 3),
"Д3": ("organamnist", 11, 1),
"Д4": ("organcmnist", 11, 1),
"Д5": ("pathmnist", 9, 3),
"Д6": ("octmnist", 4, 1),
}
WIDTHS = {"Ш1": (16, 32, 128), "Ш2": (32, 64, 256), "Ш3": (64, 128, 512)}

def load_split(path, split, n=None):
  """Розбиття split ("train" або "test") файлу .npz як TensorDataset."""
  d = np.load(path)
  x = d[f"{split}_images"][:n] # (N, 28, 28) або (N, 28, 28, 3), uint8
  y = d[f"{split}_labels"][:n].reshape(-1) # (N, 1) -> (N,)
  if x.ndim == 3:
    x = x[:, None, :, :] # (N, 1, 28, 28)
  else:
    x = x.transpose(0, 3, 1, 2) # (N, 3, 28, 28)
  x = torch.from_numpy(np.ascontiguousarray(x)).float()
  x = (x / 255 - 0.5) / 0.5
  return TensorDataset(x, torch.from_numpy(y.astype(np.int64)))

class Net(nn.Module):
  def __init__(self, C, c1, c2, h, K):
    super().__init__()
    self.features = nn.Sequential(
      # 28 x 28 -> 14 x 14
      nn.Conv2d(C, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
      # 14 x 14 -> 7 x 7
      nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    )
    self.head = nn.Sequential(
      nn.Flatten(), nn.Linear(7 * 7 * c2, h), nn.ReLU(),
      nn.Linear(h, K),
    )
    def forward(self, x):
      return self.head(self.features(x))

def n_params_formula(C, c1, c2, h, K):
  return 9 * C * c1 + c1 + 9 * c1 * c2 + c2 + 49 * c2 * h + h + h * K + K

name, K, C = DATASETS["Д4"] # за варіантом
c1, c2, h = WIDTHS["Ш1"] # за варіантом
train = load_split(f"data/{name}.npz", "train", n=7000)
test = load_split(f"data/{name}.npz", "test")
torch.manual_seed(0)
model = Net(C, c1, c2, h, K)

n_params = sum(p.numel() for p in model.parameters())
assert n_params == n_params_formula(C, c1, c2, h, K)
