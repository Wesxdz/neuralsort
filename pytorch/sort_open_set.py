import os
import numpy as np
import time
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neuralsort import NeuralSort

class NeuralSortMNIST(nn.Module):
    def __init__(self, l=1, final_dim=1):
        super(NeuralSortMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, final_dim)
        self.neural_sort = NeuralSort(tau=1.0, hard=False)

    def forward(self, x):
        # x has shape (M, n, l * 28, 28)
        M, n, _, _ = x.shape
        x = x.view(-1, 1, 28, 28)  # Reshape to (M * n, 1, 28, 28)
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4 * 4 * 64)  # Flatten to (M * n, 4 * 4 * 64)
        x = nn.functional.relu(self.fc1(x))
        scores = self.fc2(x).float()  # Output shape: (M * n, 1), ensure float type
        scores = scores.view(M, n, 1)  # Reshape to (M, n, 1)
        P_hat = self.neural_sort(scores)  # Output shape: (M, n, n)
        return P_hat
    
def load_sort_pair(open_set, pair):
    a = np.load(open_set[pair[0]])
    b = np.load(open_set[pair[1]])
    c = np.stack([a, b])
    c = np.expand_dims(c, 0)
    return torch.from_numpy(c)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralSortMNIST().to(device)
neural_sort = NeuralSort().to(device)
model.load_state_dict(torch.load("/arc/pytorch_mnist.pth", weights_only=True))
model.eval()

open_set_dir_path = "/arc/mnist_sort"
open_set_files = glob.glob(os.path.join(open_set_dir_path, '*.npy'))

# n(n/2)
sort_pairs = []
for a in range(len(open_set_files)):
    for b in range(a+1, len(open_set_files)):
        sort_pairs.append([a, b])

comparator_results = np.zeros(len(open_set_files))
z_f = 0
for pair in sort_pairs:
    start_time = time.time()
    input = load_sort_pair(open_set_files, pair)
    p_h = model(input)
    sort_permutation = p_h[0]
    print(sort_permutation)
    # comparator '0' means  a < b (I think)
    comparator = sort_permutation[0].argmax()
    print(comparator)
    # print(f'{sort_files[0][comparator]} > {sort_files[0][(comparator+1)%2]}')
    print(f'{open_set_files[sort_pairs[z_f][comparator]]} > {open_set_files[sort_pairs[z_f][(comparator+1)%2]]}')
    comparator_results[sort_pairs[z_f][comparator]] = comparator_results[sort_pairs[z_f][comparator]] + 1
    end_time = time.time()
    print(f"Search operator execution time: {(end_time - start_time) * 1000:.2f} ms")
    z_f = z_f + 1

print(comparator_results)
print(open_set_files)
sort_results = [(comparator_results[i], open_set_files[i]) for i in range(len(open_set_files))]
sort_results.sort(key=lambda x:x[0])
print(sort_results)
