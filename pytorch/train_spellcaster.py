# Sort Learning from Human Feedback

import numpy as np
from PIL import Image
import random
import syn_data
import os
from gpdl import OnlineGraphicsDataset, HumanSpellcastingDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from neuralsort import NeuralSort
import time
import socket
import shutil
import glob

class GraphicsProgramHypothesisSort(nn.Module):
    def __init__(self, l=1, final_dim=1):
        super(GraphicsProgramHypothesisSort, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.transformer = nn.TransformerEncoderLayer(d_model=4096, nhead=8, dim_feedforward=4096, dropout=0.1)
        self.fc2 = nn.Linear(4096, final_dim)
        self.neural_sort = NeuralSort(tau=1.0, hard=False)

    def forward(self, x):
        # x has shape (M, n, l * 28, 28)
        M, n, _, _ = x.shape
        x = x.view(-1, 1, 28, 28)  # Reshape to (M * n, 1, 28, 28)
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(M, n, -1)  # Reshape to (M, n, 64*4*4)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # Transformer encoder layer
        scores = self.fc2(x).float()  # Output shape: (M, n, 1), ensure float type
        scores = scores.view(M, n, 1)  # Reshape to (M, n, 1)
        P_hat = self.neural_sort(scores)  # Output shape: (M, n, n)
        return P_hat

max_val = 26
def init_neighbors(program_vars):
    neighbors = []
    node_to_var_delta_index = {} # Map the variable delta to a node, so that if it is top sorted this variable can be further iterated
    init_val = int(max_val/2)
    for v_i in range(len(program_vars)):
        node = program_vars.copy()
        for n_i in range(len(node)):
            node[n_i] = random.randint(0, 32)
        node[v_i] = init_val
        node_to_var_delta_index[hash(tuple(node))] = v_i
        neighbors.append(node)
    return neighbors, node_to_var_delta_index

# Each variable has a 'current head' which is used as the default value when generating new neighbors...
def populate_neighbors(program_vars, top_sorted_var_delta_index, node_to_var_delta_index):
    neighbors = []
    current_head = program_vars[top_sorted_var_delta_index]
    left_delta = -int(current_head / 2)
    right_delta = int((max_val - current_head) / 2)
    for delta in set([-1, left_delta, right_delta, 1]):
        new_head = current_head + delta
        # Ensure the new head is within bounds
        if 0 <= new_head <= max_val and new_head != current_head:
            node = [new_head if i == top_sorted_var_delta_index else program_vars[i] for i in range(len(program_vars))]
            node_to_var_delta_index[hash(tuple(node))] = top_sorted_var_delta_index
            neighbors.append(node)
    return neighbors

def split_dataset(dataset, test_ratio=0.2):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def train_neural_graphics_pathfinder():
    root_dir = '/spells/spells'
    dataset = HumanSpellcastingDataset(root_dir)
    train_dataset, test_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    program_state_model = GraphicsProgramHypothesisSort().to(device)
    neural_sort = NeuralSort().to(device)
    optimizer = torch.optim.Adam(program_state_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
            program_state_model.train()
            for batch_idx, (graphics_program, sort_stages, program_deltas) in enumerate(train_loader):
                data = program_deltas
                data = data.view(-1, 2, 28*4, 28)
                priorities = sort_stages.reshape(-1, 2)
                data = data.to(device)
                priorities = priorities.to(device)

                P_true = neural_sort(priorities.unsqueeze(-1))

                optimizer.zero_grad()

                output = program_state_model(data)
                logits = torch.log(output + 1e-20)
                loss = loss_fn(logits, P_true)
                loss.backward()

                optimizer.step()
                print(loss)

    torch.save(program_state_model.state_dict(), '/spells/program_state_model.pth')

train_neural_graphics_pathfinder()