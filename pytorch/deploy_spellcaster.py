import numpy as np
from PIL import Image
import random
import syn_data
import os
from gpdl import OnlineGraphicsDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import syn_data
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

def render_program(name, program, correct):
    neighbor_op_matrix = np.zeros((syn_data.input_size, syn_data.input_size))
    # Render functions are specifically for human feedback purposes and are mapped to standard matrices 
    program_img = syn_data.render_program_matrix(program)
    op_img = np.zeros((neighbor_op_matrix.shape[0], neighbor_op_matrix.shape[1], 3), dtype=np.uint8)
    syn_data.render_op(op_img, correct.tolist()[0])
    syn_data.render_search(op_img, program)
    concat_img = np.concatenate((program_img, op_img), axis=1)
    pil_img = Image.fromarray(concat_img)
    pil_img.save(name)

def split_dataset(dataset, test_ratio=0.2):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def deploy_neural_graphics_pathfinder(program_open_set):
    root_dir = 'graphics_programs'
    dataset = OnlineGraphicsDataset(root_dir, 0, 1)

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #  = GraphicsProgramHypothesisSort().to(device)
    program_state_model = GraphicsProgramHypothesisSort(torch.load('program_state_model.pth', map_location=torch.device('cpu')))
    neural_sort = NeuralSort().to(device)

    with torch.no_grad():
            for batch_idx, (input_matrix, output_matrix, op_matrix, program_matrix, program_vars) in enumerate(test_loader):
                program_open_set = program_vars.tolist()[0]
                for path_node in range(10):
                    neighbors, node_to_var_delta_index = init_neighbors(program_open_set)

                    n_next = []
                    for neighbor in neighbors:
                        for v_i in range(len(program_open_set)):
                            new_neighbors = populate_neighbors(neighbor, v_i, node_to_var_delta_index)
                            n_next.append(new_neighbors)

                    for neighbor_set in n_next:
                        neighbors.extend(neighbor_set)

                    # There are 64 neighbors
                    # Unless a var expansion model is used...
                    # In which case there are 16

                    neighbor_model_input = {}
                    # So basically what we wanna do is to sort all the neighbors var expansions
                    # and then populate new neighbors by expanding the state/var of the top sort
                    for n_i, neighbor in enumerate(neighbors):
                        neighbor_op_matrix = np.zeros((syn_data.input_size, syn_data.input_size))
                        neighbor_program_matrix = syn_data.gen_program_matrix(neighbor)
                        syn_data.place_scanlines(neighbor_op_matrix, neighbor)
                        blank = np.zeros(program_matrix.shape).squeeze()
                        correct_program = syn_data.place_scanlines(blank, program_vars.tolist()[0])
                        program_delta_matrix = np.concatenate((neighbor_program_matrix, neighbor_op_matrix, correct_program), axis=1)
                        program_train_matrix = np.concatenate((input_matrix.squeeze(), output_matrix.squeeze(), neighbor_program_matrix, neighbor_op_matrix), axis=1)
                        neighbor_model_input[n_i] = program_train_matrix

                    sort_pairs = []
                    for a in range(len(neighbors)):
                        for b in range(a+1, len(neighbors)):
                            sort_pairs.append([a, b])

                    comparator_results = np.zeros(len(neighbors))
                    z_f = 0
                    for pair in sort_pairs:
                        start_time = time.time()

                        input = np.stack([neighbor_model_input[pair[0]], neighbor_model_input[pair[1]]], axis=1)
                        input = np.reshape(input, (1, 2, 112, 28))
                        input = torch.from_numpy(input)
                        input = input.float()
                        input = input.to(device)
                        p_h = program_state_model(input)
                        sort_permutation = p_h[0]
                        # print(sort_permutation)
                        # comparator '0' means  a < b (I think)
                        comparator = sort_permutation[0].argmax()
                        # print(comparator)
                        # print(f'{sort_files[0][comparator]} > {sort_files[0][(comparator+1)%2]}')
                        # print(f'{neighbors[sort_pairs[z_f][comparator]]} > {neighbors[sort_pairs[z_f][(comparator+1)%2]]}')
                        comparator_results[sort_pairs[z_f][comparator]] = comparator_results[sort_pairs[z_f][comparator]] + 1
                        # end_time = time.time()
                        # print(f"Search operator execution time: {(end_time - start_time) * 1000:.2f} ms")
                        z_f = z_f + 1

                    # print(comparator_results)
                    sort_results = [(comparator_results[i], neighbors[i]) for i in range(len(neighbors))]
                    sort_results.sort(key=lambda x:x[0])
                    # print(sort_results)
                    
                    # for s_i, sort in enumerate(sort_results):
                    #     render_program(f"spell_debug/sort_{path_node}_{s_i}.png", sort_results[s_i][1], program_vars)
                    render_program(f"spell_debug/sort_{path_node}.png", sort_results[0][1], program_vars)

                    program_open_set = sort_results[0][1]


x = 0
y = 0
width = 0
height = 0

program_open_set_vars = [x, y, width, height]
deploy_neural_graphics_pathfinder(program_open_set_vars)
