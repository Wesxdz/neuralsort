# After sorting the graphics programs, the top selected variable to modify becomes the 'current node'

import numpy as np
from PIL import Image
import random
import syn_data
import os
from gpdl import ScanlinesDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from neuralsort import NeuralSort

class GraphicsProgramHypothesisSort(nn.Module):
    def __init__(self, l=1, final_dim=1):
        super(GraphicsProgramHypothesisSort, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.transformer = nn.TransformerEncoderLayer(d_model=64*4*4, nhead=8, dim_feedforward=64*4*4, dropout=0.1)
        self.fc2 = nn.Linear(64*4*4, final_dim)
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
def init_neighbors(program_vars, var_heads):
    neighbors = []
    node_to_var_delta_index = {} # Map the variable delta to a node, so that if it is top sorted this variable can be further iterated
    for v_i in range(len(program_vars)):
        init_val = int(max_val/2)
        var_heads[v_i] = init_val
        node = [init_val if i == v_i else 0 for i in range(len(program_vars))]
        node_to_var_delta_index[hash(tuple(node))] = v_i
        neighbors.append(node)
    return neighbors, node_to_var_delta_index

# Each variable has a 'current head' which is used as the default value when generating new neighbors...
def populate_neighbors(program_vars, var_heads, top_sorted_var_delta_index, node_to_var_delta_index):
    neighbors = []
    current_head = var_heads[top_sorted_var_delta_index]
    left_delta = -int(current_head / 2)
    right_delta = int((max_val - current_head) / 2)
    for delta in set([-1, left_delta, right_delta, 1]):
        new_head = current_head + delta
        # Ensure the new head is within bounds
        if 0 <= new_head <= max_val:
            node = [new_head if i == top_sorted_var_delta_index else var_heads[i] for i in range(len(program_vars))]
            node_to_var_delta_index[hash(tuple(node))] = top_sorted_var_delta_index
            neighbors.append(node)
    return neighbors


def split_dataset(dataset, test_ratio=0.2):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
        
def train_neural_sort_graphics_program_pathfinder(program_open_set_vars, max_val, max_iterations):
    root_dir = 'graphics_programs'
    num_samples = 100
    batch_size = 32

    dataset = ScanlinesDataset(root_dir, num_samples)
    train_dataset, test_dataset = split_dataset(dataset)

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GraphicsProgramHypothesisSort().to(device)
    neural_sort = NeuralSort().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(100):
        model.train()
        for batch_idx, (input_matrix, output_matrix, op_matrix, program_matrix) in enumerate(train_loader):
            datum = [input_matrix, output_matrix, op_matrix, program_matrix]
            for d in datum:
                d = d.to(device)

# BEGIN PATHFINDER --------------------------------------------
            # TODO: Generate new variables
            program_open_set_vars = [x, y, width, height]
            max_iterations = 20

            programs = []
            # Variable heads contains the 'current head' variable value for each variable index
            var_heads = {}
            # Node to variable delta index indicates which variable index was modified to generate the new neighbor node
            # TODO: Track prev node
            neighbors, node_to_var_delta_index = init_neighbors(program_open_set_vars, var_heads)
            for _ in range(max_iterations):

                sort_pairs = []
                for a in range(len(neighbors)):
                    for b in range(a+1, len(neighbors)):
                        sort_pairs.append([a, b])

                # TODO: Tensorflow comparator sort
                        
                
                top_sorted_node_index = random.choice(range(len(neighbors)))

                # The variable index within the current node which changed
                top_sorted_var_delta_index = node_to_var_delta_index[hash(tuple(neighbors[top_sorted_node_index]))]

                # Visualize

                # print(top_sorted_var_delta_index)
                var_heads[top_sorted_var_delta_index] = neighbors[top_sorted_node_index][top_sorted_var_delta_index]
                # Populate neighbors for the current node
                new_neighbors = populate_neighbors(program_open_set_vars, var_heads, top_sorted_var_delta_index, node_to_var_delta_index)
                top = neighbors.pop(top_sorted_node_index)
                program_matrix = syn_data.gen_program_matrix(top)

                input_transform = syn_data.place_scanlines(input.copy(), top)
                diff = input_transform - output
                loss = abs(diff).sum()
                print(loss)

                # print(top, neighbors)
                # TODO: Visited
                for node in new_neighbors:
                            node_tuple = tuple(node)
                            if node_tuple not in [tuple(n) for n in neighbors]:
                                neighbors.append(node)

                op_matrix = np.zeros((syn_data.input_size, syn_data.input_size))
                syn_data.place_scanlines(op_matrix, top)
                program_delta_matrix = np.concatenate((program_matrix, op_matrix), axis=1)
                programs.append(program_delta_matrix)

                if _ == max_iterations - 1:
                    # print(program_matrix)
                    program_search = np.concatenate([program_delta for program_delta in programs], axis=0)
                    concat_img = Image.fromarray((program_search * 255).astype(np.uint8))
                    concat_img.save(f'program.png')
                    solution = [var_heads[i] for i in range(len(program_open_set_vars))]
                    return solution
                
# END PATHFINDER --------------------------------------------
            
            data = data.view(-1, 2, 28, 28)
            target = target.reshape(-1, 2)
            P_true = neural_sort(target.unsqueeze(-1))

            optimizer.zero_grad()

            output = model(data)
            logits = torch.log(output + 1e-20)
            loss = loss_fn(logits, P_true)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_matrix, output_matrix, op_matrix, program_matrix in test_loader:
                datum = [input_matrix, output_matrix, op_matrix, program_matrix]
                for d in datum:
                    d = d.to(device)

                data = data.view(-1, 2, 28, 28)
                target = target.reshape(-1, 2)
                P_true = neural_sort(target.unsqueeze(-1))

                output = model(data)
                logits = torch.log(output + 1e-20)
                test_loss += loss_fn(logits, P_true).item()
                pred = torch.argmax(logits, dim=1)
                pred_min_index = torch.argmin(pred, dim=1)
                target_min_index = torch.argmax(target, dim=1)

                correct += pred_min_index.eq(target_min_index).sum().item()
                total += pred_min_index.size(0)  # Get the batch size

        accuracy = correct / total
        print(f'Epoch {epoch+1}, Test Loss: {test_loss / len(test_loader)}', f'Test Accuracy: {accuracy:.2f}%')
        
x = 0
y = 0
width = 0
height = 0

# Perform the mock search
program_open_set_vars = [x, y, width, height]
max_iterations = 20
solution = train_neural_sort_graphics_program_pathfinder(program_open_set_vars, max_val, max_iterations)
print("Solution found:", solution)