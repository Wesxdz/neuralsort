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

def wait_for_unix_socket(path, timeout=30):
    start_time = time.time()
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(path)
            return True
        except socket.error:
            pass
        if time.time() - start_time > timeout:
            return False
        time.sleep(1)

def split_dataset(dataset, test_ratio=0.2):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

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

def train_neural_graphics_pathfinder():
    root_dir = './spells'
    dataset = HumanSpellcastingDataset(root_dir)
    train_dataset, test_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    program_state_model = GraphicsProgramHypothesisSort().to(device)
    neural_sort = NeuralSort().to(device)
    optimizer = torch.optim.Adam(program_state_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1):
            program_state_model.train()
            for batch_idx, (graphics_program, sort_stages, program_deltas) in enumerate(train_loader):
                # print(program_deltas)
                data = program_deltas
                data = data.view(-1, 2, 28*4, 28)
                # print(type(sort_stages))
                # print(sort_stages)
                priorities = sort_stages.reshape(-1, 2)
                # print(priorities)
                P_true = neural_sort(priorities.unsqueeze(-1))

                optimizer.zero_grad()

                output = program_state_model(data)
                logits = torch.log(output + 1e-20)
                loss = loss_fn(logits, P_true)
                loss.backward()

                optimizer.step()
                # print(P_true)
                print(loss)

    torch.save(program_state_model.state_dict(), 'program_state_model.pth')
    torch.save(neural_sort.state_dict(), 'neural_sort.pth')


def sort_learning_from_human_spellcasting(program_open_set_vars):
    root_dir = 'graphics_programs'
    # Determine based on spellcasting...
    start_idx = len(os.listdir("./spells"))
    print("Start at " + str(start_idx))
    num_samples = 100

    dataset = OnlineGraphicsDataset(root_dir, start_idx, num_samples)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    neighbor_model_input = {}
    for batch_idx, (input_matrix, output_matrix, op_matrix, program_matrix, program_vars) in enumerate(train_loader):
        
        paths = ['./sort_stage/*', './hf/*', './neighbors/*']

        for path in paths:
            dir_paths = glob.glob(path)
            for dir_path in dir_paths:
                if os.path.isdir(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {str(e)}")

        programs = []
        # Node to variable delta index indicates which variable index was modified to generate the new neighbor node
        # TODO: Track prev node
        neighbors, node_to_var_delta_index = init_neighbors(program_open_set_vars)
        sort_stage = 0
        graphics_program_solved = False

        x = random.randint(0, 32)
        y = random.randint(0, 32)
        width = random.randint(0, 32)
        height = random.randint(0, 32)

        program_open_set_vars = [x, y, width, height]

        while (not graphics_program_solved):
            print(program_open_set_vars)
            print(neighbors)
            os.makedirs(f"sort_stage/{sort_stage}", exist_ok=True)
            os.makedirs(f"neighbors/{sort_stage}", exist_ok=True)
            os.makedirs(f"sort_stage/{str(sort_stage+1)}", exist_ok=True)
            for n_i, neighbor in enumerate(neighbors):
                neighbor_op_matrix = np.zeros((syn_data.input_size, syn_data.input_size))
                neighbor_program_matrix = syn_data.gen_program_matrix(neighbor)
                syn_data.place_scanlines(neighbor_op_matrix, neighbor)
                blank = np.zeros(program_matrix.shape).squeeze()
                correct_program = syn_data.place_scanlines(blank, program_vars.tolist()[0])
                program_delta_matrix = np.concatenate((neighbor_program_matrix, neighbor_op_matrix, correct_program), axis=1)
                program_train_matrix = np.concatenate((input_matrix.squeeze(), output_matrix.squeeze(), neighbor_program_matrix, neighbor_op_matrix), axis=1)
                neighbor_model_input[n_i] = program_train_matrix

                # TODO: Make this more efficient, you only need to save io pair once...
                np.save(f'neighbors/{sort_stage}/n_{n_i}.npy', program_train_matrix)
                np.save(f'neighbors/{sort_stage}/vars_{n_i}.npy', np.array(neighbor))

                render_program(f'sort_stage/{sort_stage}/program_{n_i}.png', neighbor, program_vars)
                # TODO: Save neighbor vars
            
            if sort_stage == 0:
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect("/tmp/my_socket")
                message = "init_sort"
                client_socket.send(message.encode())
                while True:
                    response = client_socket.recv(256).decode()
                    if response:
                        client_socket.close()
                        break
                new_neighbors = populate_neighbors(neighbors[0], 0, node_to_var_delta_index)

            if sort_stage > 0:
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect("/tmp/my_socket")
                message = "next_stage"
                client_socket.send(message.encode())
                while True:
                    response = client_socket.recv(256).decode()
                    if response:
                        client_socket.close()
                        break

                with open(f"hf/{sort_stage}/var_expand.txt") as f:
                    var_priorities = [(int(data.split(" ")[0]), int(data.split(" ")[1])) for data in f.readlines()]
                    top_sorted_var_expand = var_priorities[0][0]
                # Assume human_sorted_program_states has not been set, so use neighbors[0]
                new_neighbors = populate_neighbors(neighbors[0], top_sorted_var_expand, node_to_var_delta_index)

            for n_i, node in enumerate(new_neighbors):
                render_program(f'sort_stage/{str(sort_stage+1)}/program_{n_i}.png', node, program_vars)
            # After rendering default sort, we need to update
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect("/tmp/my_socket")
            message = "next_stage_default_sort"
            client_socket.send(message.encode())

            # Human feedback: 
            # Identify P_true sort for program state root
            # Identify P_true sort for var expand
            # TODO: Model that generates var expand deltas
                
                    
            while True:
                # Receive data from the server
                response = client_socket.recv(256).decode()
                print("Response from server:", response)

                if response:
                    if response == "next_graphics_program":
                        graphics_program_solved = True
                        break
                    if response in ["next_stage", "feedback_update"]:
                        with open(f"hf/{sort_stage}/var_expand.txt") as f:
                            var_priorities = [(int(data.split(" ")[0]), int(data.split(" ")[1])) for data in f.readlines()]
                            top_sorted_var_expand = var_priorities[0][0]
                        with open(f"hf/{sort_stage}/program_state.txt") as f:
                            human_sorted_program_states = [(int(data.split(" ")[0]), int(data.split(" ")[1])) for data in f.readlines()]
                            print(neighbors)
                            print(human_sorted_program_states)
                            new_neighbors = populate_neighbors(neighbors[human_sorted_program_states[0][0]], top_sorted_var_expand, node_to_var_delta_index)

                    if response == "next_stage":
                        # For a first pass on training this model, we just replace all neighbors
                        print(neighbors)
                        print(top_sorted_var_expand)
                        print(var_priorities)
                        neighbors = new_neighbors
                        print("Neighbors have been updated to:")
                        print(neighbors)
                        sort_stage = int(sort_stage) + 1
                        break
                    elif response == "feedback_update":
                        
                        dir_path = f'sort_stage/{str(sort_stage+1)}/'
                        for filename in os.listdir(dir_path):
                            os.remove(os.path.join(dir_path, filename))

                        for n_i, node in enumerate(new_neighbors):
                            render_program(f'sort_stage/{str(sort_stage+1)}/program_{n_i}.png', node, program_vars)
                        print("Rerendered neighbor programs")
                        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        client_socket.connect("/tmp/my_socket")
                        message = "reload_programs"
                        client_socket.send(message.encode())

        # Export sort sequence to spells
        gfx_program_index = start_idx + batch_idx
        spell_dir = f"./spells/{gfx_program_index}"
        os.makedirs(spell_dir, exist_ok=True)
        paths = ["./hf", "./neighbors"]
        for dir in paths:
            for dir_name in os.listdir(dir):
                dir_path = os.path.join(dir, dir_name)
                if os.path.isdir(dir_path):
                    # Check if the subdirectory already exists in the destination directory
                    dest_dir_path = os.path.join(spell_dir, dir_name)
                    if os.path.exists(dest_dir_path):
                        # Merge the contents of the two directories
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            shutil.move(file_path, dest_dir_path)
                    else:
                        # Copy the entire directory to the destination
                        shutil.move(dir_path, spell_dir)
        paths = ['./sort_stage/*']
        for path in paths:
            dir_paths = glob.glob(path)
            for dir_path in dir_paths:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)

x = 0
y = 0
width = 0
height = 0

program_open_set_vars = [x, y, width, height]
sort_learning_from_human_spellcasting(program_open_set_vars)
# train_neural_graphics_pathfinder()