import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import hashlib

class HumanSpellcastingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.num_samples = 0
        self.samples = []
        
        graphics_programs = sorted(os.listdir(root_dir), key=lambda x: int(x))
        for graphics_program in graphics_programs:
            graphics_program_dir_path = os.path.join(root_dir, graphics_program)

            sort_stages = sorted(os.listdir(graphics_program_dir_path), key=lambda x: int(x))

            # Initialize the previous sort stage's program state
            prev_program_state = None

            # Iterate through the sort stages in reverse order
            for sort_stage in sort_stages:
                sort_stage_dir_path = os.path.join(graphics_program_dir_path, sort_stage)

                # Load program state and neighbor indices
                with open(os.path.join(sort_stage_dir_path, "program_state.txt"), "r") as f:
                    program_state = [(int(data.split(" ")[0]), int(data.split(" ")[1])) for data in f.readlines()]

                # If this is not the first sort stage, generate pairwise comparisons with the previous sort stage
                if prev_program_state is not None:
                    for a in range(len(program_state)):
                        for b in range(len(prev_program_state)):
                            # Load numpy files for the pair
                            npy_file_a = os.path.join(sort_stage_dir_path, f"n_{a}.npy")
                            npy_file_b = os.path.join(graphics_program_dir_path, str(int(sort_stage) - 1), f"n_{b}.npy")

                            if os.path.exists(npy_file_a) and os.path.exists(npy_file_b):
                                # Deterministic randomization of the order
                                hash_value = int(hashlib.md5(f"{graphics_program}{sort_stage}{a}{b}".encode()).hexdigest(), 16)
                                if hash_value % 2 == 0:
                                    self.samples.append({
                                        "graphics_program": graphics_program,
                                        "sort_stages": np.array([int(sort_stage), int(sort_stage) - 1]),
                                        "program_deltas": [npy_file_a, npy_file_b]
                                    })
                                else:
                                    self.samples.append({
                                        "graphics_program": graphics_program,
                                        "sort_stages": np.array([int(sort_stage) - 1, int(sort_stage)]),
                                        "program_deltas": [npy_file_b, npy_file_a]
                                    })
                                self.num_samples += 1


                # Update the previous sort stage's program state
                prev_program_state = program_state

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        graphics_program = sample["graphics_program"]
        sort_stages = sample["sort_stages"]
        program_deltas = sample["program_deltas"]
        a = np.load(program_deltas[0])
        b = np.load(program_deltas[1])
        c = np.stack((a, b)).astype(np.float32)
        data = torch.from_numpy(c)
        return (graphics_program, sort_stages, data)


class OnlineGraphicsDataset(Dataset):
    def __init__(self, root_dir, start_idx, num_samples):
        self.root_dir = root_dir
        self.start_idx = start_idx
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - input_tensor (Tensor): Input tensor.
        - output_tensor (Tensor): Output tensor.
        - op_tensor (Tensor): Operator tensor.
        - program_tensor (Tensor): Program tensor.
        - program_vars (ndarray): Program variables.
        """
        actual_idx = self.start_idx + idx
        input_path = os.path.join(self.root_dir, f'input_{actual_idx}.npy')
        output_path = os.path.join(self.root_dir, f'output_{actual_idx}.npy')
        op_path = os.path.join(self.root_dir, f'op_{actual_idx}.npy')
        program_path = os.path.join(self.root_dir, f'program_{actual_idx}.npy')
        program_vars_path = os.path.join(self.root_dir, f'program_vars_{actual_idx}.npy')

        input_matrix = np.load(input_path)
        output_matrix = np.load(output_path)
        op_matrix = np.load(op_path)
        program_matrix = np.load(program_path)
        program_vars = np.load(program_vars_path)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_matrix).float()
        output_tensor = torch.from_numpy(output_matrix).float()
        op_tensor = torch.from_numpy(op_matrix).float()
        program_tensor = torch.from_numpy(program_matrix).float()

        return (input_tensor, output_tensor, op_tensor, program_tensor, program_vars)

class ScanlinesDataset(Dataset):
    def __init__(self, root_dir, num_samples):
        self.root_dir = root_dir
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_path = os.path.join(self.root_dir, f'input_{idx}.npy')
        output_path = os.path.join(self.root_dir, f'output_{idx}.npy')
        op_path = os.path.join(self.root_dir, f'op_{idx}.npy')
        program_path = os.path.join(self.root_dir, f'program_{idx}.npy')
        program_vars_path = os.path.join(self.root_dir, f'program_vars_{idx}.npy')

        input_matrix = np.load(input_path)
        output_matrix = np.load(output_path)
        op_matrix = np.load(op_path)
        program_matrix = np.load(program_path)
        program_vars = np.load(program_vars_path)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_matrix).float()
        output_tensor = torch.from_numpy(output_matrix).float()
        op_tensor = torch.from_numpy(op_matrix).float()
        program_tensor = torch.from_numpy(program_matrix).float()

        return (input_tensor, output_tensor, op_tensor, program_tensor, program_vars)


# Train/test split
def split_dataset(dataset, test_ratio=0.2):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Usage
    root_dir = 'graphics_programs'
    num_samples = 100
    batch_size = 32

    dataset = ScanlinesDataset(root_dir, num_samples)
    train_dataset, test_dataset = split_dataset(dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    for batch in train_dataloader:
        inputs = batch['input']
        outputs = batch['output']
        ops = batch['op']
        programs = batch['program']
        # Do something with the batch
        print(inputs.shape, outputs.shape, ops.shape, programs.shape)