import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

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

        input_matrix = np.load(input_path)
        output_matrix = np.load(output_path)
        op_matrix = np.load(op_path)
        program_matrix = np.load(program_path)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_matrix).float()
        output_tensor = torch.from_numpy(output_matrix).float()
        op_tensor = torch.from_numpy(op_matrix).float()
        program_tensor = torch.from_numpy(program_matrix).float()

        return {
            'input': input_tensor,
            'output': output_tensor,
            'op': op_tensor,
            'program': program_tensor
        }


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