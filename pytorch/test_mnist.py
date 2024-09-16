import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neuralsort import NeuralSort

transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform), batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralSort().to(device)
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(32, 2, 28, 28)
    target = target.reshape(-1, 2)
    P_true = model(target)
    break