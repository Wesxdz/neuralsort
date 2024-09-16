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
        print(x.shape)
        x = x.view(-1, 1, 28, 28)  # Reshape to (M * n, 1, 28, 28)
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4 * 4 * 64)  # Flatten to (M * n, 4 * 4 * 64)
        x = nn.functional.relu(self.fc1(x))
        scores = self.fc2(x)  # Output shape: (M * n, 1)
        scores = scores.view(M, n, 1)  # Reshape to (M, n, 1)
        print(scores.shape)
        P_hat = self.neural_sort(scores)  # Output shape: (M, n, n)
        return P_hat
    
# Set up data loaders
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform), batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
model = NeuralSortMNIST()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'Epoch {epoch+1}, Test Loss: {test_loss / len(test_loader)}', f'Test Accuracy: {accuracy:.2f}%')