import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neuralsort import NeuralSort

class TransformerNeuralSort(nn.Module):
    def __init__(self, l=1, final_dim=1):
        super(TransformerNeuralSort, self).__init__()
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

if __name__ == "__main__":
    # Set up data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform), batch_size=64, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformerNeuralSort().to(device)
    neural_sort = NeuralSort().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
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
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

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

torch.save(model.state_dict(), "/arc/pytorch_mnist_vit.pth")