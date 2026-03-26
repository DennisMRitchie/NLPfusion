import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Example Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example Model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(100, 50)  # Example layer
        self.layer2 = nn.Linear(50, 10)   # Output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Trainer class
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data, labels in self.val_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(self.val_loader.dataset)
        print(f"Accuracy: {accuracy * 100:.2f}%")

# Main Execution
if __name__ == '__main__':
    # Dummy data
    train_data = np.random.rand(1000, 100).astype(np.float32)
    train_labels = np.random.randint(0, 10, 1000)
    val_data = np.random.rand(200, 100).astype(np.float32)
    val_labels = np.random.randint(0, 10, 200)

    # Create datasets and dataloaders
    train_dataset = MyDataset(train_data, train_labels)
    val_dataset = MyDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, criterion, optimizer
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    trainer.train(epochs=10)
    trainer.evaluate()