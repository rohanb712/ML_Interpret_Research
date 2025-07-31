import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST(root='data/', train=True, download=True, transform=transform)

train_set, val_set = random_split(dataset, [50000, 10000])

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

test_set = MNIST(root='data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class FFN(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(128, 64)
        self.act2 = activation_fn
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    """
    Evaluate model on test set and return accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

print("Creating ReLU and GELU models...")
relu_model = FFN(nn.ReLU())
gelu_model = FFN(nn.GELU())

print("\n" + "="*50)
print("Training ReLU Model")
print("="*50)
relu_train_losses, relu_val_accuracies = train_model(relu_model, train_loader, val_loader)

print("\n" + "="*50)
print("Training GELU Model")
print("="*50)
gelu_train_losses, gelu_val_accuracies = train_model(gelu_model, train_loader, val_loader)

print("\n" + "="*50)
print("Final Test Results")
print("="*50)
relu_test_accuracy = evaluate_model(relu_model, test_loader)
gelu_test_accuracy = evaluate_model(gelu_model, test_loader)

print(f"ReLU Model Test Accuracy: {relu_test_accuracy:.2f}%")
print(f"GELU Model Test Accuracy: {gelu_test_accuracy:.2f}%")

print(f"\nFinal Validation Accuracies:")
print(f"ReLU: {relu_val_accuracies[-1]:.2f}%")
print(f"GELU: {gelu_val_accuracies[-1]:.2f}%")

if gelu_test_accuracy > relu_test_accuracy:
    print(f"\nGELU wins by {gelu_test_accuracy - relu_test_accuracy:.2f} percentage points!")
else:
    print(f"\nReLU wins by {relu_test_accuracy - gelu_test_accuracy:.2f} percentage points!")