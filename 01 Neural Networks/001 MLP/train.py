#==========================================#
# Title:  Image classification with MLP (train)
#==========================================#
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))

batch_size = 128
num_classes = 10  # datasets have image file of number 0 to 9
epochs = 10
seed = 42

"""
Step1: Define two MLP models with different architectures
 Model A: shallow & wide  (1 hidden layer)
 Model B: deep & narrow   (3 hidden layers)
"""
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, name, save_path, train_loader, val_loader):
    """
    Trains for `epochs`, reporting train metrics (computed online during
    the epoch) and validation metrics (full pass, no grad) each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f'[{name}] Epoch [{epoch+1:02d}/{epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f'[{name}] best val acc {best_val_acc*100:.2f}% saved to {save_path}')


if __name__ == "__main__":
    torch.manual_seed(seed)

    """
    Step2: load datasets and split train -> train/validation
     train 48,000 / validation 12,000 / test 10,000 (test is used in test.py)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)

    train_dataset, val_dataset = random_split(
        full_train_dataset, [48000, 12000],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    """
    Step3: train both models and save their weights
    """
    model_a = ModelA()
    model_b = ModelB()

    print("=== Training Model A (shallow & wide: 784-256-10) ===")
    train_model(model_a, "Model A", "model_a.pth", train_loader, val_loader)

    print("=== Training Model B (deep & narrow: 784-256-128-64-10) ===")
    train_model(model_b, "Model B", "model_b.pth", train_loader, val_loader)
