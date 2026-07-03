import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from cnn_network import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
learning_rate = 0.0002
num_epoch = 15

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

cifar_train = datasets.CIFAR10(root="../data/", train=True, transform=transform, download=True)
cifar_test = datasets.CIFAR10(root="../data/", train=False, transform=transform, download=True)

extract_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False, drop_last=False)
extract_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, drop_last=False)

model = CNN(batch_size=batch_size).to(device)

def extract_features(loader):
    model.layer.eval()
    features, labels = [], []
    with torch.no_grad():
        for image, label in loader:
            feat = model.layer(image.to(device))
            features.append(feat.cpu())
            labels.append(label)
    return torch.cat(features), torch.cat(labels)

print("Extracting ResNet50 features (one-time pass)...")
train_features, train_labels = extract_features(extract_train_loader)
test_features, test_labels = extract_features(extract_test_loader)

train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False, drop_last=True)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc_layer.parameters(), lr=learning_rate)

loss_arr =[]
for i in range(num_epoch):
    epoch_loss = 0
    corrects = 0
    total = 0
    for j,[feat,label] in enumerate(train_loader):
        x = feat.to(device)
        y= label.to(device)

        optimizer.zero_grad()

        output = model.fc_layer(x)

        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(output, 1)
        corrects += (predicted == y).sum().item()
        total += y.size(0)

    epoch_loss /= len(train_loader)
    epoch_acc = corrects / total * 100
    print(f"epoch: [{str(i+1).zfill(2)}/{num_epoch}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}")

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for feat,label in test_loader:
        x, y = feat.to(device), label.to(device)
        output = model.fc_layer(x)

        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == y).sum().float()

    print(f"Accuracy of Test Data: {100*correct/total:.5f}%")
