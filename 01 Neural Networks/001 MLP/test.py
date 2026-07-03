#==========================================#
# Title:  Image classification with MLP (test)
#==========================================#
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train import ModelA, ModelB, batch_size

"""
Step1: load the held-out test dataset (10,000 images, never seen during training)
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def evaluate(model, loader, num_examples=8):
    model.eval()
    correct, total = 0, 0
    examples = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if len(examples) < num_examples:
                for i in range(len(labels)):
                    if len(examples) < num_examples:
                        examples.append((images[i], labels[i], predicted[i]))
                    else:
                        break

    return correct / total, examples


def show_predictions(examples, title):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, (image, label, prediction) in enumerate(examples):
        image_np = image.squeeze().numpy() * 0.5 + 0.5
        ax = axes[i // 4, i % 4]
        ax.imshow(image_np, cmap='gray')
        ax.set_title(f'True: {label.item()}\nPred: {prediction.item()}', fontsize=10)
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()


if __name__ == "__main__":
    """
    Step2: load the trained weights for both models
    """
    model_a = ModelA()
    model_a.load_state_dict(torch.load('model_a.pth'))

    model_b = ModelB()
    model_b.load_state_dict(torch.load('model_b.pth'))

    """
    Step3: evaluate both models on the test set and compare
    """
    acc_a, examples_a = evaluate(model_a, test_loader)
    acc_b, examples_b = evaluate(model_b, test_loader)

    print(f'Model A (shallow & wide, 784-256-10)        Test Accuracy: {acc_a*100:.2f}%')
    print(f'Model B (deep & narrow, 784-256-128-64-10)   Test Accuracy: {acc_b*100:.2f}%')

    diff = (acc_a - acc_b) * 100
    if abs(diff) < 1e-9:
        print('Both models achieved the same accuracy.')
    else:
        better = 'Model A' if diff > 0 else 'Model B'
        print(f'{better} performed better by {abs(diff):.2f} percentage points.')

    show_predictions(examples_a, 'Model A predictions')
    show_predictions(examples_b, 'Model B predictions')
    plt.show()
