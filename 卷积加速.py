import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
import matplotlib.pyplot as plt
from copy import deepcopy

# Define Modular Convolutional Layer
class ModularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, modulus=17):
        super(ModularConv2d, self).__init__()
        self.modulus = modulus
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x = self.conv(x)
        return x % self.modulus

# Training and Evaluation Function
def train_and_evaluate(model, trainloader, testloader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_time = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()

        # Train
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        # Calculate train loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        # Calculate validation loss and accuracy
        val_loss = val_loss / len(testloader)
        val_acc = 100. * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        train_time += time.time() - start_time
        print(f"Epoch {epoch+1} completed, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    return train_time, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == '__main__':
    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Define ResNet models
    # Traditional ResNet-18
    model_traditional = resnet18(num_classes=10)

    # Modular ResNet-18
    model_modular = deepcopy(model_traditional)
    modules_to_replace = []
    for name, module in model_modular.named_modules():
        if isinstance(module, nn.Conv2d):
            modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        parent_module = model_modular
        *path, last = name.split('.')
        for part in path:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, last, ModularConv2d(
            module.in_channels, module.out_channels,
            module.kernel_size, module.stride,
            module.padding, modulus=17))

    # Run experiments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train traditional convolution model
    time_traditional, train_losses_traditional, val_losses_traditional, train_acc_traditional, val_acc_traditional = train_and_evaluate(model_traditional, trainloader, testloader, device)

    # Train modular convolution model
    time_modular, train_losses_modular, val_losses_modular, train_acc_modular, val_acc_modular = train_and_evaluate(model_modular, trainloader, testloader, device)

    # Visualization
    # Training loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_traditional, label='Traditional Conv Train Loss', color='blue')
    plt.plot(train_losses_modular, label='Modular Conv Train Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.show()

    # Validation loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_traditional, label='Traditional Conv Validation Loss', color='blue')
    plt.plot(val_losses_modular, label='Modular Conv Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.show()

    # Training accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_traditional, label='Traditional Conv Train Accuracy', color='blue')
    plt.plot(train_acc_modular, label='Modular Conv Train Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.show()

    # Validation accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(val_acc_traditional, label='Traditional Conv Validation Accuracy', color='blue')
    plt.plot(val_acc_modular, label='Modular Conv Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.show()

    # Overall training time and accuracy comparison
    methods = ['Traditional Conv', 'Modular Conv']
    times = [time_traditional, time_modular]
    accuracies = [val_acc_traditional[-1], val_acc_modular[-1]]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(methods, times, color=['blue', 'orange'], alpha=0.6)
    ax1.set_ylabel('Training Time (s)', color='blue')
    ax1.set_xlabel('Model Type')

    ax2 = ax1.twinx()
    ax2.plot(methods, accuracies, color='green', marker='o', label='Accuracy', linestyle='--')
    ax2.set_ylabel('Accuracy (%)', color='green')

    plt.title('Overall Training Time and Final Accuracy Comparison')
    plt.show()
