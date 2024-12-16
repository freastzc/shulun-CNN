import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc  # 用于生成 Sobol 序列


# 定义传统数据增强方法
def traditional_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# 定义基于数论的数据增强方法
def number_theory_augmentation():
    def mod_rotation(img):
        angles = [(i * 7) % 360 for i in range(10)]  # 数论生成旋转角度
        angle = angles[torch.randint(0, len(angles), (1,)).item()]
        return transforms.functional.rotate(img, angle)

    def sobol_crop(img):
        # 使用 Sobol 序列生成裁剪比例
        sampler = qmc.Sobol(d=2)
        sobol_seq = sampler.random_base2(m=10)  # 生成 10 个样本的 Sobol 序列
        crop_idx = torch.randint(0, len(sobol_seq), (1,)).item()
        crop_size = sobol_seq[crop_idx][0] * 0.5 + 0.5  # 缩放到合理范围
        return transforms.functional.resized_crop(img, 0, 0, int(32 * crop_size), int(32 * crop_size), size=(32, 32))

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(mod_rotation),
        transforms.Lambda(sobol_crop),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# 加载数据集
def get_dataloaders(transform, batch_size=64):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# 定义训练和测试函数
def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


# 主实验流程
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs = 10

    # 数据增强对比
    augmentations = {
        'Traditional': traditional_augmentation(),
        'Number Theory': number_theory_augmentation()
    }

    results = {}

    for name, transform in augmentations.items():
        print(f"\n=== Training with {name} Augmentation ===")
        train_loader, test_loader = get_dataloaders(transform, batch_size)

        # 加载 ResNet-18 模型并使用预训练权重
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

        # 将最后的全连接层修改为适应 CIFAR-10 的 10 类输出
        model.fc = nn.Linear(model.fc.in_features, 10)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = train_model(model, criterion, optimizer, train_loader, device)
            test_loss, test_acc = evaluate_model(model, criterion, test_loader, device)
            elapsed_time = time.time() - start_time

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
                  f"Time: {elapsed_time:.2f}s")

        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }

    # 可视化结果
    sns.set(style="darkgrid")

    # 损失曲线
    plt.figure(figsize=(10, 5))
    for name, result in results.items():
        plt.plot(range(1, num_epochs + 1), result['train_losses'], label=f'{name} Train Loss')
        plt.plot(range(1, num_epochs + 1), result['test_losses'], label=f'{name} Test Loss', linestyle='--')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 准确率曲线
    plt.figure(figsize=(10, 5))
    for name, result in results.items():
        plt.plot(range(1, num_epochs + 1), result['train_accuracies'], label=f'{name} Train Accuracy')
        plt.plot(range(1, num_epochs + 1), result['test_accuracies'], label=f'{name} Test Accuracy', linestyle='--')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
