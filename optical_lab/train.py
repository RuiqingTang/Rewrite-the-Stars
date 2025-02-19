#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2025/02/14 15:42:09
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # 忽略 UserWarning
# 来自timm忽略
warnings.filterwarnings("ignore", message=".*timm.*")
from FocalLoss import FocalLoss1

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
data_dir = 'D:\\Projects\\Python_projects\\optical_lab\\zsh_software\\Algorithms\\Data_for_AI\\Fine_tune_data\\Split_data' 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=(x == 'train'), num_workers=4)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

# 加载模型
from starnet import starnet_s1
model = starnet_s1(pretrained=False, num_classes=len(class_names))

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss1()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和验证函数
def train_model(model, criterion, optimizer, num_epochs=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 初始化记录器
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 将acc移动到CPU并转换为NumPy数组
            epoch_acc = epoch_acc.cpu().numpy()

            # 记录统计数据
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model, train_losses, val_losses, train_accs, val_accs

# 绘制曲线图
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

        # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.title('Loss vs. Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train accuracy')
    plt.plot(epochs, val_accs, label='Validation accuracy')
    plt.title('Accuracy vs. Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 测试模型
def test_model(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    running_corrects = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    test_acc = test_acc.cpu().numpy()  
    print(f'Test Accuracy: {test_acc:.4f}')

# 保存模型
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

# 主函数
if __name__ == '__main__':
    # 训练模型
    model, train_losses, val_losses, train_accs, val_accs = train_model(model, criterion, optimizer, num_epochs=15)
    # 绘制损失和准确率曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs)
    # 测试模型
    test_model(model)
    # 保存模型
    save_model(model, 'starnet_s1.pth')