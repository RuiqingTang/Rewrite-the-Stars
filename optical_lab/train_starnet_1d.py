"""
训练和测试StarNet1D模型的脚本。
处理CSV数据集，其中前16列是特征，第17列是类别标签(共6个类别)。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from starnet_1d import starnet1d_s1, starnet1d_s2, starnet1d_s3

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定义数据集类
class VectorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回形状为 [1, 16] 的特征向量和对应的标签
        return self.features[idx].unsqueeze(0), self.labels[idx]

# 加载和预处理数据
def load_data(csv_path, test_size=0.2, val_size=0.1):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 分离特征和标签
    X = df.iloc[:, :16].values  # 前16列是特征
    y = df.iloc[:, 16].values   # 第17列是标签
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_train_val
    )
    
    # 创建数据集
    train_dataset = VectorDataset(X_train, y_train)
    val_dataset = VectorDataset(X_val, y_val)
    test_dataset = VectorDataset(X_test, y_test)
    
    return train_dataset, val_dataset, test_dataset, scaler

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10):
    model.to(device)
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 早停策略
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_starnet1d_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses, train_accs, val_accs

# 测试函数
def test(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    return test_loss, test_acc, all_preds, all_labels

# 绘制训练过程
def plot_training(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# 主函数
def main():
    set_seed(42)
    
    # 配置参数
    csv_path = 'd:/Projects/Python_projects/optical_lab/Rewrite-the-Stars/optical_lab/synthetic_dataset.csv'  
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    patience = 15
    model_type = 's1'  # 可选: 's1', 's2', 's3'
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    train_dataset, val_dataset, test_dataset, scaler = load_data(csv_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    if model_type == 's1':
        model = starnet1d_s1(num_classes=6)  # 6个类别
    elif model_type == 's2':
        model = starnet1d_s2(num_classes=6)
    else:
        model = starnet1d_s3(num_classes=6)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, criterion, optimizer, device, epochs, patience
    )
    
    # 绘制训练过程
    plot_training(train_losses, val_losses, train_accs, val_accs)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_starnet1d_model.pth'))
    test_loss, test_acc, all_preds, all_labels = test(model, test_loader, criterion, device)
    
    # 保存模型和预处理器
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
    }, 'starnet1d_model_full.pth')
    
    print(f'Final test accuracy: {test_acc:.2f}%')
    print('模型已保存为 best_starnet1d_model.pth 和 starnet1d_model_full.pth')

if __name__ == '__main__':
    main()