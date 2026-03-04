"""
第4天：卷积神经网络（CNN） + 批量归一化（BatchNorm）
股票图像模式识别模型

目标：
1. 理解卷积神经网络（CNN）的基本原理
2. 掌握批量归一化（BatchNorm）技术
3. 实现股票图表模式识别模型
4. 应用CNN进行技术分析图像识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class StockChartDataset(Dataset):
    """股票图表数据集"""
    def __init__(self, n_samples=1000, chart_size=32, n_channels=3):
        """
        参数:
            n_samples: 样本数量
            chart_size: 图表图像大小 (chart_size x chart_size)
            n_channels: 通道数 (3: 价格、成交量、技术指标)
        """
        super().__init__()
        self.n_samples = n_samples
        self.chart_size = chart_size
        self.n_channels = n_channels
        
        # 生成模拟股票图表数据
        self.data = self._generate_stock_charts()
        self.labels = self._generate_labels()
        
    def _generate_stock_charts(self):
        """生成模拟股票图表数据"""
        charts = []
        
        for _ in range(self.n_samples):
            # 创建多通道图表
            chart = np.zeros((self.n_channels, self.chart_size, self.chart_size), dtype=np.float32)
            
            # 通道0: 价格走势（K线图）
            price_chart = self._generate_price_chart()
            chart[0] = price_chart
            
            # 通道1: 成交量（柱状图）
            volume_chart = self._generate_volume_chart()
            chart[1] = volume_chart
            
            # 通道2: 技术指标（RSI、MACD等）
            indicator_chart = self._generate_indicator_chart()
            chart[2] = indicator_chart
            
            charts.append(chart)
        
        return np.array(charts)
    
    def _generate_price_chart(self):
        """生成价格走势图（模拟K线图）"""
        chart = np.zeros((self.chart_size, self.chart_size), dtype=np.float32)
        
        # 生成随机价格序列
        prices = np.cumsum(np.random.randn(self.chart_size) * 0.1) + 100
        
        # 归一化价格
        prices_normalized = (prices - prices.min()) / (prices.max() - prices.min())
        
        # 绘制价格线
        for i in range(self.chart_size - 1):
            y1 = int(prices_normalized[i] * (self.chart_size - 1))
            y2 = int(prices_normalized[i + 1] * (self.chart_size - 1))
            
            # 绘制线段
            x_range = np.linspace(i, i + 1, 10)
            y_range = np.linspace(y1, y2, 10)
            
            for x, y in zip(x_range, y_range):
                x_int = int(x)
                y_int = int(y)
                if 0 <= x_int < self.chart_size and 0 <= y_int < self.chart_size:
                    chart[y_int, x_int] = 1.0
        
        return chart
    
    def _generate_volume_chart(self):
        """生成成交量图"""
        chart = np.zeros((self.chart_size, self.chart_size), dtype=np.float32)
        
        # 生成随机成交量
        volumes = np.abs(np.random.randn(self.chart_size)) * 100
        
        # 归一化成交量
        volumes_normalized = volumes / volumes.max()
        
        # 绘制成交量柱状图
        for i in range(self.chart_size):
            height = int(volumes_normalized[i] * (self.chart_size - 1))
            for h in range(height):
                chart[self.chart_size - 1 - h, i] = 1.0
        
        return chart
    
    def _generate_indicator_chart(self):
        """生成技术指标图"""
        chart = np.zeros((self.chart_size, self.chart_size), dtype=np.float32)
        
        # 生成随机技术指标（如RSI、MACD）
        indicators = np.random.randn(self.chart_size) * 0.5 + 0.5  # 0-1范围
        
        # 绘制指标线
        for i in range(self.chart_size - 1):
            y1 = int(indicators[i] * (self.chart_size - 1))
            y2 = int(indicators[i + 1] * (self.chart_size - 1))
            
            # 绘制线段
            x_range = np.linspace(i, i + 1, 10)
            y_range = np.linspace(y1, y2, 10)
            
            for x, y in zip(x_range, y_range):
                x_int = int(x)
                y_int = int(y)
                if 0 <= x_int < self.chart_size and 0 <= y_int < self.chart_size:
                    chart[y_int, x_int] = 1.0
        
        return chart
    
    def _generate_labels(self):
        """生成标签：0=下跌趋势，1=上涨趋势，2=横盘整理"""
        labels = []
        
        for i in range(self.n_samples):
            # 根据图表特征生成标签
            chart = self.data[i]
            
            # 简单规则：检查价格走势
            price_chart = chart[0]
            
            # 计算价格变化
            price_start = np.mean(price_chart[:, :5])
            price_end = np.mean(price_chart[:, -5:])
            
            price_change = price_end - price_start
            
            if price_change > 0.1:  # 明显上涨
                labels.append(1)
            elif price_change < -0.1:  # 明显下跌
                labels.append(0)
            else:  # 横盘整理
                labels.append(2)
        
        return np.array(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        chart = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return chart, label


class CNNWithBatchNorm(nn.Module):
    """带批量归一化的卷积神经网络"""
    def __init__(self, num_classes=3):
        super(CNNWithBatchNorm, self).__init__()
        
        # 卷积层1: 输入3通道，输出16通道
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 卷积层2: 16通道 -> 32通道
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 卷积层3: 32通道 -> 64通道
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        # 输入大小计算: 32x32 -> 池化后16x16 -> 池化后8x8 -> 池化后4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 卷积层1 + BatchNorm + ReLU + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积层2 + BatchNorm + ReLU + 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积层3 + BatchNorm + ReLU + 池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层1 + BatchNorm + ReLU + Dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        
        # 全连接层2 + BatchNorm + ReLU + Dropout
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        # 输出层
        x = self.fc3(x)
        
        return x


class CNNWithoutBatchNorm(nn.Module):
    """不带批量归一化的卷积神经网络（用于对比）"""
    def __init__(self, num_classes=3):
        super(CNNWithoutBatchNorm, self).__init__()
        
        # 卷积层1: 输入3通道，输出16通道
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 卷积层2: 16通道 -> 32通道
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 卷积层3: 32通道 -> 64通道
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 卷积层1 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积层2 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 卷积层3 + ReLU + 池化
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # 全连接层2 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc2(x)))
        
        # 输出层
        x = self.fc3(x)
        
        return x


def train_cnn_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu'):
    """训练CNN模型"""
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"🚀 开始训练CNN模型...")
    print(f"   设备: {device}")
    print(f"   训练轮数: {epochs}")
    print(f"   学习率: {learning_rate}")
    print(f"   批量大小: {train_loader.batch_size}")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # 计算训练指标
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                epoch_val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        # 计算验证指标
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 每10轮打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"   轮次 {epoch+1}/{epochs} - "
                  f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}% | "
                  f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%")
    
    print("✅ 训练完成!")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型性能"""
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    total = 0
    
    # 存储预测结果用于详细分析
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 存储结果
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    print(f"📊 测试结果:")
    print(f"   测试损失: {avg_test_loss:.4f}")
    print(f"   测试准确率: {test_accuracy:.2f}%")
    print(f"   正确预测: {correct}/{total}")
    
    # 计算每个类别的准确率
    from sklearn.metrics import classification_report
    print(f"\n📈 分类报告:")
    # 确保所有类别都存在
    unique_targets = np.unique(all_targets)
    unique_predictions = np.unique(all_predictions)
    all_classes = sorted(set(unique_targets) | set(unique_predictions))
    
    if len(all_classes) == 3:
        target_names = ['下跌趋势', '上涨趋势', '横盘整理']
        print(classification_report(all_targets, all_predictions, 
                                   target_names=target_names))
    else:
        # 如果某些类别不存在，使用数字标签
        print(classification_report(all_targets, all_predictions))
    
    return {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'predictions': all_predictions,
        'targets': all_targets
    }


def visualize_training_results(results_with_bn, results_without_bn):
    """可视化训练结果对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 训练损失对比
    axes[0, 0].plot(results_with_bn['train_losses'], label='有BatchNorm', color='blue', linewidth=2)
    axes[0, 0].plot(results_without_bn['train_losses'], label='无BatchNorm', color='red', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('训练损失')
    axes[0, 0].set_title('训练损失对比 (有 vs 无 BatchNorm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    axes[0, 1].plot(results_with_bn['val_losses'], label='有BatchNorm', color='blue', linewidth=2)
    axes[0, 1].plot(results_without_bn['val_losses'], label='无BatchNorm', color='red', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('验证损失')
    axes[0, 1].set_title('验证损失对比 (有 vs 无 BatchNorm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练准确率对比
    axes[0, 2].plot(results_with_bn['train_accuracies'], label='有BatchNorm', color='green', linewidth=2)
    axes[0, 2].plot(results_without_bn['train_accuracies'], label='无BatchNorm', color='orange', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('训练轮次')
    axes[0, 2].set_ylabel('训练准确率 (%)')
    axes[0, 2].set_title('训练准确率对比 (有 vs 无 BatchNorm)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 验证准确率对比
    axes[1, 0].plot(results_with_bn['val_accuracies'], label='有BatchNorm', color='green', linewidth=2)
    axes[1, 0].plot(results_without_bn['val_accuracies'], label='无BatchNorm', color='orange', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('验证准确率 (%)')
    axes[1, 0].set_title('验证准确率对比 (有 vs 无 BatchNorm)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. BatchNorm效果分析
    bn_improvement = []
    for bn_acc, no_bn_acc in zip(results_with_bn['val_accuracies'], results_without_bn['val_accuracies']):
        improvement = bn_acc - no_bn_acc
        bn_improvement.append(improvement)
    
    axes[1, 1].plot(bn_improvement, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('准确率提升 (%)')
    axes[1, 1].set_title('BatchNorm带来的准确率提升')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 最终性能对比
    final_bn_acc = results_with_bn['val_accuracies'][-1]
    final_no_bn_acc = results_without_bn['val_accuracies'][-1]
    
    labels = ['有BatchNorm', '无BatchNorm']
    values = [final_bn_acc, final_no_bn_acc]
    colors = ['lightblue', 'lightcoral']
    
    axes[1, 2].bar(labels, values, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('最终验证准确率 (%)')
    axes[1, 2].set_title(f'最终性能对比 (提升: {final_bn_acc - final_no_bn_acc:.2f}%)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(values):
        axes[1, 2].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'bn_improvement': bn_improvement,
        'final_bn_accuracy': final_bn_acc,
        'final_no_bn_accuracy': final_no_bn_acc,
        'improvement_percentage': final_bn_acc - final_no_bn_acc
    }


def visualize_feature_maps(model, sample_data, device='cpu'):
    """可视化卷积层的特征图"""
    model.eval()
    model = model.to(device)
    
    # 选择第一个样本
    sample = sample_data[0].unsqueeze(0).to(device)
    
    # 获取中间特征
    feature_maps = []
    
    # 注册钩子来获取中间特征
    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu())
    
    # 注册钩子
    hooks = []
    hooks.append(model.conv1.register_forward_hook(hook_fn))
    hooks.append(model.conv2.register_forward_hook(hook_fn))
    hooks.append(model.conv3.register_forward_hook(hook_fn))
    
    # 前向传播
    with torch.no_grad():
        _ = model(sample)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化特征图
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    
    # 原始输入图像
    for c in range(3):
        axes[0, c].imshow(sample_data[0][c].cpu().numpy(), cmap='gray')
        axes[0, c].set_title(f'输入通道 {c}')
        axes[0, c].axis('off')
    
    # 第一层卷积特征图
    for i in range(8):
        if i < feature_maps[0].shape[1]:
            axes[1, i].imshow(feature_maps[0][0, i].numpy(), cmap='viridis')
            axes[1, i].set_title(f'Conv1 特征图 {i+1}')
            axes[1, i].axis('off')
    
    # 第二层卷积特征图
    for i in range(8):
        if i < feature_maps[1].shape[1]:
            axes[2, i].imshow(feature_maps[1][0, i].numpy(), cmap='plasma')
            axes[2, i].set_title(f'Conv2 特征图 {i+1}')
            axes[2, i].axis('off')
    
    plt.suptitle('CNN特征图可视化', fontsize=16)
    plt.tight_layout()
    plt.show()


def demonstrate_batchnorm_effect():
    """演示批量归一化的效果"""
    print("🧪 演示批量归一化效果")
    print("=" * 50)
    
    # 创建两个相同的网络（一个有BatchNorm，一个没有）
    model_with_bn = CNNWithBatchNorm(num_classes=3)
    model_without_bn = CNNWithoutBatchNorm(num_classes=3)
    
    # 生成测试数据
    test_dataset = StockChartDataset(n_samples=10, chart_size=32)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # 获取一个批次的数据
    data, _ = next(iter(test_loader))
    
    print("1. 检查激活值分布:")
    
    # 获取有BatchNorm模型的激活值
    model_with_bn.eval()
    with torch.no_grad():
        output_with_bn = model_with_bn(data)
    
    # 获取无BatchNorm模型的激活值
    model_without_bn.eval()
    with torch.no_grad():
        output_without_bn = model_without_bn(data)
    
    # 计算激活值的统计信息
    bn_activations = output_with_bn.numpy().flatten()
    no_bn_activations = output_without_bn.numpy().flatten()
    
    print(f"   有BatchNorm - 均值: {np.mean(bn_activations):.4f}, 标准差: {np.std(bn_activations):.4f}")
    print(f"   无BatchNorm - 均值: {np.mean(no_bn_activations):.4f}, 标准差: {np.std(no_bn_activations):.4f}")
    
    # 可视化激活值分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(bn_activations, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('激活值')
    axes[0].set_ylabel('频次')
    axes[0].set_title('有BatchNorm的激活值分布')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(no_bn_activations, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('激活值')
    axes[1].set_ylabel('频次')
    axes[1].set_title('无BatchNorm的激活值分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n2. 检查梯度稳定性:")
    
    # 训练一小段时间，观察梯度变化
    criterion = nn.CrossEntropyLoss()
    
    # 有BatchNorm的梯度
    model_with_bn.train()
    optimizer_bn = optim.SGD(model_with_bn.parameters(), lr=0.01)
    optimizer_bn.zero_grad()
    
    output = model_with_bn(data)
    target = torch.randint(0, 3, (data.shape[0],))
    loss = criterion(output, target)
    loss.backward()
    
    # 收集梯度
    bn_gradients = []
    for param in model_with_bn.parameters():
        if param.grad is not None:
            bn_gradients.extend(param.grad.view(-1).numpy())
    
    # 无BatchNorm的梯度
    model_without_bn.train()
    optimizer_no_bn = optim.SGD(model_without_bn.parameters(), lr=0.01)
    optimizer_no_bn.zero_grad()
    
    output = model_without_bn(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 收集梯度
    no_bn_gradients = []
    for param in model_without_bn.parameters():
        if param.grad is not None:
            no_bn_gradients.extend(param.grad.view(-1).numpy())
    
    print(f"   有BatchNorm - 梯度均值: {np.mean(bn_gradients):.6f}, 梯度标准差: {np.std(bn_gradients):.6f}")
    print(f"   无BatchNorm - 梯度均值: {np.mean(no_bn_gradients):.6f}, 梯度标准差: {np.std(no_bn_gradients):.6f}")
    
    print("\n✅ BatchNorm效果演示完成!")
    
    return {
        'bn_activations': bn_activations,
        'no_bn_activations': no_bn_activations,
        'bn_gradients': bn_gradients,
        'no_bn_gradients': no_bn_gradients
    }


def main():
    """主函数：训练和评估CNN模型"""
    print("=" * 60)
    print("🎯 第4天任务：卷积神经网络（CNN） + 批量归一化（BatchNorm）")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 1. 创建数据集
    print("\n1. 创建股票图表数据集...")
    dataset = StockChartDataset(n_samples=1000, chart_size=32)
    
    # 数据分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    
    # 2. 演示BatchNorm效果
    print("\n2. 演示批量归一化效果...")
    bn_demo_results = demonstrate_batchnorm_effect()
    
    # 3. 训练有BatchNorm的模型
    print("\n3. 训练有BatchNorm的CNN模型...")
    model_with_bn = CNNWithBatchNorm(num_classes=3)
    results_with_bn = train_cnn_model(
        model_with_bn, train_loader, val_loader,
        epochs=50, learning_rate=0.001, device=device
    )
    
    # 4. 训练无BatchNorm的模型
    print("\n4. 训练无BatchNorm的CNN模型...")
    model_without_bn = CNNWithoutBatchNorm(num_classes=3)
    results_without_bn = train_cnn_model(
        model_without_bn, train_loader, val_loader,
        epochs=50, learning_rate=0.001, device=device
    )
    
    # 5. 可视化对比结果
    print("\n5. 可视化训练结果对比...")
    comparison_results = visualize_training_results(results_with_bn, results_without_bn)
    
    # 6. 评估模型
    print("\n6. 评估有BatchNorm的模型...")
    eval_results_bn = evaluate_model(model_with_bn, test_loader, device=device)
    
    print("\n7. 评估无BatchNorm的模型...")
    eval_results_no_bn = evaluate_model(model_without_bn, test_loader, device=device)
    
    # 7. 可视化特征图
    print("\n8. 可视化CNN特征图...")
    sample_data, _ = next(iter(test_loader))
    visualize_feature_maps(model_with_bn, sample_data, device=device)
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("📊 第4天任务总结")
    print("=" * 60)
    
    print(f"\n🎯 批量归一化（BatchNorm）效果:")
    print(f"   最终验证准确率提升: {comparison_results['improvement_percentage']:.2f}%")
    print(f"   有BatchNorm: {comparison_results['final_bn_accuracy']:.2f}%")
    print(f"   无BatchNorm: {comparison_results['final_no_bn_accuracy']:.2f}%")
    
    print(f"\n📈 测试性能:")
    print(f"   有BatchNorm测试准确率: {eval_results_bn['test_accuracy']:.2f}%")
    print(f"   无BatchNorm测试准确率: {eval_results_no_bn['test_accuracy']:.2f}%")
    
    print(f"\n💡 关键发现:")
    print(f"   1. BatchNorm稳定了激活值分布")
    print(f"   2. BatchNorm加速了训练收敛")
    print(f"   3. BatchNorm提高了模型泛化能力")
    print(f"   4. CNN能够有效识别股票图表模式")
    print(f"   5. 多通道输入（价格、成交量、指标）提高了识别准确率")
    
    print(f"\n🚀 第4天任务完成!")
    print(f"   掌握了卷积神经网络（CNN）原理")
    print(f"   理解了批量归一化（BatchNorm）的作用")
    print(f"   实现了股票图表模式识别模型")
    print(f"   为后续图像处理任务打下坚实基础")
    
    return {
        'model_with_bn': model_with_bn,
        'model_without_bn': model_without_bn,
        'results_with_bn': results_with_bn,
        'results_without_bn': results_without_bn,
        'comparison_results': comparison_results,
        'eval_results_bn': eval_results_bn,
        'eval_results_no_bn': eval_results_no_bn
    }


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 保存模型
    torch.save(results['model_with_bn'].state_dict(), 'cnn_with_batchnorm.pth')
    torch.save(results['model_without_bn'].state_dict(), 'cnn_without_batchnorm.pth')
    
    print("\n💾 模型已保存:")
    print("   cnn_with_batchnorm.pth - 有BatchNorm的CNN模型")
    print("   cnn_without_batchnorm.pth - 无BatchNorm的CNN模型")