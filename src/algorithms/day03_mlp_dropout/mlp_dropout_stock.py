#!/usr/bin/env python3
"""
多层感知机（MLP） + Dropout 股票预测模型
第3天任务：实现第一个神经网络模型

功能：
1. 实现多层感知机（MLP）神经网络
2. 集成Dropout正则化防止过拟合
3. 应用于股票价格预测任务
4. 可视化训练过程和结果

作者：悠悠（你的贴身小秘书）
日期：2026-02-25
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

class StockMLP(nn.Module):
    """
    多层感知机（MLP）股票预测模型
    包含Dropout正则化
    """
    def __init__(self, input_size=10, hidden_sizes=[64, 32, 16], output_size=1, dropout_rate=0.3):
        """
        初始化MLP模型
        
        参数：
        - input_size: 输入特征维度
        - hidden_sizes: 隐藏层大小列表
        - output_size: 输出维度
        - dropout_rate: Dropout概率
        """
        super(StockMLP, self).__init__()
        
        # 创建网络层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 添加中间隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)

def fetch_stock_data(ticker="AAPL", days=365):
    """
    获取股票数据
    
    参数：
    - ticker: 股票代码
    - days: 获取多少天的数据
    
    返回：
    - DataFrame: 股票数据
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📈 正在获取 {ticker} 股票数据...")
    print(f"   时间范围: {start_date.date()} 到 {end_date.date()}")
    
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock.empty:
        raise ValueError(f"无法获取 {ticker} 股票数据")
    
    print(f"✅ 成功获取 {len(stock)} 条数据")
    return stock

def prepare_features(stock_data, window_size=10):
    """
    准备特征数据
    
    参数：
    - stock_data: 股票数据
    - window_size: 滑动窗口大小
    
    返回：
    - X: 特征矩阵
    - y: 目标值（未来价格）
    """
    # 使用收盘价
    prices = stock_data['Close'].values
    
    # 创建技术指标
    df = pd.DataFrame({'Close': prices})
    
    # 简单移动平均
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # 相对强弱指数（RSI）
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # 价格变化率
    df['ROC'] = df['Close'].pct_change(periods=5) * 100
    
    # 成交量（如果有）
    if 'Volume' in stock_data.columns:
        df['Volume'] = stock_data['Volume'].values
        df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    
    # 创建滑动窗口特征
    features = []
    targets = []
    
    for i in range(window_size, len(df) - 1):
        # 获取窗口内的特征
        window_features = []
        
        # 添加技术指标
        for col in df.columns:
            if not pd.isna(df[col].iloc[i]):
                window_features.append(df[col].iloc[i])
        
        # 添加历史价格特征
        for j in range(1, 6):
            if i - j >= 0:
                window_features.append(prices[i - j])
        
        features.append(window_features)
        
        # 目标：未来1天的价格变化百分比
        future_return = (prices[i + 1] - prices[i]) / prices[i] * 100
        targets.append(future_return)
    
    # 转换为numpy数组
    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    # 处理NaN值
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    print(f"📊 特征矩阵形状: {X.shape}")
    print(f"🎯 目标值形状: {y.shape}")
    
    return X, y

def train_mlp_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
    """
    训练MLP模型
    
    参数：
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据
    - epochs: 训练轮数
    - batch_size: 批次大小
    - learning_rate: 学习率
    
    返回：
    - model: 训练好的模型
    - train_losses: 训练损失历史
    - val_losses: 验证损失历史
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = StockMLP(input_size=input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.3)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print("🚀 开始训练MLP模型...")
    print(f"   输入特征维度: {input_size}")
    print(f"   网络结构: {input_size} -> 64 -> 32 -> 16 -> 1")
    print(f"   Dropout率: 0.3")
    print(f"   训练轮数: {epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {learning_rate}")
    
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        # 批次训练
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)
            val_losses.append(val_loss.item())
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 每10轮打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"   轮次 {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss.item():.6f}")
    
    print("✅ 训练完成!")
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数：
    - model: 训练好的模型
    - X_test, y_test: 测试数据
    
    返回：
    - metrics: 性能指标字典
    """
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # 计算指标
    predictions_np = predictions.numpy().flatten()
    y_test_np = y_test.flatten()
    
    # 均方误差
    mse = np.mean((predictions_np - y_test_np) ** 2)
    
    # 平均绝对误差
    mae = np.mean(np.abs(predictions_np - y_test_np))
    
    # 方向准确率（预测涨跌方向）
    direction_correct = np.sum((predictions_np > 0) == (y_test_np > 0)) / len(y_test_np) * 100
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'Direction_Accuracy': direction_correct
    }
    
    return metrics, predictions_np

def visualize_results(train_losses, val_losses, y_true, y_pred, stock_name="AAPL"):
    """
    可视化训练结果
    
    参数：
    - train_losses: 训练损失历史
    - val_losses: 验证损失历史
    - y_true: 真实值
    - y_pred: 预测值
    - stock_name: 股票名称
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练损失曲线
    axes[0, 0].plot(train_losses, label='训练损失', color='blue', alpha=0.7)
    axes[0, 0].plot(val_losses, label='验证损失', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].set_title(f'{stock_name} - MLP训练损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 预测 vs 真实值
    axes[0, 1].scatter(y_true, y_pred, alpha=0.5, color='green')
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', label='完美预测线')
    axes[0, 1].set_xlabel('真实价格变化 (%)')
    axes[0, 1].set_ylabel('预测价格变化 (%)')
    axes[0, 1].set_title(f'{stock_name} - 预测 vs 真实值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 预测误差分布
    errors = y_pred - y_true
    axes[1, 0].hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='零误差线')
    axes[1, 0].set_xlabel('预测误差 (%)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title(f'{stock_name} - 预测误差分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 时间序列预测
    sample_size = min(50, len(y_true))
    indices = np.arange(sample_size)
    axes[1, 1].plot(indices, y_true[:sample_size], label='真实值', color='blue', marker='o')
    axes[1, 1].plot(indices, y_pred[:sample_size], label='预测值', color='orange', marker='s')
    axes[1, 1].set_xlabel('样本索引')
    axes[1, 1].set_ylabel('价格变化 (%)')
    axes[1, 1].set_title(f'{stock_name} - 时间序列预测对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/mlp_dropout_{stock_name}_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 第3天任务：多层感知机（MLP） + Dropout 股票预测模型")
    print("=" * 60)
    
    try:
        # 1. 获取数据
        stock_data = fetch_stock_data(ticker="AAPL", days=365)
        
        # 2. 准备特征
        X, y = prepare_features(stock_data, window_size=10)
        
        # 3. 数据分割
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"📊 数据分割结果:")
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   验证集: {X_val.shape[0]} 样本")
        print(f"   测试集: {X_test.shape[0]} 样本")
        
        # 4. 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # 5. 训练模型
        model, train_losses, val_losses = train_mlp_model(
            X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, learning_rate=0.001
        )
        
        # 6. 评估模型
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        print("\n📈 模型性能评估:")
        print(f"   均方误差 (MSE): {metrics['MSE']:.6f}")
        print(f"   平均绝对误差 (MAE): {metrics['MAE']:.6f}")
        print(f"   方向准确率: {metrics['Direction_Accuracy']:.2f}%")
        
        # 7. 可视化结果
        print("\n🎨 正在生成可视化图表...")
        visualize_results(train_losses, val_losses, y_test.flatten(), predictions, stock_name="AAPL")
        
        # 8. 保存模型
        torch.save(model.state_dict(), 'results/mlp_dropout_stock_model.pth')
        print("💾 模型已保存到: results/mlp_dropout_stock_model.pth")
        
        # 9. 模型架构总结
        print("\n🤖 模型架构总结:")
        print(model)
        
        # 10. Dropout效果分析
        print("\n🔍 Dropout效果分析:")
        print("   - Dropout率: 0.3 (30%的神经元在训练时被随机丢弃)")
        print("   - 作用: 防止过拟合，提高模型泛化能力")
        print("   - 训练时: Dropout激活，随机丢弃神经元")
        print("   - 测试时: Dropout关闭，使用所有神经元")
        
        return model, metrics
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()