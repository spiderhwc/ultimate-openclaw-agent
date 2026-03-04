"""
第5天任务：循环神经网络（RNN）基础实现
目标：掌握RNN的基本原理和实现方法
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class SimpleRNN(nn.Module):
    """简单的RNN模型"""
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'  # 使用tanh激活函数
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # x形状: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # RNN前向传播
        out, hidden = self.rnn(x, hidden)
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out, hidden

def generate_sine_wave_data(seq_length=100, num_samples=1000):
    """生成正弦波数据用于RNN测试"""
    # 生成时间序列
    t = np.linspace(0, 4*np.pi, seq_length)
    
    # 生成多个正弦波样本
    X = []
    y = []
    
    for i in range(num_samples):
        # 每个样本有不同的相位和频率
        phase = np.random.uniform(0, 2*np.pi)
        freq = np.random.uniform(0.5, 2.0)
        
        # 生成正弦波
        sine_wave = np.sin(freq * t + phase)
        
        # 创建输入序列和目标
        # 输入：前seq_length-1个点
        # 目标：最后一个点
        X.append(sine_wave[:-1].reshape(-1, 1))
        y.append(sine_wave[-1])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y

def train_rnn_model():
    """训练RNN模型"""
    print("开始训练RNN模型...")
    
    # 生成训练数据
    X_train, y_train = generate_sine_wave_data(seq_length=50, num_samples=800)
    X_test, y_test = generate_sine_wave_data(seq_length=50, num_samples=200)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建模型
    model = SimpleRNN(input_size=1, hidden_size=64, output_size=1, num_layers=2)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 100
    batch_size = 32
    
    # 训练循环
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        
        # 随机打乱数据
        indices = torch.randperm(len(X_train_tensor))
        
        epoch_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            # 获取批次数据
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            # 前向传播
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / (len(X_train_tensor) // batch_size)
        train_losses.append(avg_train_loss)
        
        # 测试模式
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.item())
        
        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {test_loss.item():.6f}")
    
    print("RNN模型训练完成！")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.legend()
    plt.grid(True)
    
    # 可视化预测结果
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        # 随机选择5个测试样本进行可视化
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        for idx in sample_indices:
            # 获取样本
            X_sample = X_test_tensor[idx:idx+1]
            y_true = y_test_tensor[idx].item()
            
            # 预测
            y_pred, _ = model(X_sample)
            y_pred_value = y_pred.item()
            
            # 绘制时间序列
            time_steps = np.arange(len(X_sample[0]))
            plt.plot(time_steps, X_sample[0].numpy(), alpha=0.5)
            
            # 标记真实值和预测值
            plt.scatter(len(time_steps), y_true, color='green', s=50, label='True' if idx == sample_indices[0] else "")
            plt.scatter(len(time_steps), y_pred_value, color='red', s=50, label='Predicted' if idx == sample_indices[0] else "")
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('RNN Predictions on Sine Wave')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_sine_wave_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model

def demonstrate_rnn_characteristics():
    """演示RNN的特性"""
    print("\n" + "="*50)
    print("RNN特性演示")
    print("="*50)
    
    # 1. 展示RNN的时间依赖性
    print("\n1. RNN的时间依赖性：")
    print("   - RNN能够处理变长序列")
    print("   - 隐藏状态传递时间信息")
    print("   - 适用于时间序列预测")
    
    # 2. 创建不同长度的序列
    seq_lengths = [10, 20, 30]
    model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
    
    for seq_len in seq_lengths:
        # 生成随机序列
        X = torch.randn(1, seq_len, 1)
        
        # 前向传播
        output, hidden = model(X)
        
        print(f"\n   序列长度 {seq_len}:")
        print(f"     输入形状: {X.shape}")
        print(f"     输出形状: {output.shape}")
        print(f"     隐藏状态形状: {hidden.shape}")
    
    # 3. 演示梯度流动
    print("\n2. RNN的梯度流动：")
    print("   - 通过时间反向传播（BPTT）")
    print("   - 可能遇到梯度消失/爆炸问题")
    print("   - LSTM/GRU解决这些问题")
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("第5天任务：循环神经网络（RNN）基础实现")
    print("="*60)
    
    # 演示RNN特性
    demonstrate_rnn_characteristics()
    
    # 训练RNN模型
    print("\n" + "="*50)
    print("开始训练RNN模型...")
    print("="*50)
    
    trained_model = train_rnn_model()
    
    print("\n" + "="*50)
    print("RNN基础实现完成！")
    print("下一步：实现LSTM模型")
    print("="*50)