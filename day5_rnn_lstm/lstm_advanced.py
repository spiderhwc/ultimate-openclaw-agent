"""
第5天任务：LSTM（长短期记忆网络）高级实现
目标：掌握LSTM的原理、优势和应用
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

class AdvancedLSTM(nn.Module):
    """高级LSTM模型"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout=0.2):
        super(AdvancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 单向LSTM
        )
        
        # 注意力机制（可选）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x, hidden=None):
        # x形状: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态和细胞状态
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (h0, c0)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, hidden)
        
        # 应用注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 全连接层输出
        output = self.fc(context_vector)
        
        return output, (hn, cn), attention_weights

class GRUModel(nn.Module):
    """GRU（门控循环单元）模型对比"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)
        output = self.fc(gru_out[:, -1, :])
        return output, hidden

def generate_complex_time_series(seq_length=100, num_samples=1000):
    """生成复杂时间序列数据"""
    X = []
    y = []
    
    for i in range(num_samples):
        # 生成基础信号
        t = np.linspace(0, 4*np.pi, seq_length)
        
        # 混合多个频率和相位的正弦波
        signal = np.zeros(seq_length)
        for _ in range(3):
            freq = np.random.uniform(0.5, 3.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.3, 1.0)
            signal += amplitude * np.sin(freq * t + phase)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, seq_length)
        signal += noise
        
        # 添加趋势
        trend = np.linspace(0, np.random.uniform(-1, 1), seq_length)
        signal += trend
        
        # 创建输入和目标
        X.append(signal[:-1].reshape(-1, 1))
        y.append(signal[-1])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y

def compare_rnn_lstm_gru():
    """比较RNN、LSTM和GRU的性能"""
    print("开始比较RNN、LSTM和GRU性能...")
    
    # 生成测试数据
    X, y = generate_complex_time_series(seq_length=50, num_samples=500)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 创建三个模型
    models = {
        'RNN': nn.RNN(input_size=1, hidden_size=32, batch_first=True),
        'LSTM': nn.LSTM(input_size=1, hidden_size=32, batch_first=True),
        'GRU': nn.GRU(input_size=1, hidden_size=32, batch_first=True)
    }
    
    # 添加输出层
    for name in models:
        if name == 'RNN':
            models[name] = nn.Sequential(
                models[name],
                nn.Linear(32, 1)
            )
        else:
            # 对于LSTM和GRU，需要自定义forward
            class WrappedModel(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    self.fc = nn.Linear(32, 1)
                
                def forward(self, x):
                    if isinstance(self.base_model, nn.LSTM):
                        out, (hn, cn) = self.base_model(x)
                    else:
                        out, hn = self.base_model(x)
                    return self.fc(out[:, -1, :])
            
            models[name] = WrappedModel(models[name])
    
    # 训练参数
    criterion = nn.MSELoss()
    learning_rate = 0.001
    num_epochs = 50
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练{name}模型...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 前向传播
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        training_time = time.time() - start_time
        
        results[name] = {
            'final_loss': losses[-1],
            'training_time': training_time,
            'losses': losses
        }
        
        print(f"  {name}训练完成 - 最终损失: {losses[-1]:.6f}, 训练时间: {training_time:.2f}秒")
    
    # 可视化比较结果
    plt.figure(figsize=(15, 5))
    
    # 1. 训练损失对比
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['losses'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失对比 (RNN vs LSTM vs GRU)')
    plt.legend()
    plt.grid(True)
    
    # 2. 最终损失对比
    plt.subplot(1, 3, 2)
    names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in names]
    bars = plt.bar(names, final_losses, color=['blue', 'green', 'red'])
    plt.ylabel('Final Loss')
    plt.title('最终损失对比')
    plt.grid(True, axis='y')
    
    # 在柱状图上添加数值
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{loss:.6f}', ha='center', va='bottom')
    
    # 3. 训练时间对比
    plt.subplot(1, 3, 3)
    training_times = [results[name]['training_time'] for name in names]
    bars = plt.bar(names, training_times, color=['blue', 'green', 'red'])
    plt.ylabel('Training Time (seconds)')
    plt.title('训练时间对比')
    plt.grid(True, axis='y')
    
    # 在柱状图上添加数值
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rnn_lstm_gru_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def demonstrate_lstm_gates():
    """演示LSTM的门控机制"""
    print("\n" + "="*60)
    print("LSTM门控机制演示")
    print("="*60)
    
    # LSTM的四个门：
    # 1. 遗忘门 (Forget Gate) - 决定丢弃哪些信息
    # 2. 输入门 (Input Gate) - 决定更新哪些信息
    # 3. 候选记忆门 (Candidate Gate) - 创建新的候选记忆
    # 4. 输出门 (Output Gate) - 决定输出哪些信息
    
    print("\nLSTM的四个关键门控机制：")
    print("1. 遗忘门 (Forget Gate): f_t = σ(W_f · [h_{t-1}, x_t] + b_f)")
    print("   作用：决定从细胞状态中丢弃哪些信息")
    print("   输出范围：[0, 1]，0表示完全遗忘，1表示完全保留")
    
    print("\n2. 输入门 (Input Gate): i_t = σ(W_i · [h_{t-1}, x_t] + b_i)")
    print("   作用：决定更新哪些新信息到细胞状态")
    
    print("\n3. 候选记忆门 (Candidate Gate): g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)")
    print("   作用：创建新的候选记忆值")
    
    print("\n4. 输出门 (Output Gate): o_t = σ(W_o · [h_{t-1}, x_t] + b_o)")
    print("   作用：基于细胞状态决定输出什么")
    
    print("\n细胞状态更新公式：")
    print("   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t")
    print("   h_t = o_t ⊙ tanh(c_t)")
    
    print("\n其中：")
    print("   ⊙ 表示逐元素相乘")
    print("   σ 表示sigmoid激活函数")
    print("   tanh 表示双曲正切激活函数")
    
    # 可视化门控机制
    plt.figure(figsize=(12, 8))
    
    # 模拟门控信号
    time_steps = np.arange(50)
    
    # 模拟四个门的信号
    np.random.seed(42)
    forget_gate = 0.5 + 0.3 * np.sin(time_steps * 0.2)
    input_gate = 0.5 + 0.3 * np.cos(time_steps * 0.15)
    candidate_gate = np.tanh(0.5 * np.sin(time_steps * 0.1))
    output_gate = 0.5 + 0.2 * np.sin(time_steps * 0.25)
    
    # 绘制门控信号
    gates = [forget_gate, input_gate, candidate_gate, output_gate]
    gate_names = ['Forget Gate', 'Input Gate', 'Candidate Gate', 'Output Gate']
    colors = ['red', 'green', 'blue', 'purple']
    
    for i, (gate, name, color) in enumerate(zip(gates, gate_names, colors)):
        plt.subplot(2, 2, i+1)
        plt.plot(time_steps, gate, color=color, linewidth=2)
        plt.fill_between(time_steps, 0, gate, alpha=0.3, color=color)
        plt.title(name)
        plt.xlabel('Time Step')
        plt.ylabel('Gate Value')
        plt.ylim(0, 1.1)
        plt.grid(True)
    
    plt.suptitle('LSTM Gate Mechanisms Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig('lstm_gates_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("LSTM优势总结：")
    print("="*60)
    print("1. 解决梯度消失问题：通过细胞状态长期保存信息")
    print("2. 选择性记忆：遗忘门控制信息保留")
    print("3. 长期依赖学习：适合长序列任务")
    print("4. 稳定训练：相比普通RNN更稳定")

def train_lstm_for_stock_prediction():
    """训练LSTM进行股票预测"""
    print("\n" + "="*60)
    print("LSTM股票价格预测演示")
    print("="*60)
    
    # 生成模拟股票数据
    def generate_stock_like_data(num_days=500):
        """生成类似股票价格的时间序列"""
        # 基础趋势
        trend = np.linspace(100, 150, num_days)
        
        # 周期性波动（季度效应）
        seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, num_days))
        
        # 随机波动
        random_walk = np.cumsum(np.random.normal(0, 2, num_days))
        
        # 合成价格
        price = trend + seasonal + random_walk
        
        # 确保价格为正
        price = np.abs(price)
        
        return price
    
    # 生成数据
    stock_prices = generate_stock_like_data(500)
    
    # 准备序列数据
    seq_length = 30  # 使用前30天预测第31天
    X = []
    y = []
    
    for i in range(len(stock_prices) - seq_length):
        X.append(stock_prices[i:i+seq_length])
        y.append(stock_prices[i+seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    # 数据标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, seq_length, 1)
    y_scaled = scaler.transform(y)
    
    # 分割训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train = torch.FloatTensor(X_scaled[:split_idx])
    y_train = torch.FloatTensor(y_scaled[:split_idx])
    X_test = torch.FloatTensor(X_scaled[split_idx:])
    y_test = torch.FloatTensor(y_scaled[split_idx:])
    
    # 创建LSTM模型
    model = AdvancedLSTM(
        input_size=1,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        dropout=0.1
    )
    
    # 训练参数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    batch_size = 32
    
    # 训练循环
    train_losses = []
    test_losses = []
    
    print(f"\n训练数据: {len(X_train)} 个序列")
    print(f"测试数据: {len(X_test)} 个序列")
    print(f"开始训练LSTM股票预测模型...")
    
    for epoch in range(num_epochs):
        model.train()
        
        # 随机打乱训练数据
        indices = torch.randperm(len(X_train))
        epoch_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 前向传播
            outputs, _, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (len(X_train) // batch_size)
        train_losses.append(avg_train_loss)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs, _, _ = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {test_loss.item():.6f}")
    
    print("\nLSTM股票预测模型训练完成！")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 1. 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM股票预测训练损失')
    plt.legend()
    plt.grid(True)
    
    # 2. 预测结果
    plt.subplot(2, 2, 2)
    model.eval()
    with torch.no_grad():
        # 在测试集上进行预测
        predictions, _, attention_weights = model(X_test)
        
        # 反标准化
        predictions_actual = scaler.inverse_transform(predictions.numpy())
        y_test_actual = scaler.inverse_transform(y_test.numpy())
        
        # 绘制最后100个预测点
        plt.plot(y_test_actual[-100:], label='Actual Prices', linewidth=2, alpha=0.7)
        plt.plot(predictions_actual[-100:], label='Predicted Prices', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Stock Price')
        plt.title('LSTM股票价格预测结果')
        plt.legend()
        plt.grid(True)
    
    # 3. 注意力权重可视化
    plt.subplot(2, 2, 3)
    # 选择一个样本展示注意力权重
    sample_idx = 0
    attention_sample = attention_weights[sample_idx].squeeze().numpy()
    
    plt.imshow(attention_sample.reshape(1, -1), aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Time Step')
    plt.title('LSTM注意力权重分布')
    plt.yticks([])
    
    # 4. 预测误差分布
    plt.subplot(2, 2, 4)
    errors = predictions_actual - y_test_actual
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('预测误差分布')
    plt.grid(True, axis='y')
    
    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f}')
    plt.axvline(mean_error + std_error, color='orange', linestyle=':', label=f'±1 STD')
    plt.axvline(mean_error - std_error, color='orange', linestyle=':')
    plt.legend()
    
    plt.suptitle('LSTM股票价格预测系统', fontsize=16)
    plt.tight_layout()
    plt.savefig('lstm_stock_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 计算性能指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    mse = mean_squared_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions_actual)
    
    print("\n" + "="*60)
    print("LSTM股票预测性能指标：")
    print("="*60)
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    return model, scaler

if __name__ == "__main__":
    print("="*70)
    print("第5天任务：LSTM（长短期记忆网络）高级实现")
    print("="*70)
    
    # 演示LSTM门控机制
    demonstrate_lstm_gates()
    
    # 比较RNN、LSTM和GRU
    print("\n" + "="*60)
    print("开始比较RNN、LSTM和GRU性能...")
    print("="*60)
    
    comparison_results = compare_rnn_lstm_gru()
    
    # 训练LSTM股票预测模型
    print("\n" + "="*60)
    print("开始训练LSTM股票预测模型...")
    print("="*60)
    
    lstm_model, price_scaler = train_lstm_for_stock_prediction()
    
    print("\n" + "="*70)
    print("LSTM高级实现完成！")
    print("关键收获：")
    print("1. LSTM通过门控机制解决梯度消失问题")
    print("2. LSTM适合长序列和时间依赖任务")
    print("3. 在股票预测等任务上表现优于普通RNN")
    print("4. 注意力机制可提高模型解释性")
    print("="*70)