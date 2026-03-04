"""
第5天任务：完整的股票预测系统
整合RNN、LSTM、GRU，实现多模型股票价格预测
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictionSystem:
    """股票预测系统"""
    
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.models = {}
        self.results = {}
        
    def load_or_generate_stock_data(self, num_days=1000):
        """加载或生成股票数据"""
        print("生成模拟股票数据...")
        
        # 基础趋势（长期上涨）
        trend = np.linspace(100, 200, num_days)
        
        # 季节性波动（季度效应）
        seasonal = 15 * np.sin(np.linspace(0, 12*np.pi, num_days))
        
        # 周期性波动（月度效应）
        monthly = 8 * np.sin(np.linspace(0, 24*np.pi, num_days))
        
        # 随机波动（市场噪声）
        random_walk = np.cumsum(np.random.normal(0, 3, num_days))
        
        # 合成价格序列
        price = trend + seasonal + monthly + random_walk
        
        # 确保价格为正
        price = np.abs(price)
        
        # 添加一些市场事件（大涨大跌）
        event_indices = np.random.choice(num_days, 10, replace=False)
        for idx in event_indices:
            if idx < num_days - 10:
                event_size = np.random.uniform(-20, 30)
                price[idx:idx+10] += event_size * np.exp(-np.arange(10)/3)
        
        # 创建DataFrame
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Price': price,
            'Volume': np.random.lognormal(10, 1, num_days)  # 模拟交易量
        })
        
        # 计算技术指标
        df['MA_7'] = df['Price'].rolling(window=7).mean()
        df['MA_30'] = df['Price'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Price'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Price'])
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"生成 {len(df)} 天股票数据")
        print(f"价格范围: {df['Price'].min():.2f} - {df['Price'].max():.2f}")
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """计算相对强弱指数RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def prepare_sequences(self, df, feature_columns=None):
        """准备序列数据"""
        if feature_columns is None:
            feature_columns = ['Price', 'Volume', 'MA_7', 'MA_30', 'RSI', 'MACD']
        
        # 选择特征
        features = df[feature_columns].values
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 创建序列
        X, y = [], []
        
        for i in range(len(features_scaled) - self.seq_length):
            X.append(features_scaled[i:i+self.seq_length])
            y.append(features_scaled[i+self.seq_length, 0])  # 预测价格
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"创建 {len(X)} 个序列，每个序列长度 {self.seq_length}")
        print(f"特征维度: {X.shape[2]}")
        
        return X, y
    
    def create_models(self, input_size, hidden_size=128):
        """创建多个预测模型"""
        
        # 1. 简单RNN模型
        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                rnn_out, _ = self.rnn(x)
                return self.fc(rnn_out[:, -1, :])
        
        # 2. LSTM模型
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                lstm_out, (hn, cn) = self.lstm(x)
                # 注意力机制
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attention_weights * lstm_out, dim=1)
                return self.fc(context), attention_weights
        
        # 3. GRU模型
        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                return self.fc(gru_out[:, -1, :])
        
        # 4. 双向LSTM模型
        class BiLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bilstm = nn.LSTM(input_size, hidden_size//2, batch_first=True, 
                                     num_layers=2, dropout=0.2, bidirectional=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                bilstm_out, _ = self.bilstm(x)
                return self.fc(bilstm_out[:, -1, :])
        
        self.models = {
            'RNN': SimpleRNNModel(),
            'LSTM': LSTMModel(),
            'GRU': GRUModel(),
            'BiLSTM': BiLSTMModel()
        }
        
        print(f"创建了 {len(self.models)} 个模型")
        for name, model in self.models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {params:,} 参数")
        
        return self.models
    
    def train_models(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """训练所有模型"""
        print("\n开始训练所有模型...")
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            
            train_losses = []
            test_losses = []
            
            for epoch in range(epochs):
                model.train()
                
                # 随机打乱数据
                indices = torch.randperm(len(X_train_tensor))
                epoch_loss = 0
                
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_train_tensor[batch_indices]
                    y_batch = y_train_tensor[batch_indices]
                    
                    # 前向传播
                    if name == 'LSTM':
                        outputs, _ = model(X_batch)
                    else:
                        outputs = model(X_batch)
                    
                    loss = criterion(outputs, y_batch)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / (len(X_train_tensor) // batch_size)
                train_losses.append(avg_train_loss)
                
                # 测试
                model.eval()
                with torch.no_grad():
                    if name == 'LSTM':
                        test_outputs, _ = model(X_test_tensor)
                    else:
                        test_outputs = model(X_test_tensor)
                    
                    test_loss = criterion(test_outputs, y_test_tensor)
                    test_losses.append(test_loss.item())
                
                scheduler.step(test_loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Test Loss: {test_loss.item():.6f}")
            
            # 保存训练结果
            self.results[name] = {
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1]
            }
            
            print(f"  {name}训练完成 - 最终测试损失: {test_losses[-1]:.6f}")
        
        return self.results
    
    def evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        print("\n评估模型性能...")
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_actual = self.scaler.inverse_transform(
            np.concatenate([y_test, np.zeros((len(y_test), X_test.shape[2]-1))], axis=1)
        )[:, 0]
        
        evaluation_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            model.eval()
            
            with torch.no_grad():
                if name == 'LSTM':
                    predictions, _ = model(X_test_tensor)
                else:
                    predictions = model(X_test_tensor)
                
                # 反标准化预测结果
                predictions_np = predictions.numpy()
                # 创建完整特征向量进行反标准化
                full_predictions = np.concatenate([
                    predictions_np, 
                    np.zeros((len(predictions_np), X_test.shape[2]-1))
                ], axis=1)
                predictions_actual = self.scaler.inverse_transform(full_predictions)[:, 0]
            
            # 计算性能指标
            mae = mean_absolute_error(y_test_actual, predictions_actual)
            mse = mean_squared_error(y_test_actual, predictions_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, predictions_actual)
            
            evaluation_results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'predictions': predictions_actual
            }
            
            print(f"\n{name} 性能指标:")
            print(f"  平均绝对误差 (MAE): {mae:.4f}")
            print(f"  均方误差 (MSE): {mse:.4f}")
            print(f"  均方根误差 (RMSE): {rmse:.4f}")
            print(f"  决定系数 (R²): {r2:.4f}")
        
        return evaluation_results
    
    def visualize_results(self, df, evaluation_results, last_n_days=100):
        """可视化结果"""
        print("\n生成可视化结果...")
        
        plt.figure(figsize=(20, 12))
        
        # 1. 原始股票价格
        plt.subplot(2, 3, 1)
        plt.plot(df['Price'], label='Stock Price', linewidth=1, alpha=0.7)
        plt.plot(df['MA_7'], label='7-Day MA', linewidth=1.5, alpha=0.8)
        plt.plot(df['MA_30'], label='30-Day MA', linewidth=1.5, alpha=0.8)
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title('Stock Price with Moving Averages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 技术指标
        plt.subplot(2, 3, 2)
        plt.plot(df['RSI'], label='RSI', linewidth=1.5, color='green')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.xlabel('Days')
        plt.ylabel('RSI')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(df['MACD'], label='MACD', linewidth=1.5, color='blue')
        plt.plot(df['Signal'], label='Signal Line', linewidth=1.5, color='red')
        plt.xlabel('Days')
        plt.ylabel('MACD')
        plt.title('MACD Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 模型损失对比
        plt.subplot(2, 3, 4)
        for name, result in self.results.items():
            plt.plot(result['test_losses'], label=name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')
        plt.title('Model Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 预测结果对比（最后N天）
        plt.subplot(2, 3, 5)
        actual_prices = df['Price'].values[-last_n_days:]
        time_steps = np.arange(len(actual_prices))
        
        plt.plot(time_steps, actual_prices, label='Actual Price', linewidth=2, color='black', alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange']
        for (name, result), color in zip(evaluation_results.items(), colors):
            if len(result['predictions']) >= last_n_days:
                pred_slice = result['predictitions'][-last_n_days:]
                plt.plot(time_steps, pred_slice, label=f'{name} Pred', 
                        linewidth=1.5, alpha=0.7, color=color)
        
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title(f'Model Predictions (Last {last_n_days} Days)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 性能指标雷达图
        plt.subplot(2, 3, 6, projection='polar')
        
        metrics = ['MAE', 'RMSE', 'R2']
        num_metrics = len(metrics)
        
        # 角度
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for name, result in evaluation_results.items():
            values = []
            for metric in metrics:
                if metric == 'R2':
                    # R2越高越好，所以取倒数用于雷达图（越小越好）
                    values.append(1.0 / (result[metric] + 0.1))  # 避免除零
                else:
                    values.append(result[metric])
            
            values += values[:1]  # 闭合图形
            
            plt.plot(angles, values, linewidth=2, label=name)
            plt.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], metrics)
        plt.title('Model Performance Radar Chart\n(Lower is better for all metrics)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.grid(True)
        
        plt.suptitle('Stock Prediction System - Complete Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('stock_prediction_complete_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 6. 创建性能对比表格
        print("\n" + "="*70)
        print("模型性能对比总结")
        print("="*70)
        
        performance_df = pd.DataFrame()
        for name, result in evaluation_results.items():
            performance_df[name] = pd.Series({
                'MAE': f"{result['MAE']:.4f}",
                'RMSE': f"{result['RMSE']:.4f}",
                'R²': f"{result['R2']:.4f}",
                'Final Test Loss': f"{self.results[name]['final_test_loss']:.6f}"
            })
        
        print(performance_df.T)
        
        # 找出最佳模型
        best_model = min(evaluation_results.items(), key=lambda x: x[1]['RMSE'])
        print(f"\n🏆 最佳模型: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})")
        
        return performance_df

def main():
    """主函数"""
    print("="*80)
    print("股票预测系统 - 第5天任务完整实现")
    print("="*80)
    
    # 创建预测系统
    system = StockPredictionSystem(seq_length=30)
    
    # 1. 生成/加载数据
    print("\n1. 数据准备阶段")
    print("-"*40)
    df = system.load_or_generate_stock_data(num_days=1500)
    
    # 2. 准备序列数据
    print("\n2. 序列数据准备")
    print("-"*40)
    X, y = system.prepare_sequences(df)
    
    # 分割训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 3. 创建模型
    print("\n3. 模型创建")
    print("-"*40)
    system.create_models(input_size=X.shape[2], hidden_size=128)
    
    # 4. 训练模型
    print("\n4. 模型训练")
    print("-"*40)
    system.train_models(X_train, y_train, X_test, y_test, epochs=80, batch_size=32)
    
    # 5. 评估模型
    print("\n5. 模型评估")
    print("-"*40)
    evaluation_results = system.evaluate_models(X_test, y_test)
    
    # 6. 可视化结果
    print("\n6. 结果可视化")
    print("-"*40)
    performance_df = system.visualize_results(df, evaluation_results, last_n_days=150)
    
    # 7. 保存模型和结果
    print("\n7. 保存结果")
    print("-"*40)
    
    # 保存最佳模型
    best_model_name = min(evaluation_results.items(), key=lambda x: x[1]['RMSE'])[0]
    best_model = system.results[best_model_name]['model']
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': system.scaler,
        'seq_length': system.seq_length,
        'performance': evaluation_results[best_model_name]
    }, 'best_stock_prediction_model.pth')
    
    print(f"✅ 最佳模型已保存: {best_model_name}")
    print(f"📁 模型文件: best_stock_prediction_model.pth")
    print(f"📊 最佳性能: RMSE = {evaluation_results[best_model_name]['RMSE']:.4f}")
    
    print("\n" + "="*80)
    print("股票预测系统完成！")
    print("="*80)
    print("\n🎯 第5天任务总结：")
    print("1. ✅ 实现了RNN、LSTM、GRU、BiLSTM四种循环神经网络")
    print("2. ✅ 构建了完整的股票预测系统")
    print("3. ✅ 集成了技术指标（MA、RSI、MACD）")
    print("4. ✅ 实现了多模型对比和性能评估")
    print("5. ✅ 生成了全面的可视化分析")
    print("6. ✅ 保存了最佳模型供后续使用")
    print("\n📈 下一步：")
    print("   - 使用真实股票数据测试")
    print("   - 集成更多技术指标")
    print("   - 实现交易策略回测")
    print("   - 部署为实时预测系统")

if __name__ == "__main__":
    main()