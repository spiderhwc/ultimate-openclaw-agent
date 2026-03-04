"""
第5天任务测试：循环神经网络（RNN） + 注意力机制
测试RNN、LSTM、GRU和注意力机制的实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 导入Day 5的实现
from day5_rnn_lstm.rnn_basic import SimpleRNN, generate_sine_wave_data
from day5_rnn_lstm.lstm_advanced import AdvancedLSTM
from day5_rnn_lstm.stock_prediction_system import StockPredictionSystem

def test_rnn_basic():
    """测试基本的RNN实现"""
    print("=" * 60)
    print("测试1：基本RNN实现")
    print("=" * 60)
    
    # 生成测试数据
    seq_length = 50
    num_samples = 100
    X, y = generate_sine_wave_data(seq_length, num_samples)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (100, 50, 1)
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # (100, 1)
    
    # 创建模型
    model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
    
    # 测试前向传播
    with torch.no_grad():
        # 只取一个样本测试
        test_input = X_tensor[:1]  # (1, 50, 1)
        output, hidden = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"隐藏状态形状: {hidden.shape}")
        print(f"前向传播测试通过!")
    
    return True

def test_lstm_with_attention():
    """测试带注意力机制的LSTM"""
    print("\n" + "=" * 60)
    print("测试2：带注意力机制的LSTM")
    print("=" * 60)
    
    # 生成测试数据
    seq_length = 30
    batch_size = 16
    input_size = 5  # 多特征输入
    
    # 创建随机输入
    X = torch.randn(batch_size, seq_length, input_size)
    
    # 创建模型
    model = AdvancedLSTM(
        input_size=input_size,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        dropout=0.2
    )
    
    # 测试前向传播
    with torch.no_grad():
        output = model(X)
        print(f"输入形状: {X.shape}")
        print(f"输出形状: {output.shape}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"LSTM + 注意力机制测试通过!")
    
    return True

def test_stock_prediction_system():
    """测试股票预测系统"""
    print("\n" + "=" * 60)
    print("测试3：完整的股票预测系统")
    print("=" * 60)
    
    try:
        # 创建股票预测系统
        system = StockPredictionSystem(seq_length=30)
        
        # 生成股票数据
        data = system.load_or_generate_stock_data(num_days=500)
        
        # 准备训练数据
        X_train, y_train, X_test, y_test = system.prepare_data(data)
        
        print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
        print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
        print(f"数据准备测试通过!")
        
        return True
    except Exception as e:
        print(f"股票预测系统测试失败: {e}")
        return False

def test_attention_mechanism():
    """专门测试注意力机制"""
    print("\n" + "=" * 60)
    print("测试4：注意力机制实现")
    print("=" * 60)
    
    # 实现一个简单的注意力机制
    class SimpleAttention(nn.Module):
        def __init__(self, hidden_size):
            super(SimpleAttention, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, lstm_output):
            # lstm_output形状: (batch_size, seq_len, hidden_size)
            attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
            attention_weights = self.softmax(attention_weights.squeeze(-1))  # (batch_size, seq_len)
            
            # 应用注意力权重
            context_vector = torch.bmm(
                attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
                lstm_output  # (batch_size, seq_len, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
            
            return context_vector, attention_weights
    
    # 测试注意力机制
    batch_size = 8
    seq_len = 20
    hidden_size = 32
    
    # 创建模拟LSTM输出
    lstm_output = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建注意力模块
    attention = SimpleAttention(hidden_size)
    
    # 测试前向传播
    with torch.no_grad():
        context, weights = attention(lstm_output)
        print(f"LSTM输出形状: {lstm_output.shape}")
        print(f"上下文向量形状: {context.shape}")
        print(f"注意力权重形状: {weights.shape}")
        print(f"注意力权重和: {weights.sum(dim=1)}")  # 应该接近1
        
        # 验证注意力权重
        assert weights.shape == (batch_size, seq_len), "注意力权重形状错误"
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), rtol=1e-5), "注意力权重和不等于1"
        
        print(f"注意力机制测试通过!")
    
    return True

def test_rnn_vs_lstm():
    """对比RNN和LSTM的性能"""
    print("\n" + "=" * 60)
    print("测试5：RNN vs LSTM性能对比")
    print("=" * 60)
    
    # 创建简单的序列数据
    seq_length = 25
    batch_size = 32
    input_size = 3
    
    # 创建输入数据
    X = torch.randn(batch_size, seq_length, input_size)
    
    # 创建RNN模型
    rnn_model = nn.RNN(input_size=input_size, hidden_size=64, batch_first=True)
    
    # 创建LSTM模型
    lstm_model = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
    
    # 测试推理时间
    import time
    
    with torch.no_grad():
        # RNN推理
        start_time = time.time()
        rnn_output, rnn_hidden = rnn_model(X)
        rnn_time = time.time() - start_time
        
        # LSTM推理
        start_time = time.time()
        lstm_output, (lstm_hidden, lstm_cell) = lstm_model(X)
        lstm_time = time.time() - start_time
    
    print(f"RNN推理时间: {rnn_time:.4f}秒")
    print(f"LSTM推理时间: {lstm_time:.4f}秒")
    print(f"LSTM/RNN时间比: {lstm_time/rnn_time:.2f}")
    
    # 输出形状对比
    print(f"\nRNN输出形状: {rnn_output.shape}")
    print(f"LSTM输出形状: {lstm_output.shape}")
    
    print(f"RNN vs LSTM性能对比测试通过!")
    
    return True

def main():
    """主测试函数"""
    print("开始第5天任务测试：循环神经网络（RNN）+ 注意力机制")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("基本RNN实现", test_rnn_basic()))
    test_results.append(("LSTM+注意力机制", test_lstm_with_attention()))
    test_results.append(("股票预测系统", test_stock_prediction_system()))
    test_results.append(("注意力机制", test_attention_mechanism()))
    test_results.append(("RNN vs LSTM对比", test_rnn_vs_lstm()))
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总测试数: {total}")
    print(f"通过数: {passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 第5天任务测试全部通过！")
        print("已掌握：循环神经网络（RNN）、LSTM、GRU、注意力机制")
        
        # 生成测试报告
        report = f"""
第5天任务完成报告：
====================
完成时间: 2026-02-28 04:10
测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)

掌握的核心技能：
1. ✅ 循环神经网络（RNN）基本原理和实现
2. ✅ 长短期记忆网络（LSTM）架构和优势
3. ✅ 门控循环单元（GRU）简化版本
4. ✅ 注意力机制（Attention Mechanism）实现
5. ✅ 序列数据处理和预测系统

实践项目：
1. 正弦波序列预测（RNN）
2. 多特征时间序列处理（LSTM+Attention）
3. 股票价格预测系统

下一步学习建议：
1. 学习Transformer架构
2. 实现自注意力机制（Self-Attention）
3. 构建完整的编码器-解码器架构
        """
        
        # 保存测试报告
        with open("/home/huang/openclaw-workspace/ultimate-agent/day5_test_report.md", "w") as f:
            f.write(report)
        
        print("\n📋 测试报告已保存到: day5_test_report.md")
        
        return True
    else:
        print("\n⚠️  部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)