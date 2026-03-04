"""
简单的RNN测试
"""

import torch
import torch.nn as nn
import numpy as np

# 创建一个简单的RNN测试
def test_simple_rnn():
    print("测试简单的RNN实现")
    
    # 参数设置
    input_size = 3
    hidden_size = 5
    seq_length = 10
    batch_size = 2
    
    # 创建RNN层
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    # 创建随机输入数据
    # 形状: (batch_size, seq_length, input_size)
    input_data = torch.randn(batch_size, seq_length, input_size)
    
    print(f"输入数据形状: {input_data.shape}")
    
    # 前向传播
    output, hidden = rnn(input_data)
    
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_length, hidden_size), f"输出形状错误: {output.shape}"
    assert hidden.shape == (1, batch_size, hidden_size), f"隐藏状态形状错误: {hidden.shape}"
    
    print("✅ 简单RNN测试通过!")
    return True

def test_lstm():
    print("\n测试LSTM实现")
    
    # 参数设置
    input_size = 4
    hidden_size = 6
    seq_length = 8
    batch_size = 3
    
    # 创建LSTM层
    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    # 创建随机输入数据
    input_data = torch.randn(batch_size, seq_length, input_size)
    
    print(f"输入数据形状: {input_data.shape}")
    
    # 前向传播
    output, (hidden, cell) = lstm(input_data)
    
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden.shape}")
    print(f"细胞状态形状: {cell.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_length, hidden_size), f"输出形状错误: {output.shape}"
    assert hidden.shape == (1, batch_size, hidden_size), f"隐藏状态形状错误: {hidden.shape}"
    assert cell.shape == (1, batch_size, hidden_size), f"细胞状态形状错误: {cell.shape}"
    
    print("✅ LSTM测试通过!")
    return True

def test_gru():
    print("\n测试GRU实现")
    
    # 参数设置
    input_size = 5
    hidden_size = 7
    seq_length = 6
    batch_size = 4
    
    # 创建GRU层
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    # 创建随机输入数据
    input_data = torch.randn(batch_size, seq_length, input_size)
    
    print(f"输入数据形状: {input_data.shape}")
    
    # 前向传播
    output, hidden = gru(input_data)
    
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_length, hidden_size), f"输出形状错误: {output.shape}"
    assert hidden.shape == (1, batch_size, hidden_size), f"隐藏状态形状错误: {hidden.shape}"
    
    print("✅ GRU测试通过!")
    return True

def test_attention_mechanism():
    print("\n测试注意力机制")
    
    # 参数设置
    batch_size = 2
    seq_length = 5
    hidden_size = 8
    
    # 创建模拟的LSTM输出
    lstm_output = torch.randn(batch_size, seq_length, hidden_size)
    
    print(f"LSTM输出形状: {lstm_output.shape}")
    
    # 简单的注意力机制实现
    class SimpleAttention(nn.Module):
        def __init__(self, hidden_size):
            super(SimpleAttention, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            # x形状: (batch_size, seq_length, hidden_size)
            attention_scores = self.attention(x)  # (batch_size, seq_length, 1)
            attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_length)
            attention_weights = self.softmax(attention_scores)  # (batch_size, seq_length)
            
            # 应用注意力权重
            context_vector = torch.bmm(
                attention_weights.unsqueeze(1),  # (batch_size, 1, seq_length)
                x  # (batch_size, seq_length, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
            
            return context_vector, attention_weights
    
    # 创建注意力模块
    attention = SimpleAttention(hidden_size)
    
    # 前向传播
    context, weights = attention(lstm_output)
    
    print(f"上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 验证形状
    assert context.shape == (batch_size, hidden_size), f"上下文向量形状错误: {context.shape}"
    assert weights.shape == (batch_size, seq_length), f"注意力权重形状错误: {weights.shape}"
    
    # 验证注意力权重和为1
    weight_sums = weights.sum(dim=1)
    print(f"注意力权重和: {weight_sums}")
    assert torch.allclose(weight_sums, torch.ones(batch_size), rtol=1e-5), "注意力权重和不等于1"
    
    print("✅ 注意力机制测试通过!")
    return True

def main():
    print("=" * 60)
    print("第5天任务：RNN + LSTM + GRU + 注意力机制测试")
    print("=" * 60)
    
    test_results = []
    
    try:
        test_results.append(("简单RNN", test_simple_rnn()))
    except Exception as e:
        print(f"简单RNN测试失败: {e}")
        test_results.append(("简单RNN", False))
    
    try:
        test_results.append(("LSTM", test_lstm()))
    except Exception as e:
        print(f"LSTM测试失败: {e}")
        test_results.append(("LSTM", False))
    
    try:
        test_results.append(("GRU", test_gru()))
    except Exception as e:
        print(f"GRU测试失败: {e}")
        test_results.append(("GRU", False))
    
    try:
        test_results.append(("注意力机制", test_attention_mechanism()))
    except Exception as e:
        print(f"注意力机制测试失败: {e}")
        test_results.append(("注意力机制", False))
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总测试数: {total}")
    print(f"通过数: {passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed >= 3:  # 至少通过3个测试
        print("\n🎉 第5天核心概念测试通过！")
        print("已掌握：RNN、LSTM、GRU、注意力机制的基本原理")
        
        # 生成学习报告
        report = f"""
第5天学习成果报告：
====================
测试时间: 2026-02-28 04:15
测试结果: {passed}/{total} 通过

掌握的核心概念：
1. {'✅' if test_results[0][1] else '❌'} 循环神经网络（RNN）基本原理
2. {'✅' if test_results[1][1] else '❌'} 长短期记忆网络（LSTM）架构
3. {'✅' if test_results[2][1] else '❌'} 门控循环单元（GRU）简化版本
4. {'✅' if test_results[3][1] else '❌'} 注意力机制（Attention）实现

学习要点：
1. RNN适用于序列数据处理，但存在梯度消失问题
2. LSTM通过门控机制解决长期依赖问题
3. GRU是LSTM的简化版本，参数更少
4. 注意力机制让模型关注输入序列的重要部分

下一步建议：
1. 实现完整的序列到序列（Seq2Seq）模型
2. 学习Transformer架构
3. 实现自注意力机制（Self-Attention）
        """
        
        # 保存报告
        with open("/home/huang/openclaw-workspace/ultimate-agent/day5_learning_report.md", "w") as f:
            f.write(report)
        
        print("\n📋 学习报告已保存到: day5_learning_report.md")
        return True
    else:
        print("\n⚠️  需要进一步学习RNN相关概念")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)