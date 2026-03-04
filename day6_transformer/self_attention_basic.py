"""
Day 6 任务：自注意力机制基础实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Args:
            Q: 查询张量，形状为 (batch_size, n_heads, seq_len, d_k)
            K: 键张量，形状为 (batch_size, n_heads, seq_len, d_k)
            V: 值张量，形状为 (batch_size, n_heads, seq_len, d_v)
            mask: 掩码张量，形状为 (batch_size, 1, seq_len, seq_len)
        Returns:
            注意力输出和注意力权重
        """
        # 计算Q和K的点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # 注意力机制
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def split_heads(self, x, batch_size):
        """将输入分割成多个头"""
        # x形状: (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        # 转置为: (batch_size, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x, batch_size):
        """将多个头合并"""
        # x形状: (batch_size, n_heads, seq_len, d_k)
        x = x.transpose(1, 2).contiguous()
        # 合并为: (batch_size, seq_len, d_model)
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 残差连接
        residual = Q
        
        # 线性变换并分割成多个头
        Q = self.split_heads(self.W_Q(Q), batch_size)
        K = self.split_heads(self.W_K(K), batch_size)
        V = self.split_heads(self.W_V(V), batch_size)
        
        # 计算注意力
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = self.combine_heads(attention_output, batch_size)
        
        # 输出线性变换
        output = self.W_O(attention_output)
        output = self.dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

def test_self_attention():
    """测试自注意力机制"""
    print("测试自注意力机制...")
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建输入
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # 前向传播
    output, attention_weights = mha(Q, K, V)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len), \
        f"注意力权重形状错误: {attention_weights.shape}"
    
    print("✅ 自注意力机制测试通过!")
    return True

def test_positional_encoding():
    """测试位置编码"""
    print("\n测试位置编码...")
    
    # 参数设置
    batch_size = 2
    seq_len = 20
    d_model = 512
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 创建位置编码模块
    pe = PositionalEncoding(d_model=d_model)
    
    # 前向传播
    output = pe(x)
    
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    
    # 验证位置编码被添加
    assert not torch.allclose(x, output), "位置编码应该改变输入值"
    
    print("✅ 位置编码测试通过!")
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("Day 6 基础测试：自注意力机制 + 位置编码")
    print("=" * 60)
    
    test_results = []
    
    try:
        test_results.append(("自注意力机制", test_self_attention()))
    except Exception as e:
        print(f"自注意力机制测试失败: {e}")
        test_results.append(("自注意力机制", False))
    
    try:
        test_results.append(("位置编码", test_positional_encoding()))
    except Exception as e:
        print(f"位置编码测试失败: {e}")
        test_results.append(("位置编码", False))
    
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
    
    if passed == total:
        print("\n🎉 Day 6 基础测试全部通过！")
        print("已为明天的Transformer学习做好准备")
        return True
    else:
        print("\n⚠️  需要进一步调试")
        return False

if __name__ == "__main__":
    main()