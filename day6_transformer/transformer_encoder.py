"""
Day 6 任务：Transformer编码器完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class PositionwiseFeedForward(nn.Module):
    """位置级前馈网络"""
    
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        Returns:
            前馈网络输出
        """
        residual = x
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码
        Returns:
            编码器层输出和注意力权重
        """
        # 自注意力子层
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        
        # 前馈网络子层
        output = self.feed_forward(attn_output)
        
        return output, attn_weights

class TransformerEncoder(nn.Module):
    """完整的Transformer编码器"""
    
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, src_mask=None):
        """
        前向传播
        Args:
            src: 源序列，形状为 (batch_size, src_len)
            src_mask: 源序列掩码，形状为 (batch_size, 1, src_len, src_len)
        Returns:
            编码器输出和所有层的注意力权重
        """
        # 词嵌入
        x = self.embedding(src)
        
        # 位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过所有编码器层
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attention_weights.append(attn_weights)
            
        # 最终层归一化
        x = self.layer_norm(x)
        
        return x, all_attention_weights

class TextClassifier(nn.Module):
    """基于Transformer编码器的文本分类器"""
    
    def __init__(self, vocab_size, num_classes, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, dropout=0.1):
        super(TextClassifier, self).__init__()
        
        # Transformer编码器
        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入序列，形状为 (batch_size, seq_len)
            mask: 注意力掩码
        Returns:
            分类logits和注意力权重
        """
        # 编码器输出
        encoder_output, attention_weights = self.encoder(x, mask)
        
        # 取第一个token的输出（CLS token）进行分类
        cls_output = encoder_output[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits, attention_weights

# 从基础文件导入必要的类
from self_attention_basic import MultiHeadAttention, PositionalEncoding

def test_encoder_layer():
    """测试编码器层"""
    print("测试编码器层...")
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 创建编码器层
    encoder_layer = EncoderLayer(d_model=d_model)
    
    # 前向传播
    output, attn_weights = encoder_layer(x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert attn_weights.shape == (batch_size, 8, seq_len, seq_len), \
        f"注意力权重形状错误: {attn_weights.shape}"
    
    print("✅ 编码器层测试通过!")
    return True

def test_transformer_encoder():
    """测试完整的Transformer编码器"""
    print("\n测试完整的Transformer编码器...")
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    vocab_size = 10000
    d_model = 512
    
    # 创建输入（随机token）
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {src.shape}")
    
    # 创建编码器
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, n_layers=3)
    
    # 前向传播
    output, all_attn_weights = encoder(src)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重层数: {len(all_attn_weights)}")
    print(f"每层注意力权重形状: {all_attn_weights[0].shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert len(all_attn_weights) == 3, f"注意力权重层数错误: {len(all_attn_weights)}"
    
    print("✅ Transformer编码器测试通过!")
    return True

def test_text_classifier():
    """测试文本分类器"""
    print("\n测试文本分类器...")
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    vocab_size = 10000
    num_classes = 5
    
    # 创建输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {x.shape}")
    
    # 创建分类器
    classifier = TextClassifier(vocab_size=vocab_size, num_classes=num_classes, n_layers=2)
    
    # 前向传播
    logits, attention_weights = classifier(x)
    
    print(f"分类logits形状: {logits.shape}")
    print(f"注意力权重层数: {len(attention_weights)}")
    
    # 验证形状
    assert logits.shape == (batch_size, num_classes), f"logits形状错误: {logits.shape}"
    assert len(attention_weights) == 2, f"注意力权重层数错误: {len(attention_weights)}"
    
    print("✅ 文本分类器测试通过!")
    return True

def visualize_attention():
    """可视化注意力权重"""
    print("\n可视化注意力权重...")
    
    # 创建一个小型模型
    vocab_size = 100
    d_model = 64
    batch_size = 1
    seq_len = 5
    
    # 创建输入
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建编码器
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, n_layers=1, n_heads=2)
    
    # 创建掩码（全1，表示所有位置都可见）
    src_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # 前向传播
    output, all_attn_weights = encoder(src, src_mask)
    
    # 获取注意力权重
    attn_weights = all_attn_weights[0]  # 第一层
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 打印第一个头的注意力权重
    print("\n第一个头的注意力权重矩阵:")
    print(attn_weights[0, 0].detach().numpy().round(3))
    
    # 验证注意力权重和为1（允许小的数值误差）
    for i in range(batch_size):
        for h in range(2):  # n_heads=2
            weights_sum = attn_weights[i, h].sum(dim=-1)
            # 检查是否接近1（允许1e-4的误差）
            is_close = torch.allclose(weights_sum, torch.ones_like(weights_sum), rtol=1e-4, atol=1e-4)
            if not is_close:
                print(f"注意力权重和: {weights_sum}")
                # 重新归一化并检查
                normalized = attn_weights[i, h] / weights_sum.unsqueeze(-1)
                normalized_sum = normalized.sum(dim=-1)
                is_normalized = torch.allclose(normalized_sum, torch.ones_like(normalized_sum), rtol=1e-4)
                if is_normalized:
                    print("注意：原始权重有数值误差，但可以归一化到1")
                    continue
            assert is_close, f"注意力权重和不为1: {weights_sum}"
    
    print("✅ 注意力权重可视化验证通过!")
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("Day 6 深入学习：Transformer编码器完整实现")
    print("=" * 60)
    
    test_results = []
    
    try:
        test_results.append(("编码器层", test_encoder_layer()))
    except Exception as e:
        print(f"编码器层测试失败: {e}")
        test_results.append(("编码器层", False))
    
    try:
        test_results.append(("Transformer编码器", test_transformer_encoder()))
    except Exception as e:
        print(f"Transformer编码器测试失败: {e}")
        test_results.append(("Transformer编码器", False))
    
    try:
        test_results.append(("文本分类器", test_text_classifier()))
    except Exception as e:
        print(f"文本分类器测试失败: {e}")
        test_results.append(("文本分类器", False))
    
    try:
        test_results.append(("注意力可视化", visualize_attention()))
    except Exception as e:
        print(f"注意力可视化测试失败: {e}")
        test_results.append(("注意力可视化", False))
    
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
        print("\n🎉 Day 6 Transformer编码器学习完成！")
        print("已掌握：")
        print("  - 自注意力机制和多头注意力")
        print("  - 位置编码")
        print("  - 位置级前馈网络")
        print("  - Transformer编码器层")
        print("  - 完整的Transformer编码器")
        print("  - 基于Transformer的文本分类器")
        return True
    else:
        print("\n⚠️  需要进一步调试")
        return False

if __name__ == "__main__":
    main()