"""
Day 7 任务：Transformer解码器实现
基于Day 6的编码器实现，添加解码器部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# 首先导入Day 6中已经实现的基础组件
# 假设这些组件已经存在，这里重新定义以确保完整性

class MultiHeadAttention(nn.Module):
    """多头注意力机制（来自Day 6，添加掩码支持）"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        Args:
            query: 查询张量，形状为 (batch_size, seq_len_q, d_model)
            key: 键张量，形状为 (batch_size, seq_len_k, d_model)
            value: 值张量，形状为 (batch_size, seq_len_v, d_model)
            mask: 注意力掩码，形状为 (batch_size, seq_len_q, seq_len_k)
        Returns:
            注意力输出和注意力权重
        """
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 调整掩码形状以匹配注意力分数
            # 原始掩码形状可能是 (batch_size, seq_len_q, seq_len_k)
            # 需要调整为 (batch_size, 1, seq_len_q, seq_len_k) 以匹配多头
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 添加头维度
            elif mask.dim() == 4:
                # 已经是正确形状
                pass
            else:
                raise ValueError(f"掩码维度不正确: {mask.dim()}")
            
            # 掩码值为False的位置需要被屏蔽（设置为负无穷）
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值上
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出线性变换
        output = self.w_o(context)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """位置级前馈网络（来自Day 6）"""
    
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
    """Transformer编码器层（来自Day 6）"""
    
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
            编码器层输出
        """
        # 自注意力子层
        residual = x
        x, _ = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络子层
        x = self.feed_forward(x)
        
        return x

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 掩码自注意力（第一个注意力层）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 编码器-解码器注意力（第二个注意力层）
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        Args:
            x: 解码器输入，形状为 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出，形状为 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码，形状为 (batch_size, 1, src_seq_len)
            tgt_mask: 目标序列掩码，形状为 (batch_size, tgt_seq_len, tgt_seq_len)
        Returns:
            解码器层输出
        """
        # 1. 掩码自注意力子层（自回归特性）
        residual = x
        x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.layer_norm1(residual + x)
        
        # 2. 编码器-解码器注意力子层
        residual = x
        x, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x = self.layer_norm2(residual + x)
        
        # 3. 前馈网络子层
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm3(residual + x)
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码（来自Day 6）"""
    
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # 形状: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码
    Args:
        seq: 输入序列，形状为 (batch_size, seq_len)
        pad_idx: 填充token的索引
    Returns:
        填充掩码，形状为 (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_lookahead_mask(size):
    """
    创建前瞻掩码（用于解码器的自回归特性）
    Args:
        size: 序列长度
    Returns:
        前瞻掩码，形状为 (size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask

def create_decoder_mask(tgt, pad_idx=0):
    """
    创建解码器掩码（结合填充掩码和前瞻掩码）
    Args:
        tgt: 目标序列，形状为 (batch_size, tgt_seq_len)
        pad_idx: 填充token的索引
    Returns:
        解码器掩码，形状为 (batch_size, tgt_seq_len, tgt_seq_len)
    """
    batch_size, seq_len = tgt.shape
    
    # 创建填充掩码 (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(tgt, pad_idx)
    
    # 创建前瞻掩码 (seq_len, seq_len)
    lookahead_mask = create_lookahead_mask(seq_len)
    
    # 扩展前瞻掩码到批次维度 (batch_size, 1, seq_len, seq_len)
    lookahead_mask = lookahead_mask.unsqueeze(0).unsqueeze(0)
    lookahead_mask = lookahead_mask.expand(batch_size, 1, seq_len, seq_len)
    
    # 结合两种掩码
    # 填充掩码为True的位置是有效位置，前瞻掩码为0的位置是有效位置
    # 我们需要的位置是两者都有效的位置
    combined_mask = padding_mask & (lookahead_mask == 0)
    
    # 调整形状为 (batch_size, seq_len, seq_len)
    combined_mask = combined_mask.squeeze(1).squeeze(1)
    
    # 转换为布尔值
    combined_mask = combined_mask.bool()
    
    return combined_mask

def create_decoder_mask_for_inference(seq_len, device='cpu'):
    """
    为推理创建解码器掩码（只有前瞻掩码，没有填充掩码）
    Args:
        seq_len: 序列长度
        device: 设备
    Returns:
        解码器掩码，形状为 (1, seq_len, seq_len)
    """
    # 只创建前瞻掩码
    lookahead_mask = create_lookahead_mask(seq_len)
    
    # 转换为布尔掩码（True表示有效位置）
    mask = (lookahead_mask == 0).bool()
    
    # 添加批次维度
    mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    return mask

# 测试代码
if __name__ == "__main__":
    print("测试Transformer解码器组件...")
    
    # 测试掩码创建函数
    print("\n1. 测试掩码创建函数:")
    
    # 测试填充掩码
    test_seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    padding_mask = create_padding_mask(test_seq, pad_idx=0)
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"填充掩码值:\n{padding_mask}")
    
    # 测试前瞻掩码
    lookahead_mask = create_lookahead_mask(5)
    print(f"\n前瞻掩码形状: {lookahead_mask.shape}")
    print(f"前瞻掩码值:\n{lookahead_mask}")
    
    # 测试解码器掩码
    decoder_mask = create_decoder_mask(test_seq, pad_idx=0)
    print(f"\n解码器掩码形状: {decoder_mask.shape}")
    print(f"解码器掩码值（第一个样本）:\n{decoder_mask[0]}")
    
    # 测试解码器层
    print("\n2. 测试解码器层:")
    batch_size = 2
    tgt_seq_len = 5
    src_seq_len = 7
    d_model = 512
    
    # 创建模拟数据
    decoder_input = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    # 创建掩码
    src_mask = create_padding_mask(torch.randint(1, 10, (batch_size, src_seq_len)))
    tgt_mask = create_decoder_mask(torch.randint(1, 10, (batch_size, tgt_seq_len)))
    
    # 创建解码器层
    decoder_layer = DecoderLayer(d_model=d_model, n_heads=8, d_ff=2048)
    
    # 前向传播
    output = decoder_layer(decoder_input, encoder_output, src_mask, tgt_mask)
    
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    
    print("\n✅ 解码器层测试通过！")