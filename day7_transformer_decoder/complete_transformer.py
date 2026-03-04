"""
Day 7 任务：完整Transformer模型实现
包含编码器、解码器和完整的Transformer类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# 导入基础组件
from transformer_decoder import (
    MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer,
    PositionalEncoding, create_padding_mask, create_decoder_mask,
    create_lookahead_mask
)

class Encoder(nn.Module):
    """Transformer编码器（多层编码器层堆叠）"""
    
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 创建多层编码器
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        """
        前向传播
        Args:
            src: 源序列，形状为 (batch_size, src_seq_len)
            src_mask: 源序列掩码
        Returns:
            编码器输出
        """
        # 词嵌入
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 通过多层编码器
        for layer in self.layers:
            x = layer(x, src_mask)
            
        x = self.layer_norm(x)
        
        return x

class Decoder(nn.Module):
    """Transformer解码器（多层解码器层堆叠）"""
    
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 创建多层解码器
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        Args:
            tgt: 目标序列，形状为 (batch_size, tgt_seq_len)
            encoder_output: 编码器输出，形状为 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            解码器输出
        """
        # 词嵌入
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 通过多层解码器
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        x = self.layer_norm(x)
        
        return x

class Transformer(nn.Module):
    """完整的Transformer模型（编码器-解码器架构）"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, 
                              d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads,
                              d_ff, dropout, max_len)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播（训练模式）
        Args:
            src: 源序列，形状为 (batch_size, src_seq_len)
            tgt: 目标序列，形状为 (batch_size, tgt_seq_len)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            模型输出，形状为 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 编码器前向传播
        encoder_output = self.encoder(src, src_mask)
        
        # 解码器前向传播
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_layer(decoder_output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """仅编码（用于推理）"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """仅解码（用于推理）"""
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

class SimpleTranslationDataset:
    """简单的机器翻译数据集（用于演示）"""
    
    def __init__(self, num_samples=1000, max_len=20, vocab_size=1000):
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # 生成随机数据
        self.src_data = []
        self.tgt_data = []
        
        for _ in range(num_samples):
            # 生成随机源序列（英语）
            src_len = random.randint(5, max_len)
            src = torch.randint(2, vocab_size//2, (src_len,))  # 英语词汇
            
            # 生成目标序列（法语）- 简单转换：每个token加一个偏移量
            tgt_len = src_len + random.randint(-2, 2)  # 长度可能略有不同
            tgt = src[:tgt_len] + vocab_size//2  # 法语词汇（偏移）
            
            # 填充到固定长度
            src_padded = F.pad(src, (0, max_len - src_len), value=0)
            tgt_padded = F.pad(tgt, (0, max_len - tgt_len), value=0)
            
            self.src_data.append(src_padded)
            self.tgt_data.append(tgt_padded)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def train_transformer(model, train_data, val_data, num_epochs=10, 
                     batch_size=32, learning_rate=0.001):
    """训练Transformer模型"""
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充token
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            # 创建掩码
            src_mask = create_padding_mask(src, pad_idx=0)
            tgt_mask = create_decoder_mask(tgt, pad_idx=0)
            
            # 前向传播
            # 对于训练，我们使用teacher forcing：输入是完整的目标序列（去掉最后一个token）
            # 输出应该预测目标序列（去掉第一个token）
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 调整掩码
            tgt_mask = tgt_mask[:, :-1, :-1]
            
            # 模型前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), 
                           tgt_output.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src_mask = create_padding_mask(src, pad_idx=0)
                tgt_mask = create_decoder_mask(tgt, pad_idx=0)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = tgt_mask[:, :-1, :-1]
                
                output = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), 
                               tgt_output.reshape(-1))
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)
    
    return model

def greedy_decode(model, src, src_mask, max_len=20, start_token=1, end_token=2):
    """
    贪婪解码（推理时生成序列）
    Args:
        model: Transformer模型
        src: 源序列，形状为 (1, src_seq_len)
        src_mask: 源序列掩码
        max_len: 最大生成长度
        start_token: 开始token
        end_token: 结束token
    Returns:
        生成的序列
    """
    model.eval()
    
    # 编码源序列
    encoder_output = model.encode(src, src_mask)
    
    # 初始化目标序列（只有开始token）
    tgt = torch.tensor([[start_token]], dtype=torch.long)
    
    for i in range(max_len - 1):
        # 获取当前序列长度
        seq_len = tgt.size(1)
        
        # 为推理创建解码器掩码（只有前瞻掩码）
        tgt_mask = create_lookahead_mask(seq_len)
        tgt_mask = (tgt_mask == 0).bool().unsqueeze(0)  # (1, seq_len, seq_len)
        
        # 解码
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 获取下一个token（贪婪选择）
        next_token = output[:, -1:, :].argmax(dim=-1)
        
        # 添加到序列
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # 如果生成结束token，停止
        if next_token.item() == end_token:
            break
    
    return tgt

# 测试代码
if __name__ == "__main__":
    import random
    
    print("测试完整Transformer模型...")
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    
    # 创建模拟数据集
    print("\n1. 创建模拟数据集...")
    vocab_size = 1000
    dataset = SimpleTranslationDataset(num_samples=100, max_len=15, vocab_size=vocab_size)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"数据集大小: {len(dataset)}")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    
    # 创建模型
    print("\n2. 创建Transformer模型...")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,  # 使用较小的维度以加快训练
        n_layers=2,   # 使用较少的层数
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 12
    
    # 创建模拟批次数据
    src = torch.randint(1, vocab_size//2, (batch_size, src_seq_len))
    tgt = torch.randint(vocab_size//2, vocab_size, (batch_size, tgt_seq_len))
    
    # 创建掩码
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_decoder_mask(tgt, pad_idx=0)
    
    # 调整目标序列用于训练（teacher forcing）
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_mask = tgt_mask[:, :-1, :-1]
    
    # 前向传播
    output = model(src, tgt_input, src_mask, tgt_mask)
    
    print(f"输入形状:")
    print(f"  src: {src.shape}")
    print(f"  tgt_input: {tgt_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"目标形状: {tgt_output.shape}")
    
    # 测试贪婪解码
    print("\n4. 测试贪婪解码...")
    test_src = torch.randint(1, vocab_size//2, (1, 8))
    test_src_mask = create_padding_mask(test_src, pad_idx=0)
    
    generated = greedy_decode(model, test_src, test_src_mask, max_len=10)
    print(f"源序列: {test_src[0].tolist()}")
    print(f"生成序列: {generated[0].tolist()}")
    
    # 训练模型（简化版，只训练几个批次）
    print("\n5. 简化训练测试...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for i in range(3):  # 只训练3个批次
        # 获取一个批次
        src_batch, tgt_batch = next(iter(torch.utils.data.DataLoader(
            train_data, batch_size=batch_size
        )))
        
        # 创建掩码
        src_mask = create_padding_mask(src_batch, pad_idx=0)
        tgt_mask = create_decoder_mask(tgt_batch, pad_idx=0)
        
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        tgt_mask = tgt_mask[:, :-1, :-1]
        
        # 前向传播
        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), 
                        tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}, Loss: {loss.item():.4f}")
    
    print("\n✅ 完整Transformer模型测试通过！")
    print("\n下一步：")
    print("1. 使用更大数据集训练")
    print("2. 实现束搜索（Beam Search）")
    print("3. 添加学习率调度器")
    print("4. 添加梯度裁剪")
    print("5. 使用真实翻译数据集（如WMT）")