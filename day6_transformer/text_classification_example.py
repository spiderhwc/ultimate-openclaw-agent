"""
Day 6 实践项目：基于Transformer的文本分类示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from transformer_encoder import TextClassifier

class TextClassificationDataset(Dataset):
    """简单的文本分类数据集"""
    
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=1000, num_classes=3):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # 生成随机数据
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # 生成随机文本序列
            text = torch.randint(0, vocab_size, (seq_len,))
            
            # 根据文本特征生成标签（简单规则）
            # 规则1：如果包含某些特定token，则分类为0
            # 规则2：如果序列平均值高，则分类为1
            # 规则3：否则分类为2
            if torch.any(text < 100):  # 包含小token
                label = 0
            elif text.float().mean() > vocab_size * 0.7:  # 平均值高
                label = 1
            else:
                label = 2
                
            self.data.append(text)
            self.labels.append(label)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    print("开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            logits, _ = model(data)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                logits, _ = model(data)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_accuracies

def analyze_attention(model, sample_data):
    """分析注意力机制"""
    print("\n分析注意力机制...")
    
    model.eval()
    with torch.no_grad():
        logits, all_attention_weights = model(sample_data.unsqueeze(0))
        
    print(f"输入序列: {sample_data.tolist()}")
    print(f"预测类别: {torch.argmax(logits, dim=1).item()}")
    
    # 分析第一层的注意力权重
    first_layer_attention = all_attention_weights[0][0]  # (n_heads, seq_len, seq_len)
    print(f"\n第一层注意力权重形状: {first_layer_attention.shape}")
    
    # 计算每个头的平均注意力分布
    for head_idx in range(first_layer_attention.shape[0]):
        attention_matrix = first_layer_attention[head_idx]
        print(f"\n头 {head_idx} 的注意力分布:")
        
        # 计算每个token对其他token的平均注意力
        for token_idx in range(min(5, attention_matrix.shape[0])):  # 只显示前5个token
            attention_to_others = attention_matrix[token_idx]
            top_3_indices = torch.topk(attention_to_others, 3).indices.tolist()
            top_3_values = torch.topk(attention_to_others, 3).values.tolist()
            
            print(f"  Token {token_idx} (值={sample_data[token_idx]}) 最关注: ", end="")
            for idx, val in zip(top_3_indices, top_3_values):
                print(f"Token {idx}({sample_data[idx]}:{val:.3f}) ", end="")
            print()

def compare_with_rnn():
    """与RNN模型对比"""
    print("\n" + "=" * 60)
    print("Transformer vs RNN 对比分析")
    print("=" * 60)
    
    # Transformer优势
    print("\n📊 Transformer的优势:")
    print("1. 并行计算: 可以同时处理所有位置，训练速度快")
    print("2. 长距离依赖: 自注意力机制可以直接捕捉任意距离的依赖")
    print("3. 可解释性: 注意力权重可视化，理解模型关注点")
    print("4. 扩展性: 多头注意力可以学习不同的表示子空间")
    
    # RNN局限性
    print("\n⚠️  RNN的局限性:")
    print("1. 顺序计算: 必须按顺序处理序列，训练慢")
    print("2. 梯度消失: 长序列中梯度容易消失")
    print("3. 信息瓶颈: 隐藏状态可能成为信息瓶颈")
    print("4. 并行性差: 难以利用GPU并行计算")
    
    # 实际应用场景
    print("\n🎯 实际应用场景:")
    print("• Transformer: 机器翻译、文本生成、BERT/GPT等大模型")
    print("• RNN/LSTM: 实时序列处理、时间序列预测、简单分类任务")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("Day 6 实践项目：基于Transformer的文本分类")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建数据集
    print("\n创建数据集...")
    dataset = TextClassificationDataset(
        num_samples=1000, 
        seq_len=20, 
        vocab_size=500, 
        num_classes=3
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建Transformer文本分类器...")
    model = TextClassifier(
        vocab_size=500,
        num_classes=3,
        d_model=128,      # 较小的模型，便于快速训练
        n_layers=2,       # 2层编码器
        n_heads=4,        # 4个头
        d_ff=512,         # 前馈网络维度
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=5, learning_rate=0.001
    )
    
    # 分析注意力
    print("\n" + "=" * 60)
    print("注意力机制分析")
    print("=" * 60)
    
    # 选择一个样本进行分析
    sample_idx = random.randint(0, len(val_dataset) - 1)
    sample_data, sample_label = val_dataset[sample_idx]
    analyze_attention(model, sample_data)
    
    # 与RNN对比
    compare_with_rnn()
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 Day 6 学习总结")
    print("=" * 60)
    
    print("\n✅ 已完成的学习内容:")
    print("1. 自注意力机制原理和实现")
    print("2. 多头注意力机制")
    print("3. 位置编码（正弦余弦编码）")
    print("4. Transformer编码器架构")
    print("5. 位置级前馈网络")
    print("6. 残差连接和层归一化")
    print("7. 基于Transformer的文本分类器")
    print("8. 注意力权重可视化分析")
    
    print("\n📈 学习成果:")
    print(f"• 最终验证准确率: {val_accuracies[-1]:.4f}")
    print("• 成功实现完整的Transformer编码器")
    print("• 掌握注意力机制的可视化分析")
    print("• 理解Transformer vs RNN的优劣")
    
    print("\n🚀 下一步:")
    print("Day 7: Transformer解码器 + 完整Transformer模型")
    print("Day 8: 预训练模型（BERT/GPT）原理")
    print("Day 9: 微调预训练模型")
    
    return True

if __name__ == "__main__":
    main()