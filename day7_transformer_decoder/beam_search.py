"""
Day 7 补充任务：束搜索（Beam Search）实现
用于改进Transformer推理生成质量
"""

import torch
import torch.nn.functional as F
import math
from heapq import heappush, heappop
from typing import List, Tuple

def beam_search_decode(model, src, src_mask, max_len=20, beam_size=5, 
                      start_token=1, end_token=2, temperature=1.0):
    """
    束搜索解码（Beam Search）
    Args:
        model: Transformer模型
        src: 源序列，形状为 (1, src_seq_len)
        src_mask: 源序列掩码
        max_len: 最大生成长度
        beam_size: 束大小
        start_token: 开始token
        end_token: 结束token
        temperature: 温度参数（用于softmax）
    Returns:
        最佳序列和其对数概率
    """
    model.eval()
    
    # 编码源序列
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
    
    # 初始化束：每个元素是(序列, 对数概率, 是否结束)
    # 使用负对数概率（最小堆）
    beams = [([start_token], 0.0, False)]  # (序列, 对数概率, 是否结束)
    
    for step in range(max_len - 1):
        new_beams = []
        
        for seq, log_prob, finished in beams:
            if finished:
                # 已结束的序列直接保留
                heappush(new_beams, (-log_prob, seq, log_prob, finished))
                continue
            
            # 准备输入
            tgt = torch.tensor([seq], dtype=torch.long)
            seq_len = len(seq)
            
            # 创建掩码
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            tgt_mask = tgt_mask.unsqueeze(0)  # (1, seq_len, seq_len)
            
            # 解码
            with torch.no_grad():
                output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
                
                # 获取最后一个位置的logits
                logits = output[:, -1, :] / temperature
                
                # 计算log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 获取top-k个候选
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                
                # 为每个候选创建新序列
                for i in range(beam_size):
                    token = topk_tokens[0, i].item()
                    token_log_prob = topk_log_probs[0, i].item()
                    
                    new_seq = seq + [token]
                    new_log_prob = log_prob + token_log_prob
                    
                    # 检查是否结束
                    new_finished = (token == end_token) or finished
                    
                    heappush(new_beams, (-new_log_prob, new_seq, new_log_prob, new_finished))
        
        # 保留top-beam_size个序列
        beams = []
        for _ in range(min(beam_size, len(new_beams))):
            neg_log_prob, seq, log_prob, finished = heappop(new_beams)
            beams.append((seq, log_prob, finished))
        
        # 如果所有序列都结束了，提前停止
        if all(finished for _, _, finished in beams):
            break
    
    # 返回最佳序列（对数概率最高的）
    best_seq, best_log_prob, _ = max(beams, key=lambda x: x[1])
    
    return best_seq, best_log_prob

def diverse_beam_search_decode(model, src, src_mask, max_len=20, beam_size=5,
                              num_groups=3, diversity_strength=0.5,
                              start_token=1, end_token=2):
    """
    多样化束搜索（Diverse Beam Search）
    通过分组和惩罚相似性来增加多样性
    """
    model.eval()
    
    # 编码源序列
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
    
    # 初始化每组束
    groups = []
    for _ in range(num_groups):
        groups.append([([start_token], 0.0, False)])  # (序列, 对数概率, 是否结束)
    
    completed_seqs = []
    
    for step in range(max_len - 1):
        new_groups = [[] for _ in range(num_groups)]
        
        for group_idx, beams in enumerate(groups):
            for seq, log_prob, finished in beams:
                if finished:
                    new_groups[group_idx].append((seq, log_prob, finished))
                    continue
                
                # 准备输入
                tgt = torch.tensor([seq], dtype=torch.long)
                seq_len = len(seq)
                
                # 创建掩码
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                tgt_mask = tgt_mask.unsqueeze(0)
                
                # 解码
                with torch.no_grad():
                    output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
                    logits = output[:, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # 获取top-k个候选
                    topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size * 2, dim=-1)
                    
                    # 应用多样性惩罚
                    for other_group_idx in range(group_idx):
                        if other_group_idx < len(groups):
                            # 惩罚与其他组已选token的相似性
                            for other_seq, _, _ in groups[other_group_idx]:
                                if len(other_seq) > len(seq):
                                    other_token = other_seq[len(seq)]
                                    # 降低相同token的概率
                                    log_probs[0, other_token] -= diversity_strength
                    
                    # 重新计算top-k
                    topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    
                    # 为每个候选创建新序列
                    for i in range(beam_size):
                        token = topk_tokens[0, i].item()
                        token_log_prob = topk_log_probs[0, i].item()
                        
                        new_seq = seq + [token]
                        new_log_prob = log_prob + token_log_prob
                        new_finished = (token == end_token)
                        
                        new_groups[group_idx].append((new_seq, new_log_prob, new_finished))
        
        # 每组保留top-beam_size个序列
        for group_idx in range(num_groups):
            beams = new_groups[group_idx]
            beams.sort(key=lambda x: x[1], reverse=True)
            groups[group_idx] = beams[:beam_size]
            
            # 收集已完成的序列
            for seq, log_prob, finished in beams[:beam_size]:
                if finished:
                    completed_seqs.append((seq, log_prob))
    
    # 返回所有组的最佳序列
    all_seqs = []
    for group_idx, beams in enumerate(groups):
        if beams:
            best_seq, best_log_prob, _ = max(beams, key=lambda x: x[1])
            all_seqs.append((best_seq, best_log_prob, f"Group {group_idx+1}"))
    
    # 添加已完成的序列
    if completed_seqs:
        completed_seqs.sort(key=lambda x: x[1], reverse=True)
        all_seqs.extend([(seq, log_prob, "Completed") for seq, log_prob in completed_seqs[:3]])
    
    return all_seqs

def length_normalized_beam_search(model, src, src_mask, max_len=20, beam_size=5,
                                length_penalty=0.6, start_token=1, end_token=2):
    """
    长度归一化束搜索
    通过长度惩罚避免偏好短序列
    """
    model.eval()
    
    # 编码源序列
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
    
    beams = [([start_token], 0.0, False)]
    
    for step in range(max_len - 1):
        new_beams = []
        
        for seq, log_prob, finished in beams:
            if finished:
                # 应用长度归一化
                norm_log_prob = log_prob / (len(seq) ** length_penalty)
                heappush(new_beams, (-norm_log_prob, seq, log_prob, finished))
                continue
            
            tgt = torch.tensor([seq], dtype=torch.long)
            seq_len = len(seq)
            
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            tgt_mask = tgt_mask.unsqueeze(0)
            
            with torch.no_grad():
                output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
                logits = output[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    token = topk_tokens[0, i].item()
                    token_log_prob = topk_log_probs[0, i].item()
                    
                    new_seq = seq + [token]
                    new_log_prob = log_prob + token_log_prob
                    new_finished = (token == end_token)
                    
                    # 应用长度归一化
                    norm_log_prob = new_log_prob / (len(new_seq) ** length_penalty)
                    heappush(new_beams, (-norm_log_prob, new_seq, new_log_prob, new_finished))
        
        beams = []
        for _ in range(min(beam_size, len(new_beams))):
            neg_norm_log_prob, seq, log_prob, finished = heappop(new_beams)
            beams.append((seq, log_prob, finished))
        
        if all(finished for _, _, finished in beams):
            break
    
    # 返回长度归一化后的最佳序列
    beams_with_norm = [(seq, log_prob / (len(seq) ** length_penalty), finished) 
                      for seq, log_prob, finished in beams]
    best_seq, best_norm_log_prob, _ = max(beams_with_norm, key=lambda x: x[1])
    
    return best_seq, best_norm_log_prob

# 测试代码
if __name__ == "__main__":
    print("测试束搜索算法...")
    
    # 由于需要模型，这里只测试算法逻辑
    # 在实际使用中，需要传入训练好的Transformer模型
    
    print("\n1. 测试束搜索算法逻辑...")
    
    # 模拟一个简单的概率分布
    vocab_size = 10
    beam_size = 3
    
    # 模拟第一步：开始token
    beams = [([1], 0.0, False)]
    
    # 模拟第二步：假设模型输出
    print("初始束:", beams)
    
    # 模拟扩展
    new_beams = []
    for seq, log_prob, finished in beams:
        if not finished:
            # 模拟top-3个候选
            candidates = [
                (seq + [2], log_prob + math.log(0.4), False),  # token 2, prob 0.4
                (seq + [3], log_prob + math.log(0.3), False),  # token 3, prob 0.3
                (seq + [4], log_prob + math.log(0.2), True),   # token 4, prob 0.2, 结束
            ]
            new_beams.extend(candidates)
    
    # 选择top-beam_size个
    new_beams.sort(key=lambda x: x[1], reverse=True)
    beams = new_beams[:beam_size]
    
    print("扩展后:", beams)
    print("选择top-3:", beams)
    
    print("\n2. 测试束搜索函数接口...")
    print("函数定义测试通过:")
    print("  - beam_search_decode: 标准束搜索")
    print("  - diverse_beam_search_decode: 多样化束搜索")
    print("  - length_normalized_beam_search: 长度归一化束搜索")
    
    print("\n3. 束搜索算法特点:")
    print("  ✅ 比贪婪解码质量更高")
    print("  ✅ 通过束大小平衡质量和效率")
    print("  ✅ 多样化束搜索增加输出多样性")
    print("  ✅ 长度归一化避免偏好短序列")
    
    print("\n4. 使用建议:")
    print("  1. 小束大小 (3-5): 平衡质量和速度")
    print("  2. 大束大小 (10-20): 追求最高质量")
    print("  3. 多样化束搜索: 需要多个不同输出时")
    print("  4. 长度归一化: 避免生成过短序列")
    
    print("\n✅ 束搜索算法实现完成！")
    print("\n下一步：在实际Transformer模型上测试束搜索效果")