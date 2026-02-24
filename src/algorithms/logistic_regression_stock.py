"""
逻辑回归算法：股票涨跌预测模型
终极全能智能体 · 第一阶段第1天
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import json
import os
from datetime import datetime

class StockUpDownPredictor(nn.Module):
    """逻辑回归模型：股票涨跌二分类预测"""
    def __init__(self, input_features=5):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

def generate_stock_features(n_samples=1000, feature_names=None):
    """生成股票特征数据"""
    print(f"📊 生成{n_samples}个股票特征样本...")
    
    if feature_names is None:
        feature_names = [
            "价格动量",      # 过去5天收益率
            "成交量变化率",  # 成交量变化
            "RSI指标",      # 相对强弱指标
            "MACD指标",     # 移动平均收敛发散
            "波动率"        # 平均真实范围
        ]
    
    np.random.seed(42)
    
    # 1. 价格动量（过去5天收益率）- 均值为正表示上涨趋势
    momentum_mean = 0.001  # 轻微上涨趋势
    momentum = np.random.normal(momentum_mean, 0.02, n_samples)
    
    # 2. 成交量变化率 - 上涨时成交量通常增加
    volume_change_mean = 0.1  # 成交量平均增加10%
    volume_change = np.random.normal(volume_change_mean, 0.3, n_samples)
    
    # 3. RSI相对强弱指标（30-70为正常范围）
    rsi_mean = 50  # 中性
    rsi = np.random.normal(rsi_mean, 10, n_samples)
    rsi = np.clip(rsi, 30, 70)  # 限制在30-70之间
    
    # 4. MACD指标 - 正值表示上涨动量
    macd_mean = 0.2  # 轻微上涨动量
    macd = np.random.normal(macd_mean, 0.5, n_samples)
    
    # 5. 波动率（ATR） - 高波动率可能伴随趋势
    volatility_mean = 0.03  # 3%的平均日波动率
    volatility = np.random.uniform(0.01, 0.05, n_samples)
    
    # 标签：1=上涨，0=下跌（基于综合特征）
    # 使用更复杂的规则模拟真实市场
    labels = np.zeros(n_samples, dtype=int)
    
    # 规则1：动量>0且RSI>50 → 上涨概率高
    condition1 = (momentum > 0) & (rsi > 50)
    
    # 规则2：MACD>0且成交量增加 → 上涨概率高
    condition2 = (macd > 0) & (volume_change > 0)
    
    # 规则3：低波动率下的正动量 → 稳定上涨
    condition3 = (momentum > 0.005) & (volatility < 0.02)
    
    # 综合条件
    labels = ((condition1.astype(int) + condition2.astype(int) + condition3.astype(int)) >= 2).astype(int)
    
    # 添加市场噪声（20%的样本标签随机翻转）
    noise_mask = np.random.random(n_samples) < 0.2
    labels[noise_mask] = 1 - labels[noise_mask]
    
    # 计算类别平衡
    positive_ratio = labels.mean()
    print(f"  上涨样本比例: {positive_ratio:.2%}")
    
    features = np.column_stack([momentum, volume_change, rsi, macd, volatility])
    
    return {
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
        "n_samples": n_samples,
        "positive_ratio": positive_ratio,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def train_logistic_regression(data, epochs=500, learning_rate=0.01, test_size=0.2):
    """训练逻辑回归模型"""
    print("🎯 开始训练逻辑回归模型（股票涨跌预测）...")
    
    # 准备数据
    X_np = data["features"]
    y_np = data["labels"]
    
    # 划分训练集和测试集
    n_samples = len(X_np)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]
    
    X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
    y_train_np, y_test_np = y_np[train_idx], y_np[test_idx]
    
    # 标准化特征（仅使用训练集拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    # 转换为PyTorch张量
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)
    
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")
    
    # 创建模型
    input_features = X_train.shape[1]
    model = StockUpDownPredictor(input_features=input_features)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        train_predictions = (train_outputs > 0.5).float()
        train_accuracy = (train_predictions == y_train).float().mean()
        
        # 测试阶段
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions == y_test).float().mean()
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accuracies.append(train_accuracy.item())
        test_accuracies.append(test_accuracy.item())
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_loss.item():.4f}, "
                  f"Train Acc = {train_accuracy.item():.2%}, "
                  f"Test Acc = {test_accuracy.item():.2%}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        # 训练集评估
        train_outputs = model(X_train)
        train_predictions = (train_outputs > 0.5).float().numpy().flatten()
        train_labels = y_train.numpy().flatten()
        
        # 测试集评估
        test_outputs = model(X_test)
        test_predictions = (test_outputs > 0.5).float().numpy().flatten()
        test_labels = y_test.numpy().flatten()
        test_probabilities = test_outputs.numpy().flatten()
    
    # 计算详细指标
    train_metrics = calculate_metrics(train_labels, train_predictions, "训练集")
    test_metrics = calculate_metrics(test_labels, test_predictions, "测试集")
    
    # 计算特征重要性（基于权重绝对值）
    feature_importance = np.abs(model.linear.weight.detach().numpy().flatten())
    feature_importance = feature_importance / feature_importance.sum()
    
    print(f"✅ 训练完成！")
    print(f"  最终训练准确率: {train_accuracies[-1]:.2%}")
    print(f"  最终测试准确率: {test_accuracies[-1]:.2%}")
    
    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_labels": test_labels,
        "feature_importance": feature_importance,
        "feature_names": data["feature_names"]
    }

def calculate_metrics(true_labels, pred_labels, dataset_name):
    """计算分类指标"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "dataset": dataset_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

def visualize_results(data, training_results, save_path="results/logistic_regression"):
    """可视化结果"""
    os.makedirs(save_path, exist_ok=True)
    
    train_losses = training_results["train_losses"]
    test_losses = training_results["test_losses"]
    train_accuracies = training_results["train_accuracies"]
    test_accuracies = training_results["test_accuracies"]
    test_probabilities = training_results["test_probabilities"]
    test_labels = training_results["test_labels"]
    feature_importance = training_results["feature_importance"]
    feature_names = training_results["feature_names"]
    train_metrics = training_results["train_metrics"]
    test_metrics = training_results["test_metrics"]
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(test_losses, 'r--', label='测试损失', linewidth=2)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练与测试损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(train_accuracies, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(test_accuracies, 'r--', label='测试准确率', linewidth=2)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练与测试准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. 特征重要性
    ax3 = axes[0, 2]
    colors = plt.cm.Set3(np.arange(len(feature_names)))
    bars = ax3.barh(feature_names, feature_importance, color=colors)
    ax3.set_xlabel('重要性权重')
    ax3.set_title('特征重要性分析')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, importance in zip(bars, feature_importance):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.2%}', ha='left', va='center')
    
    # 4. ROC曲线
    ax4 = axes[1, 0]
    fpr, tpr, thresholds = roc_curve(test_labels, test_probabilities)
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {roc_auc:.3f})', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'r--', label='随机分类器', linewidth=1, alpha=0.5)
    ax4.set_xlabel('假正率')
    ax4.set_ylabel('真正率')
    ax4.set_title('ROC曲线')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # 5. 预测概率分布
    ax5 = axes[1, 1]
    # 按真实标签分组
    prob_up = test_probabilities[test_labels == 1]
    prob_down = test_probabilities[test_labels == 0]
    
    ax5.hist(prob_up, bins=20, alpha=0.7, label='实际上涨', color='green')
    ax5.hist(prob_down, bins=20, alpha=0.7, label='实际下跌', color='red')
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax5.set_xlabel('预测概率')
    ax5.set_ylabel('样本数量')
    ax5.set_title('预测概率分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 混淆矩阵热图
    ax6 = axes[1, 2]
    cm = np.array(test_metrics["confusion_matrix"])
    im = ax6.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax6.set_title('测试集混淆矩阵')
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax6.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    tick_marks = np.arange(2)
    ax6.set_xticks(tick_marks)
    ax6.set_yticks(tick_marks)
    ax6.set_xticklabels(['下跌', '上涨'])
    ax6.set_yticklabels(['下跌', '上涨'])
    ax6.set_ylabel('真实标签')
    ax6.set_xlabel('预测标签')
    
    # 添加统计信息文本框
    stats_text = f"""模型性能统计：
测试集准确率: {test_metrics['accuracy']:.2%}
精确率: {test_metrics['precision']:.2%}
召回率: {test_metrics['recall']:.2%}
F1分数: {test_metrics['f1_score']:.2%}
AUC分数: {roc_auc:.3f}

混淆矩阵:
TN={test_metrics['true_negative']} FP={test_metrics['false_positive']}
FN={test_metrics['false_negative']} TP={test_metrics['true_positive']}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{save_path}/logistic_regression_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 可视化结果已保存: {image_path}")
    return image_path, roc_auc

def save_results(data, training_results, image_path, roc_auc, save_dir="results"):
    """保存所有结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"{save_dir}/stock_updown_predictor_{timestamp}.pth"
    torch.save({
        'model_state_dict': training_results["model"].state_dict(),
        'scaler_mean': training_results["scaler"].mean_,
        'scaler_scale': training_results["scaler"].scale_,
        'feature_names': training_results["feature_names"],
        'training_metrics': training_results["train_metrics"],
        'test_metrics': training_results["test_metrics"],
        'roc_auc': roc_auc,
        'training_time': timestamp
    }, model_path)
    
    # 保存JSON结果
    results = {
        "data_info": {
            "n_samples": data["n_samples"],
            "positive_ratio": data["positive_ratio"],
            "feature_names": data["feature_names"],
            "generated_at": data["generated_at"]
        },
        "training_results": {
            "train_metrics": training_results["train_metrics"],
            "test_metrics": training_results["test_metrics"],
            "roc_auc": float(roc_auc),
            "feature_importance": training_results["feature_importance"].tolist(),
            "final_train_accuracy": training_results["train_accuracies"][-1],
            "final_test_accuracy": training_results["test_accuracies"][-1],
            "image_path": image_path,
            "model_path": model_path
        },
        "execution_info": {
            "timestamp": timestamp,
            "algorithm": "logistic_regression",
            "framework": "PyTorch",
            "version": "1.0.0"
        }
    }
    
    json_path = f"{save_dir}/logistic_regression_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 模型已保存: {model_path}")
    print(f"📄 结果已保存: {json_path}")
    
    return model_path, json_path

def predict_new_sample(model, scaler, feature_names, sample_features):
    """预测新样本"""
    # 标准化特征
    sample_scaled = scaler.transform([sample_features])
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
    
    # 预测
    model.eval()
    with torch.no_grad():
        probability = model(sample_tensor).item()
    
    prediction = "📈上涨" if probability > 0.5 else "📉下跌"
    confidence = probability if probability > 0.5 else 1 - probability
    
    print(f"\n🔮 新样本预测结果:")
    print(f"  特征值: {dict(zip(feature_names, sample_features))}")
    print(f"  预测: {prediction}")
    print(f"  概率: {probability:.2%}")
    print(f"  置信度: {confidence:.2%}")
    
    return {
        "prediction": "上涨" if probability > 0.5 else "下跌",
        "probability": probability,
        "confidence": confidence,
        "features": dict(zip(feature_names, sample_features))
    }

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 终极全能智能体 - 逻辑回归算法（股票涨跌预测）")
    print("=" * 60)
    
    # 1. 生成股票特征数据
    feature_names = ["价格动量", "成交量变化率", "RSI指标", "MACD指标", "波动率"]
    data = generate_stock_features(n_samples=1000, feature_names=feature_names)
    
    # 2. 训练逻辑回归模型
    training_results = train_logistic_regression(data, epochs=500, learning_rate=0.01)
    
    # 3. 可视化结果
    image_path, roc_auc = visualize_results(data, training_results)
    
    # 4. 保存结果
    model_path, json_path = save_results(data, training_results, image_path, roc_auc)
    
    # 5. 测试新样本预测
    print("\n🧪 测试新样本预测:")
    
    # 测试样本1：强势上涨特征
    strong_up_features = [0.02, 0.25, 65, 0.8, 0.02]
    predict_new_sample(training_results["model"], training_results["scaler"], 
                      feature_names, strong_up_features)
    
    # 测试样本2：明显下跌特征
    strong_down_features = [-0.015, -0.1, 35, -0.5, 0.04]
    predict_new_sample(training_results["model"], training_results["scaler"], 
                      feature_names, strong_down_features)
    
    # 测试样本3：中性特征
    neutral_features = [0.001, 0.05, 50, 0.1, 0.03]
    predict_new_sample(training_results["model"], training_results["scaler"], 
                      feature_names, neutral_features)
    
    # 6. 生成报告
    print("\n" + "=" * 60)
    print("📋 算法执行报告")
    print("=" * 60)
    print(f"数据规模: {data['n_samples']} 个样本")
    print(f"类别平衡: 上涨 {data['positive_ratio']:.2%}")
    print(f"模型性能:")
    print(f"  测试准确率: {training_results['test_metrics']['accuracy']:.2%}")
    print(f"  精确率: {training_results['test_metrics']['precision']:.2%}")
    print(f"  召回率: {training_results['test_metrics']['recall']:.2%}")
    print(f"  F1分数: {training_results['test_metrics']['f1_score']:.2%}")
    print(f"  AUC分数: {roc_auc:.3f}")
    print(f"特征重要性:")
    for name, importance in zip(feature_names, training_results["feature_importance"]):
        print(f"  {name}: {importance:.2%}")
    print(f"文件保存:")
    print(f"  - 模型文件: {model_path}")
    print(f"  - 结果文件: {json_path}")
    print(f"  - 可视化图: {image_path}")
    print("=" * 60)
    
    return {
        "success": True,
        "data": data,
        "training_results": training_results,
        "roc_auc": roc_auc,
        "files": {
            "model": model_path,
            "results": json_path,
            "image": image_path
        }
    }

if __name__ == "__main__":
    try:
        result = main()
        if result["success"]:
            print("✅ 逻辑回归算法执行成功！")
        else:
            print("❌ 逻辑回归算法执行失败！")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()