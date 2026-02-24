"""
线性回归算法：股价趋势拟合模型
终极全能智能体 · 第一阶段第1天
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

class StockTrendPredictor(nn.Module):
    """线性回归模型：股价趋势拟合"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 单输入单输出
        
    def forward(self, x):
        return self.linear(x)

def generate_stock_data(days=30, stock_name="模拟股票"):
    """生成模拟股价数据"""
    print(f"📊 生成{stock_name}的{days}天模拟数据...")
    
    # 基础趋势：缓慢上涨（年化收益率约20%）
    base_trend = np.linspace(100, 120, days)
    
    # 随机波动（日波动率约3%）
    random_noise = np.random.normal(0, 3, days)
    
    # 周期性波动（模拟周效应）
    weekly_cycle = 2 * np.sin(2 * np.pi * np.arange(days) / 7)
    
    # 趋势加速（模拟市场情绪）
    trend_acceleration = 0.5 * np.sin(2 * np.pi * np.arange(days) / 30)
    
    prices = base_trend + random_noise + weekly_cycle + trend_acceleration
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # 计算技术指标
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
    
    data = {
        "stock_name": stock_name,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "prices": prices.tolist(),
        "returns": returns.tolist(),
        "volatility": float(volatility),
        "days": days,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return data

def train_linear_regression(stock_data, epochs=1000, learning_rate=0.01):
    """训练线性回归模型"""
    print("🎯 开始训练线性回归模型（股价趋势拟合）...")
    
    # 准备数据
    prices = np.array(stock_data["prices"])
    X = torch.arange(len(prices), dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(prices, dtype=torch.float32).reshape(-1, 1)
    
    # 创建模型
    model = StockTrendPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 获取模型参数
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    
    # 预测未来5天
    future_days = torch.arange(len(prices), len(prices) + 5, dtype=torch.float32).reshape(-1, 1)
    future_predictions = model(future_days)
    
    # 计算模型评估指标
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, y).item()
        mae = torch.abs(predictions - y).mean().item()
        r_squared = 1 - mse / torch.var(y).item()
    
    print(f"✅ 训练完成！")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  模型参数: y = {weight:.4f}x + {bias:.4f}")
    print(f"  R²分数: {r_squared:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"📈 未来5天预测: {future_predictions.flatten().tolist()}")
    
    return {
        "model": model,
        "X": X,
        "y": y,
        "future_days": future_days,
        "future_predictions": future_predictions,
        "losses": losses,
        "metrics": {
            "mse": mse,
            "mae": mae,
            "r_squared": r_squared,
            "weight": weight,
            "bias": bias
        }
    }

def visualize_results(stock_data, training_results, save_path="results/linear_regression"):
    """可视化结果"""
    os.makedirs(save_path, exist_ok=True)
    
    model = training_results["model"]
    X = training_results["X"]
    y = training_results["y"]
    future_days = training_results["future_days"]
    future_predictions = training_results["future_predictions"]
    losses = training_results["losses"]
    metrics = training_results["metrics"]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 实际价格 vs 预测价格
    ax1 = axes[0, 0]
    with torch.no_grad():
        predictions = model(X)
    
    dates = stock_data["dates"]
    ax1.plot(dates, y.numpy().flatten(), 'b-', label='实际价格', linewidth=2)
    ax1.plot(dates, predictions.numpy().flatten(), 'r--', label='预测价格', linewidth=2)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{stock_data["stock_name"]} - 线性回归拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 损失曲线
    ax2 = axes[0, 1]
    ax2.plot(losses, 'g-', linewidth=2)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失值')
    ax2.set_title('训练损失曲线')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 残差分析
    ax3 = axes[1, 0]
    residuals = y.numpy().flatten() - predictions.numpy().flatten()
    ax3.scatter(predictions.numpy().flatten(), residuals, alpha=0.6)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('预测值')
    ax3.set_ylabel('残差')
    ax3.set_title('残差分析图')
    ax3.grid(True, alpha=0.3)
    
    # 4. 未来预测
    ax4 = axes[1, 1]
    all_dates = dates + [f'D+{i+1}' for i in range(5)]
    all_prices = list(y.numpy().flatten()) + list(future_predictions.numpy().flatten())
    
    ax4.plot(all_dates[:len(dates)], all_prices[:len(dates)], 'b-', label='历史数据', linewidth=2)
    ax4.plot(all_dates[len(dates)-1:], all_prices[len(dates)-1:], 'r--', label='未来预测', linewidth=2)
    ax4.scatter(all_dates[len(dates):], all_prices[len(dates):], color='red', s=100, zorder=5)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('价格')
    ax4.set_title('未来5天价格预测')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加统计信息文本框
    stats_text = f"""模型统计信息：
R² = {metrics['r_squared']:.4f}
MSE = {metrics['mse']:.4f}
MAE = {metrics['mae']:.4f}
方程: y = {metrics['weight']:.4f}x + {metrics['bias']:.4f}
波动率: {stock_data['volatility']:.2%}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{save_path}/linear_regression_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 可视化结果已保存: {image_path}")
    return image_path

def save_results(stock_data, training_results, image_path, save_dir="results"):
    """保存所有结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"{save_dir}/stock_trend_predictor_{timestamp}.pth"
    torch.save({
        'model_state_dict': training_results["model"].state_dict(),
        'metrics': training_results["metrics"],
        'stock_data': stock_data,
        'training_time': timestamp
    }, model_path)
    
    # 保存JSON结果
    results = {
        "stock_data": stock_data,
        "training_results": {
            "metrics": training_results["metrics"],
            "final_loss": training_results["losses"][-1],
            "future_predictions": training_results["future_predictions"].flatten().tolist(),
            "image_path": image_path,
            "model_path": model_path
        },
        "execution_info": {
            "timestamp": timestamp,
            "algorithm": "linear_regression",
            "framework": "PyTorch",
            "version": "1.0.0"
        }
    }
    
    json_path = f"{save_dir}/linear_regression_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 模型已保存: {model_path}")
    print(f"📄 结果已保存: {json_path}")
    
    return model_path, json_path

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 终极全能智能体 - 线性回归算法（股价趋势拟合）")
    print("=" * 60)
    
    # 1. 生成模拟股票数据
    stock_data = generate_stock_data(days=30, stock_name="AI科技股")
    
    # 2. 训练线性回归模型
    training_results = train_linear_regression(stock_data, epochs=1000, learning_rate=0.01)
    
    # 3. 可视化结果
    image_path = visualize_results(stock_data, training_results)
    
    # 4. 保存结果
    model_path, json_path = save_results(stock_data, training_results, image_path)
    
    # 5. 生成报告
    print("\n" + "=" * 60)
    print("📋 算法执行报告")
    print("=" * 60)
    print(f"股票名称: {stock_data['stock_name']}")
    print(f"数据天数: {stock_data['days']}天")
    print(f"年化波动率: {stock_data['volatility']:.2%}")
    print(f"模型性能: R² = {training_results['metrics']['r_squared']:.4f}")
    print(f"未来5天预测: {training_results['future_predictions'].flatten().tolist()}")
    print(f"趋势方向: {'📈上涨' if training_results['metrics']['weight'] > 0 else '📉下跌'}")
    print(f"文件保存:")
    print(f"  - 模型文件: {model_path}")
    print(f"  - 结果文件: {json_path}")
    print(f"  - 可视化图: {image_path}")
    print("=" * 60)
    
    return {
        "success": True,
        "stock_data": stock_data,
        "training_results": training_results,
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
            print("✅ 线性回归算法执行成功！")
        else:
            print("❌ 线性回归算法执行失败！")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()