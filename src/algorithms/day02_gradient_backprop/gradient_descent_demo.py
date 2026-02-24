"""
梯度下降算法演示 - 第2天：深入理解神经网络训练原理

目标：
1. 手动实现梯度下降算法
2. 可视化梯度下降过程
3. 理解学习率对收敛的影响
4. 应用到简单的线性回归问题
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List

class GradientDescentDemo:
    """梯度下降算法演示类"""
    
    def __init__(self):
        self.history = []
        
    def generate_data(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成线性回归测试数据
        y = 2x + 3 + noise
        """
        np.random.seed(42)
        X = np.random.randn(n_samples, 1) * 2
        true_w = 2.0
        true_b = 3.0
        noise = np.random.randn(n_samples, 1) * 0.5
        y = true_w * X + true_b + noise
        
        return X, y
    
    def compute_loss(self, w: float, b: float, X: np.ndarray, y: np.ndarray) -> float:
        """计算均方误差损失"""
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def compute_gradients(self, w: float, b: float, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """计算损失函数对w和b的梯度"""
        n = len(X)
        y_pred = w * X + b
        
        # 梯度公式：dL/dw = (2/n) * Σ(y_pred - y) * x
        #         dL/db = (2/n) * Σ(y_pred - y)
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)
        
        return dw, db
    
    def gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                        learning_rate: float = 0.01, 
                        n_iterations: int = 100,
                        initial_w: float = 0.0,
                        initial_b: float = 0.0) -> Tuple[float, float, List]:
        """
        执行梯度下降算法
        
        参数：
            X: 输入特征
            y: 目标值
            learning_rate: 学习率
            n_iterations: 迭代次数
            initial_w: 初始权重
            initial_b: 初始偏置
            
        返回：
            w: 最终权重
            b: 最终偏置
            history: 训练历史记录
        """
        w = initial_w
        b = initial_b
        history = []
        
        for i in range(n_iterations):
            # 计算梯度
            dw, db = self.compute_gradients(w, b, X, y)
            
            # 更新参数
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            # 计算当前损失
            loss = self.compute_loss(w, b, X, y)
            
            # 记录历史
            history.append({
                'iteration': i + 1,
                'w': w,
                'b': b,
                'loss': loss,
                'dw': dw,
                'db': db
            })
            
            # 每10次迭代打印进度
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: w={w:.4f}, b={b:.4f}, loss={loss:.4f}")
        
        return w, b, history
    
    def visualize_gradient_descent(self, X: np.ndarray, y: np.ndarray, history: List):
        """可视化梯度下降过程"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 损失函数下降曲线
        iterations = [h['iteration'] for h in history]
        losses = [h['loss'] for h in history]
        
        axes[0, 0].plot(iterations, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('梯度下降：损失函数下降曲线')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 参数w和b的变化轨迹
        w_vals = [h['w'] for h in history]
        b_vals = [h['b'] for h in history]
        
        axes[0, 1].plot(w_vals, b_vals, 'r-', linewidth=2, alpha=0.7)
        axes[0, 1].scatter(w_vals, b_vals, c=iterations, cmap='viridis', s=30)
        axes[0, 1].set_xlabel('权重 w')
        axes[0, 1].set_ylabel('偏置 b')
        axes[0, 1].set_title('梯度下降：参数空间轨迹')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 梯度大小变化
        dw_vals = [h['dw'] for h in history]
        db_vals = [h['db'] for h in history]
        
        axes[1, 0].plot(iterations, dw_vals, 'g-', label='dw/dw', linewidth=2)
        axes[1, 0].plot(iterations, db_vals, 'orange', label='db/db', linewidth=2)
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('梯度值')
        axes[1, 0].set_title('梯度下降：梯度大小变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最终拟合结果
        final_w = w_vals[-1]
        final_b = b_vals[-1]
        y_pred = final_w * X + final_b
        
        axes[1, 1].scatter(X, y, alpha=0.6, label='原始数据')
        axes[1, 1].plot(X, y_pred, 'r-', linewidth=3, label=f'拟合直线: y={final_w:.2f}x+{final_b:.2f}')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('梯度下降：最终拟合结果')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/gradient_descent_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def compare_learning_rates(self, X: np.ndarray, y: np.ndarray):
        """比较不同学习率对梯度下降的影响"""
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        colors = ['blue', 'green', 'red', 'purple']
        
        plt.figure(figsize=(10, 6))
        
        for lr, color in zip(learning_rates, colors):
            _, _, history = self.gradient_descent(X, y, learning_rate=lr, n_iterations=50)
            losses = [h['loss'] for h in history]
            iterations = [h['iteration'] for h in history]
            
            plt.plot(iterations, losses, color=color, linewidth=2, 
                    label=f'学习率={lr}')
        
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('不同学习率对梯度下降收敛速度的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """主函数：演示梯度下降算法"""
    print("=" * 60)
    print("🎯 梯度下降算法演示 - 第2天：神经网络训练原理")
    print("=" * 60)
    
    # 创建演示实例
    demo = GradientDescentDemo()
    
    # 1. 生成测试数据
    print("\n1. 生成测试数据...")
    X, y = demo.generate_data(n_samples=100)
    print(f"   数据形状: X={X.shape}, y={y.shape}")
    print(f"   真实参数: w=2.0, b=3.0")
    
    # 2. 执行梯度下降
    print("\n2. 执行梯度下降算法...")
    print("   初始参数: w=0.0, b=0.0")
    print("   学习率: 0.01, 迭代次数: 100")
    
    w_final, b_final, history = demo.gradient_descent(
        X, y, 
        learning_rate=0.01, 
        n_iterations=100,
        initial_w=0.0,
        initial_b=0.0
    )
    
    print(f"\n   最终参数: w={w_final:.4f}, b={b_final:.4f}")
    print(f"   最终损失: {history[-1]['loss']:.4f}")
    
    # 3. 可视化梯度下降过程
    print("\n3. 可视化梯度下降过程...")
    demo.visualize_gradient_descent(X, y, history)
    
    # 4. 比较不同学习率
    print("\n4. 比较不同学习率的影响...")
    demo.compare_learning_rates(X, y)
    
    # 5. PyTorch自动梯度演示
    print("\n5. PyTorch自动梯度演示...")
    
    # 转换为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    # 定义可训练参数
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    
    # 前向传播
    y_pred = w * X_tensor + b
    loss = torch.mean((y_pred - y_tensor) ** 2)
    
    # 反向传播
    loss.backward()
    
    print(f"   PyTorch计算梯度:")
    print(f"   dw = {w.grad.item():.4f}")
    print(f"   db = {b.grad.item():.4f}")
    
    # 6. 关键概念总结
    print("\n" + "=" * 60)
    print("📚 梯度下降关键概念总结")
    print("=" * 60)
    print("1. 梯度 (Gradient): 损失函数对参数的偏导数")
    print("2. 学习率 (Learning Rate): 控制参数更新步长")
    print("3. 批量梯度下降: 使用全部数据计算梯度")
    print("4. 随机梯度下降: 使用单个样本计算梯度")
    print("5. 小批量梯度下降: 使用小批量数据计算梯度")
    print("6. 动量 (Momentum): 加速收敛，减少震荡")
    print("7. 自适应学习率: Adam, RMSprop等优化器")
    
    return demo, history

if __name__ == "__main__":
    # 确保结果目录存在
    import os
    os.makedirs("results", exist_ok=True)
    
    demo, history = main()