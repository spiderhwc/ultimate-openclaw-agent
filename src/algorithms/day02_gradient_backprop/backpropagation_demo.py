"""
反向传播算法演示 - 第2天：深入理解神经网络训练原理

目标：
1. 手动实现反向传播算法
2. 理解链式法则在神经网络中的应用
3. 可视化反向传播过程
4. 实现简单的两层神经网络
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json

class SimpleNeuralNetwork:
    """简单的两层神经网络（输入层-隐藏层-输出层）"""
    
    def __init__(self, input_size: int = 2, hidden_size: int = 3, output_size: int = 1):
        """
        初始化神经网络
        
        参数：
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置（小随机值）
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1  # 输入到隐藏层的权重
        self.b1 = np.zeros((1, hidden_size))                      # 隐藏层偏置
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1 # 隐藏到输出层的权重
        self.b2 = np.zeros((1, output_size))                      # 输出层偏置
        
        # 存储中间值用于反向传播
        self.cache = {}
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数的导数"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数：
            X: 输入数据，形状 (n_samples, input_size)
            
        返回：
            output: 网络输出，形状 (n_samples, output_size)
        """
        # 隐藏层计算
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.sigmoid(self.cache['Z1'])
        
        # 输出层计算
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        return self.cache['A2']
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        反向传播计算梯度
        
        使用链式法则：
        1. 计算输出层梯度
        2. 计算隐藏层梯度
        3. 计算权重和偏置的梯度
        """
        n_samples = X.shape[0]
        
        # 输出层梯度
        dA2 = 2 * (y_pred - y_true) / n_samples  # dL/dA2
        dZ2 = dA2 * self.sigmoid_derivative(self.cache['Z2'])  # dL/dZ2 = dL/dA2 * dA2/dZ2
        
        # 隐藏层到输出层的权重和偏置梯度
        dW2 = np.dot(self.cache['A1'].T, dZ2)  # dL/dW2 = A1^T * dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # dL/db2 = Σ dZ2
        
        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)  # dL/dA1 = dZ2 * W2^T
        dZ1 = dA1 * self.sigmoid_derivative(self.cache['Z1'])  # dL/dZ1 = dL/dA1 * dA1/dZ1
        
        # 输入层到隐藏层的权重和偏置梯度
        dW1 = np.dot(X.T, dZ1)  # dL/dW1 = X^T * dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # dL/db1 = Σ dZ1
        
        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
        
        return gradients
    
    def update_parameters(self, gradients: Dict, learning_rate: float = 0.01):
        """使用梯度下降更新参数"""
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              learning_rate: float = 0.1, 
              n_iterations: int = 1000,
              verbose: bool = True) -> Dict:
        """
        训练神经网络
        
        返回：
            history: 训练历史记录
        """
        history = {
            'losses': [],
            'W1_norm': [],
            'W2_norm': [],
            'grad_norm': []
        }
        
        for i in range(n_iterations):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            
            # 反向传播
            gradients = self.backward(X, y, y_pred)
            
            # 更新参数
            self.update_parameters(gradients, learning_rate)
            
            # 记录历史
            history['losses'].append(loss)
            history['W1_norm'].append(np.linalg.norm(self.W1))
            history['W2_norm'].append(np.linalg.norm(self.W2))
            
            # 计算梯度范数
            grad_norm = np.sqrt(
                np.sum(gradients['dW1']**2) + 
                np.sum(gradients['db1']**2) + 
                np.sum(gradients['dW2']**2) + 
                np.sum(gradients['db2']**2)
            )
            history['grad_norm'].append(grad_norm)
            
            # 打印进度
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}")
        
        return history

class BackpropagationVisualizer:
    """反向传播可视化类"""
    
    def __init__(self):
        pass
    
    def generate_xor_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成XOR问题数据"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X, y
    
    def visualize_neural_network(self, nn: SimpleNeuralNetwork, X: np.ndarray, y: np.ndarray):
        """可视化神经网络结构和前向传播"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 神经网络结构图
        ax = axes[0, 0]
        self._draw_neural_network(ax, nn)
        ax.set_title('神经网络结构 (2-3-1)')
        ax.axis('off')
        
        # 2. 输入数据和目标输出
        ax = axes[0, 1]
        colors = ['red' if val == 0 else 'blue' for val in y.flatten()]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, alpha=0.7, edgecolors='black')
        ax.set_xlabel('输入 x1')
        ax.set_ylabel('输入 x2')
        ax.set_title('XOR问题输入数据')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        
        # 3. 隐藏层激活可视化
        ax = axes[1, 0]
        hidden_activations = nn.cache['A1']
        for i in range(hidden_activations.shape[1]):
            ax.bar(i, hidden_activations[0, i], alpha=0.7, label=f'神经元 {i+1}')
        ax.set_xlabel('隐藏层神经元')
        ax.set_ylabel('激活值')
        ax.set_title('隐藏层激活值 (第一个样本)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 网络输出
        ax = axes[1, 1]
        predictions = nn.forward(X)
        ax.bar(range(len(predictions)), predictions.flatten(), alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='决策边界 (0.5)')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('预测概率')
        ax.set_title('神经网络预测输出')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/neural_network_structure.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _draw_neural_network(self, ax, nn: SimpleNeuralNetwork):
        """绘制神经网络结构图"""
        # 定义层的位置
        layer_positions = [
            [(0, i) for i in range(nn.input_size)],      # 输入层
            [(1, i) for i in range(nn.hidden_size)],     # 隐藏层
            [(2, i) for i in range(nn.output_size)]      # 输出层
        ]
        
        # 绘制神经元
        for layer_idx, positions in enumerate(layer_positions):
            for pos in positions:
                circle = plt.Circle(pos, 0.1, color='lightblue', ec='black', zorder=2)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], f'L{layer_idx+1}', 
                       ha='center', va='center', fontsize=8)
        
        # 绘制连接线（权重）
        for i in range(nn.input_size):
            for j in range(nn.hidden_size):
                weight = nn.W1[i, j]
                color = 'red' if weight < 0 else 'green'
                alpha = min(abs(weight) * 2, 0.8)
                ax.plot([0, 1], [i, j], color=color, alpha=alpha, linewidth=abs(weight)*3)
        
        for i in range(nn.hidden_size):
            for j in range(nn.output_size):
                weight = nn.W2[i, j]
                color = 'red' if weight < 0 else 'green'
                alpha = min(abs(weight) * 2, 0.8)
                ax.plot([1, 2], [i, j], color=color, alpha=alpha, linewidth=abs(weight)*3)
        
        # 设置坐标轴
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1, max(nn.input_size, nn.hidden_size, nn.output_size))
        ax.set_aspect('equal')
    
    def visualize_training_history(self, history: Dict):
        """可视化训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 损失函数下降
        axes[0, 0].plot(history['losses'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('反向传播：损失函数下降曲线')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 权重范数变化
        axes[0, 1].plot(history['W1_norm'], 'g-', label='W1范数', linewidth=2)
        axes[0, 1].plot(history['W2_norm'], 'orange', label='W2范数', linewidth=2)
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('权重范数')
        axes[0, 1].set_title('反向传播：权重范数变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 梯度范数变化
        axes[1, 0].plot(history['grad_norm'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('梯度范数')
        axes[1, 0].set_title('反向传播：梯度范数变化')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失vs梯度（散点图）
        axes[1, 1].scatter(history['losses'], history['grad_norm'], 
                          c=range(len(history['losses'])), cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('损失值')
        axes[1, 1].set_ylabel('梯度范数')
        axes[1, 1].set_title('反向传播：损失vs梯度关系')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/backprop_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_backpropagation_step(self, nn: SimpleNeuralNetwork, gradients: Dict):
        """可视化单步反向传播的梯度"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. W1梯度热图
        im1 = axes[0, 0].imshow(gradients['dW1'], cmap='RdBu', aspect='auto')
        axes[0, 0].set_title('输入层到隐藏层权重梯度 (dW1)')
        axes[0, 0].set_xlabel('隐藏层神经元')
        axes[0, 0].set_ylabel('输入层神经元')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. W2梯度热图
        im2 = axes[0, 1].imshow(gradients['dW2'], cmap='RdBu', aspect='auto')
        axes[0, 1].set_title('隐藏层到输出层权重梯度 (dW2)')
        axes[0, 1].set_xlabel('输出层神经元')
        axes[0, 1].set_ylabel('隐藏层神经元')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. b1梯度
        axes[1, 0].bar(range(len(gradients['db1'].flatten())), 
                      gradients['db1'].flatten())
        axes[1, 0].set_title('隐藏层偏置梯度 (db1)')
        axes[1, 0].set_xlabel('隐藏层神经元')
        axes[1, 0].set_ylabel('梯度值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. b2梯度
        axes[1, 1].bar(range(len(gradients['db2'].flatten())), 
                      gradients['db2'].flatten())
        axes[1, 1].set_title('输出层偏置梯度 (db2)')
        axes[1, 1].set_xlabel('输出层神经元')
        axes[1, 1].set_ylabel('梯度值')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/backprop_gradients.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """主函数：演示反向传播算法"""
    print("=" * 60)
    print("🎯 反向传播算法演示 - 第2天：神经网络训练原理")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = BackpropagationVisualizer()
    
    # 1. 生成XOR问题数据
    print("\n1. 生成XOR问题数据...")
    X, y = visualizer.generate_xor_data()
    print(f"   输入数据:\n{X}")
    print(f"   目标输出:\n{y}")
    
    # 2. 创建神经网络
    print("\n2. 创建神经网络 (2-3-1结构)...")
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    print(f"   网络结构: 输入层({nn.input_size}) -> 隐藏层({nn.hidden_size}) -> 输出层({nn.output_size})")
    print(f"   权重形状: W1={nn.W1.shape}, W2={nn.W2.shape}")
    print(f"   偏置形状: b1={nn.b1.shape}, b2={nn.b2.shape}")
    
    # 3. 前向传播演示
    print("\n3. 前向传播演示...")
    y_pred_initial = nn.forward(X)
    initial_loss = nn.compute_loss(y_pred_initial, y)
    print(f"   初始预测:\n{y_pred_initial}")
    print(f"   初始损失: {initial_loss:.6f}")
    
    # 可视化神经网络结构
    visualizer.visualize_neural_network(nn, X, y)
    
    # 4. 单步反向传播演示
    print("\n4. 单步反向传播演示...")
    gradients = nn.backward(X, y, y_pred_initial)
    print(f"   梯度计算完成:")
    print(f"   dW1形状: {gradients['dW1'].shape}, 范数: {np.linalg.norm(gradients['dW1']):.6f}")
    print(f"   dW2形状: {gradients['dW2'].shape}, 范数: {np.linalg.norm(gradients['dW2']):.6f}")
    
    # 可视化梯度
    visualizer.visualize_backpropagation_step(nn, gradients)
    
    # 5. 训练神经网络
    print("\n5. 训练神经网络 (1000次迭代)...")
    history = nn.train(X, y, learning_rate=0.5, n_iterations=1000, verbose=True)
    
    # 6. 训练后评估
    print("\n6. 训练后评估...")
    y_pred_final = nn.forward(X)
    final_loss = nn.compute_loss(y_pred_final, y)
    print(f"   最终预测:\n{y_pred_final}")
    print(f"   最终损失: {final_loss:.6f}")
    print(f"   损失减少: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
    
    # 可视化训练历史
    visualizer.visualize_training_history(history)
    
    # 7. 链式法则演示
    print("\n" + "=" * 60)
    print("📚 反向传播链式法则总结")
    print("=" * 60)
    print("1. 前向传播: 计算网络输出和损失")
    print("2. 反向传播: 从输出层向输入层传播误差")
    print("3. 链式法则: dL/dW = dL/dA * dA/dZ * dZ/dW")
    print("4. 权重更新: W = W - η * dL/dW")
    print("5. 偏置更新: b = b - η * dL/db")
    print("\n关键公式:")
    print("  • 输出层梯度: dZ2 = (A2 - Y) * σ'(Z2)")
    print("  • 隐藏层梯度: dZ1 = (dZ2 · W2^T) * σ'(Z1)")
    print("  • 权重梯度: dW2 = A1^T · dZ2, dW1 = X^T · dZ1")
    print("  • 偏置梯度: db2 = Σ dZ2, db1 = Σ dZ1")
    
    # 8. 保存结果
    print("\n7. 保存结果...")
    results = {
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'loss_reduction_percent': float((initial_loss - final_loss) / initial_loss * 100),
        'final_predictions': y_pred_final.tolist(),
        'network_structure': {
            'input_size': nn.input_size,
            'hidden_size': nn.hidden_size,
            'output_size': nn.output_size
        }
    }
    
    with open('results/backprop_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   结果已保存到: results/backprop_results.json")
    
    return nn, history

if __name__ == "__main__":
    # 确保结果目录存在
    import os
    os.makedirs("results", exist_ok=True)
    
    nn, history = main()