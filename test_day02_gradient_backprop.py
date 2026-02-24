#!/usr/bin/env python3
"""
第2天：梯度下降与反向传播算法测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gradient_descent():
    """测试梯度下降算法"""
    print("🧪 测试梯度下降算法...")
    
    try:
        from src.algorithms.day02_gradient_backprop.gradient_descent_demo import GradientDescentDemo
        
        demo = GradientDescentDemo()
        X, y = demo.generate_data(n_samples=50)
        
        # 测试梯度计算
        w, b = 0.0, 0.0
        dw, db = demo.compute_gradients(w, b, X, y)
        
        print(f"✅ 梯度计算测试通过")
        print(f"   初始梯度: dw={dw:.4f}, db={db:.4f}")
        
        # 测试梯度下降
        w_final, b_final, history = demo.gradient_descent(
            X, y, learning_rate=0.01, n_iterations=50
        )
        
        print(f"✅ 梯度下降测试通过")
        print(f"   最终参数: w={w_final:.4f}, b={b_final:.4f}")
        print(f"   最终损失: {history[-1]['loss']:.4f}")
        print(f"   迭代次数: {len(history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度下降测试失败: {e}")
        return False

def test_backpropagation():
    """测试反向传播算法"""
    print("\n🧪 测试反向传播算法...")
    
    try:
        from src.algorithms.day02_gradient_backprop.backpropagation_demo import SimpleNeuralNetwork
        
        # 创建神经网络
        nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
        
        # 生成测试数据
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # 测试前向传播
        y_pred = nn.forward(X)
        loss = nn.compute_loss(y_pred, y)
        
        print(f"✅ 前向传播测试通过")
        print(f"   预测形状: {y_pred.shape}")
        print(f"   初始损失: {loss:.6f}")
        
        # 测试反向传播
        gradients = nn.backward(X, y, y_pred)
        
        print(f"✅ 反向传播测试通过")
        print(f"   梯度形状: dW1={gradients['dW1'].shape}, dW2={gradients['dW2'].shape}")
        
        # 测试参数更新
        nn.update_parameters(gradients, learning_rate=0.1)
        
        print(f"✅ 参数更新测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 反向传播测试失败: {e}")
        return False

def test_visualizations():
    """测试可视化功能"""
    print("\n🧪 测试可视化功能...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        # 检查matplotlib是否可用
        import matplotlib.pyplot as plt
        
        print(f"✅ Matplotlib版本: {matplotlib.__version__}")
        print(f"✅ 可视化后端: {matplotlib.get_backend()}")
        
        # 创建简单的测试图
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 9], 'b-', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('测试图')
        ax.grid(True, alpha=0.3)
        
        # 保存测试图
        test_path = 'results/test_visualization.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(test_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_path):
            print(f"✅ 可视化保存测试通过")
            print(f"   测试图已保存到: {test_path}")
            return True
        else:
            print(f"❌ 可视化保存测试失败: 文件未创建")
            return False
            
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def test_pytorch_gradient():
    """测试PyTorch自动梯度"""
    print("\n🧪 测试PyTorch自动梯度...")
    
    try:
        import torch
        
        # 创建需要梯度的张量
        x = torch.tensor(2.0, requires_grad=True)
        w = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        
        # 前向传播
        y = w * x + b
        
        # 计算损失
        target = torch.tensor(10.0)
        loss = (y - target) ** 2
        
        # 反向传播
        loss.backward()
        
        print(f"✅ PyTorch自动梯度测试通过")
        print(f"   计算图: y = {w.item()} * {x.item()} + {b.item()} = {y.item()}")
        print(f"   损失: {loss.item():.4f}")
        print(f"   梯度: dw/dx = {x.grad.item():.4f}, dloss/dw = {w.grad.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch自动梯度测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 第2天：梯度下降与反向传播算法测试")
    print("=" * 60)
    
    # 导入numpy
    global np
    import numpy as np
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行所有测试
    tests = [
        ("梯度下降算法", test_gradient_descent),
        ("反向传播算法", test_backpropagation),
        ("可视化功能", test_visualizations),
        ("PyTorch自动梯度", test_pytorch_gradient)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 开始测试: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 测试完成: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！第2天算法学习环境就绪！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)