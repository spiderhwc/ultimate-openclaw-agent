#!/usr/bin/env python3
"""
终极全能智能体算法测试脚本
简化版本，避免网络依赖
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_linear_regression():
    """测试线性回归算法"""
    print("🧪 测试线性回归算法...")
    
    try:
        # 导入线性回归模块
        from src.algorithms.linear_regression_stock import generate_stock_data
        
        # 生成测试数据
        stock_data = generate_stock_data(days=10, stock_name="测试股票")
        
        print(f"✅ 数据生成成功:")
        print(f"   股票名称: {stock_data['stock_name']}")
        print(f"   数据天数: {stock_data['days']}")
        print(f"   价格数量: {len(stock_data['prices'])}")
        print(f"   波动率: {stock_data['volatility']:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ 线性回归测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logistic_regression():
    """测试逻辑回归算法"""
    print("🧪 测试逻辑回归算法...")
    
    try:
        # 导入逻辑回归模块
        from src.algorithms.logistic_regression_stock import generate_stock_features
        
        # 生成测试数据
        feature_names = ["动量", "成交量", "RSI", "MACD", "波动率"]
        data = generate_stock_features(n_samples=100, feature_names=feature_names)
        
        print(f"✅ 特征生成成功:")
        print(f"   样本数量: {data['n_samples']}")
        print(f"   特征数量: {data['features'].shape[1]}")
        print(f"   标签数量: {len(data['labels'])}")
        print(f"   上涨比例: {data['positive_ratio']:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ 逻辑回归测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_installation():
    """测试PyTorch安装"""
    print("🧪 测试PyTorch安装...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ NumPy版本: {np.__version__}")
        
        # 测试简单的张量操作
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        
        print(f"✅ 张量运算测试: {x} + {y} = {z}")
        
        # 测试简单的神经网络
        model = nn.Linear(3, 1)
        output = model(x.unsqueeze(0))
        
        print(f"✅ 神经网络测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment():
    """测试环境配置"""
    print("🧪 测试Python环境...")
    
    try:
        import sys
        import platform
        
        print(f"✅ Python版本: {sys.version}")
        print(f"✅ 操作系统: {platform.system()} {platform.release()}")
        print(f"✅ 工作目录: {os.getcwd()}")
        
        # 检查关键包
        required_packages = ['torch', 'numpy', 'matplotlib', 'sklearn']
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}: 已安装")
            except ImportError:
                print(f"❌ {package}: 未安装")
        
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 终极全能智能体算法测试")
    print("=" * 60)
    
    test_results = {}
    
    # 运行所有测试
    test_results['environment'] = test_environment()
    test_results['pytorch'] = test_pytorch_installation()
    test_results['linear_regression'] = test_linear_regression()
    test_results['logistic_regression'] = test_logistic_regression()
    
    # 生成测试报告
    print("\n" + "=" * 60)
    print("📋 测试报告")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！终极全能智能体算法环境就绪！")
        return True
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)