"""
Chapter 5: Neural Networks (神经网络)
=====================================

本章介绍前馈神经网络，这是深度学习的基础。

主要内容：
1. 前馈网络函数 (5.1)
   - 网络架构
   - 激活函数
   - 权重初始化

2. 网络训练 (5.2)
   - 参数优化
   - 梯度下降

3. 误差反向传播 (5.3)
   - 链式法则
   - 高效梯度计算
   - 梯度检查

4. Hessian矩阵 (5.4)
   - 二阶优化
   - 快速乘积算法

5. 正则化 (5.5)
   - 早停
   - 权重衰减
   - Dropout

6. 混合密度网络 (5.6)
   - 多模态输出

7. 贝叶斯神经网络 (5.7)
   - 参数不确定性

核心思想：
神经网络是通用函数近似器。
通过组合简单的非线性单元，可以近似任意复杂的函数。

反向传播使得训练深层网络成为可能。
这个算法的重新发现是深度学习革命的开端。

本章为理解现代深度学习奠定基础。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .feedforward_network import (
    Activation,
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
    Layer,
    FeedForwardNetwork,
    visualize_activation_functions,
    demonstrate_forward_propagation,
    compare_weight_initialization
)

from .backpropagation import (
    Loss,
    MSELoss,
    CrossEntropyLoss,
    NeuralNetworkTrainer,
    gradient_check,
    visualize_gradients,
    demonstrate_backpropagation,
    compare_optimization_methods
)


def run_chapter05(cfg: DictConfig) -> None:
    """
    运行第5章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第5章：神经网络 (Neural Networks)")
    print("="*80)
    
    # 5.1 前馈网络函数
    print("\n" + "-"*60)
    print("5.1 前馈网络函数 (Feed-forward Network Functions)")
    print("-"*60)
    
    # 激活函数比较
    visualize_activation_functions(
        show_plot=cfg.visualization.show_plots
    )
    
    # 前向传播演示
    demonstrate_forward_propagation(
        show_plot=cfg.visualization.show_plots
    )
    
    # 权重初始化比较
    compare_weight_initialization(
        n_trials=100,
        show_plot=cfg.visualization.show_plots
    )
    
    # 5.2-5.3 网络训练和反向传播
    print("\n" + "-"*60)
    print("5.2-5.3 网络训练与误差反向传播")
    print("-"*60)
    
    # 反向传播演示
    demonstrate_backpropagation(
        show_plot=cfg.visualization.show_plots
    )
    
    # 梯度检查
    print("\n执行梯度检查...")
    
    # 创建小网络进行梯度检查
    network = FeedForwardNetwork(
        layer_dims=[2, 3, 2],
        activations=['tanh', 'sigmoid'],
        weight_init='xavier'
    )
    
    # 生成测试数据
    X_test = np.random.randn(5, 2)
    y_test = np.random.randn(5, 2)
    
    # 执行梯度检查
    gradient_check(
        network=network,
        X=X_test,
        y=y_test,
        loss_fn=MSELoss(),
        epsilon=1e-7
    )
    
    # 可视化梯度流
    visualize_gradients(
        network=network,
        X=X_test,
        y=y_test,
        loss_fn=MSELoss(),
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较优化方法
    compare_optimization_methods(
        show_plot=cfg.visualization.show_plots
    )
    
    # 5.5 正则化演示
    print("\n" + "-"*60)
    print("5.5 正则化在神经网络中的应用")
    print("-"*60)
    
    # 演示过拟合
    demonstrate_overfitting(
        show_plot=cfg.visualization.show_plots
    )
    
    # 演示正则化效果
    demonstrate_regularization(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第5章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 神经网络是通用函数近似器")
    print("2. 激活函数引入非线性")
    print("3. 反向传播高效计算梯度")
    print("4. 正确的初始化很重要")
    print("5. 学习率选择影响训练")
    print("6. 正则化防止过拟合")


def demonstrate_overfitting(show_plot: bool = True) -> None:
    """
    演示过拟合现象
    
    使用大网络和小数据集展示过拟合。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n过拟合演示")
    print("=" * 60)
    
    # 生成小数据集
    np.random.seed(42)
    n_train = 20
    n_test = 100
    
    # 训练数据
    X_train = np.random.uniform(-1, 1, (n_train, 1))
    y_train = np.sin(3 * np.pi * X_train) + 0.2 * np.random.randn(n_train, 1)
    
    # 测试数据
    X_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
    y_test = np.sin(3 * np.pi * X_test)
    
    # 创建大网络（容易过拟合）
    network_large = FeedForwardNetwork(
        layer_dims=[1, 50, 50, 1],
        activations=['tanh', 'tanh', 'linear'],
        weight_init='xavier'
    )
    
    # 创建小网络（不容易过拟合）
    network_small = FeedForwardNetwork(
        layer_dims=[1, 5, 1],
        activations=['tanh', 'linear'],
        weight_init='xavier'
    )
    
    # 训练大网络
    trainer_large = NeuralNetworkTrainer(
        network=network_large,
        loss_fn=MSELoss(),
        learning_rate=0.01
    )
    
    history_large = trainer_large.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=500,
        batch_size=20,
        verbose=False
    )
    
    # 训练小网络
    trainer_small = NeuralNetworkTrainer(
        network=network_small,
        loss_fn=MSELoss(),
        learning_rate=0.01
    )
    
    history_small = trainer_small.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=500,
        batch_size=20,
        verbose=False
    )
    
    print(f"\n大网络（100个隐藏单元）：")
    print(f"  训练损失: {history_large['loss_history'][-1]:.4f}")
    print(f"  测试损失: {history_large['val_loss_history'][-1]:.4f}")
    
    print(f"\n小网络（5个隐藏单元）：")
    print(f"  训练损失: {history_small['loss_history'][-1]:.4f}")
    print(f"  测试损失: {history_small['val_loss_history'][-1]:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 大网络拟合
        ax1 = axes[0, 0]
        y_pred_large = network_large.forward(X_test)
        ax1.plot(X_test, y_test, 'g-', label='真实函数', linewidth=2)
        ax1.plot(X_test, y_pred_large, 'r-', label='大网络预测', linewidth=2)
        ax1.scatter(X_train, y_train, c='blue', s=50, zorder=5, label='训练数据')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('大网络（过拟合）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 小网络拟合
        ax2 = axes[0, 1]
        y_pred_small = network_small.forward(X_test)
        ax2.plot(X_test, y_test, 'g-', label='真实函数', linewidth=2)
        ax2.plot(X_test, y_pred_small, 'r-', label='小网络预测', linewidth=2)
        ax2.scatter(X_train, y_train, c='blue', s=50, zorder=5, label='训练数据')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('小网络（适度拟合）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 大网络损失曲线
        ax3 = axes[1, 0]
        ax3.plot(history_large['loss_history'], label='训练损失', linewidth=2)
        ax3.plot(history_large['val_loss_history'], label='验证损失', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('大网络损失曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 小网络损失曲线
        ax4 = axes[1, 1]
        ax4.plot(history_small['loss_history'], label='训练损失', linewidth=2)
        ax4.plot(history_small['val_loss_history'], label='验证损失', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('小网络损失曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.suptitle('过拟合现象', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 大网络在训练数据上表现很好，但泛化差")
    print("2. 小网络虽然训练误差大，但泛化好")
    print("3. 验证损失的上升是过拟合的标志")
    print("4. 模型容量应该与数据量匹配")


def demonstrate_regularization(show_plot: bool = True) -> None:
    """
    演示正则化效果
    
    展示L2正则化如何减少过拟合。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n正则化效果演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n_train = 30
    
    X_train = np.random.uniform(-1, 1, (n_train, 1))
    y_train = np.sin(3 * np.pi * X_train) + 0.2 * np.random.randn(n_train, 1)
    
    X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_test = np.sin(3 * np.pi * X_test)
    
    # 不同的正则化强度
    reg_strengths = [0, 0.001, 0.01, 0.1]
    
    results = {}
    
    for reg in reg_strengths:
        print(f"\n正则化系数 λ={reg}:")
        
        # 创建网络
        network = FeedForwardNetwork(
            layer_dims=[1, 30, 30, 1],
            activations=['tanh', 'tanh', 'linear'],
            weight_init='xavier'
        )
        
        # 训练
        trainer = NeuralNetworkTrainer(
            network=network,
            loss_fn=MSELoss(),
            learning_rate=0.01,
            regularization=reg
        )
        
        history = trainer.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=300,
            batch_size=30,
            verbose=False
        )
        
        # 预测
        y_pred = network.forward(X_test)
        
        results[reg] = {
            'network': network,
            'history': history,
            'y_pred': y_pred
        }
        
        print(f"  训练损失: {history['loss_history'][-1]:.4f}")
        print(f"  测试损失: {history['val_loss_history'][-1]:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, len(reg_strengths), 
                                figsize=(4*len(reg_strengths), 8))
        
        for idx, reg in enumerate(reg_strengths):
            result = results[reg]
            
            # 拟合曲线
            ax1 = axes[0, idx]
            ax1.plot(X_test, y_test, 'g-', label='真实', linewidth=2)
            ax1.plot(X_test, result['y_pred'], 'r-', 
                    label='预测', linewidth=2)
            ax1.scatter(X_train, y_train, c='blue', s=30, alpha=0.5)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'λ={reg}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([-2, 2])
            
            # 损失曲线
            ax2 = axes[1, idx]
            ax2.plot(result['history']['loss_history'], 
                    label='训练', linewidth=2)
            ax2.plot(result['history']['val_loss_history'], 
                    label='验证', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'损失曲线 (λ={reg})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.suptitle('L2正则化效果', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. λ=0：无正则化，过拟合严重")
    print("2. λ适中：平衡拟合和泛化")
    print("3. λ过大：欠拟合，函数过于平滑")
    print("4. 正则化通过限制权重大小防止过拟合")