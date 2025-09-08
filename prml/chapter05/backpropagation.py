"""
5.2-5.3 误差反向传播 (Error Backpropagation)
============================================

反向传播是训练神经网络的核心算法。
它高效地计算损失函数对所有权重的梯度。

核心思想：
链式法则 + 动态规划

对于复合函数 f(g(h(x)))：
df/dx = df/dg · dg/dh · dh/dx

反向传播的步骤：
1. 前向传播：计算所有激活值
2. 计算输出层误差
3. 反向传播误差
4. 计算梯度
5. 更新权重

数学推导：
定义误差信号 δ^(l) = ∂E/∂a^(l)

输出层：δ^(L) = ∂E/∂y · h'(a^(L))
隐藏层：δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ h'(a^(l))

权重梯度：∂E/∂W^(l) = δ^(l) (z^(l-1))^T
偏置梯度：∂E/∂b^(l) = δ^(l)

计算复杂度：
前向：O(W)，其中W是权重总数
反向：O(W)，与前向相同！

这就是反向传播的魔力：
梯度计算只需要前向传播2倍的时间。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable
from .feedforward_network import FeedForwardNetwork, Layer
import warnings
warnings.filterwarnings('ignore')


class Loss:
    """
    损失函数基类
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算损失值"""
        raise NotImplementedError
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算损失对预测的梯度"""
        raise NotImplementedError


class MSELoss(Loss):
    """
    均方误差损失
    
    L = (1/2) Σ(y_true - y_pred)²
    
    适用于回归任务。
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算MSE损失
        
        Args:
            y_true: 真实值，shape (batch_size, output_dim)
            y_pred: 预测值，shape (batch_size, output_dim)
            
        Returns:
            损失值
        """
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        MSE损失的梯度
        
        ∂L/∂y = y_pred - y_true
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            梯度
        """
        return (y_pred - y_true) / len(y_true)


class CrossEntropyLoss(Loss):
    """
    交叉熵损失
    
    L = -Σ y_true * log(y_pred)
    
    适用于分类任务。
    假设y_pred已经过softmax。
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算交叉熵
        
        Args:
            y_true: 真实标签（one-hot），shape (batch_size, n_classes)
            y_pred: 预测概率，shape (batch_size, n_classes)
            
        Returns:
            损失值
        """
        # 避免log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        交叉熵的梯度
        
        如果配合softmax，梯度简化为：
        ∂L/∂z = y_pred - y_true
        
        Args:
            y_true: 真实标签
            y_pred: 预测概率
            
        Returns:
            梯度
        """
        # 这里假设y_pred是softmax输出
        # 组合softmax和交叉熵的梯度
        return (y_pred - y_true) / len(y_true)


class NeuralNetworkTrainer:
    """
    神经网络训练器
    
    实现完整的训练流程：
    1. 前向传播
    2. 计算损失
    3. 反向传播
    4. 更新权重
    """
    
    def __init__(self, network: FeedForwardNetwork,
                 loss_fn: Loss,
                 learning_rate: float = 0.01,
                 regularization: float = 0.0):
        """
        初始化训练器
        
        Args:
            network: 神经网络
            loss_fn: 损失函数
            learning_rate: 学习率
            regularization: L2正则化系数
        """
        self.network = network
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # 训练历史
        self.loss_history = []
        self.val_loss_history = []
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        单步训练
        
        完整的前向-反向传播过程。
        
        Args:
            X: 输入，shape (batch_size, input_dim)
            y: 标签，shape (batch_size, output_dim)
            
        Returns:
            损失值
        """
        # 前向传播
        y_pred = self.network.forward(X)
        
        # 计算损失
        loss = self.loss_fn.forward(y, y_pred)
        
        # 添加L2正则化
        if self.regularization > 0:
            reg_loss = 0
            for layer in self.network.layers:
                reg_loss += 0.5 * self.regularization * np.sum(layer.W ** 2)
            loss += reg_loss
        
        # 反向传播
        dY = self.loss_fn.backward(y, y_pred)
        self.network.backward(dY)
        
        # 添加正则化梯度
        if self.regularization > 0:
            for layer in self.network.layers:
                layer.dW += self.regularization * layer.W
        
        # 更新权重
        self.network.update_weights(self.learning_rate)
        
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32,
             verbose: bool = True) -> Dict:
        """
        训练网络
        
        使用小批量梯度下降。
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印进度
            
        Returns:
            训练历史
        """
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.loss_history = []
        self.val_loss_history = []
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # 小批量训练
            epoch_loss = 0
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
            
            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)
            
            # 验证
            if X_val is not None:
                y_val_pred = self.network.forward(X_val)
                val_loss = self.loss_fn.forward(y_val, y_val_pred)
                self.val_loss_history.append(val_loss)
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
        
        return {
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history
        }


def gradient_check(network: FeedForwardNetwork,
                  X: np.ndarray, y: np.ndarray,
                  loss_fn: Loss,
                  epsilon: float = 1e-7) -> Dict:
    """
    梯度检查
    
    使用数值梯度验证反向传播的正确性。
    
    数值梯度：
    ∂f/∂θ ≈ [f(θ + ε) - f(θ - ε)] / (2ε)
    
    Args:
        network: 神经网络
        X: 输入数据
        y: 标签
        loss_fn: 损失函数
        epsilon: 扰动大小
        
    Returns:
        检查结果
    """
    print("\n梯度检查")
    print("=" * 60)
    
    # 前向传播和反向传播
    y_pred = network.forward(X)
    loss = loss_fn.forward(y, y_pred)
    dY = loss_fn.backward(y, y_pred)
    network.backward(dY)
    
    results = {}
    
    # 检查每层的梯度
    for layer_idx, layer in enumerate(network.layers):
        print(f"\n第{layer_idx + 1}层:")
        
        # 检查权重梯度
        W_shape = layer.W.shape
        n_checks = min(10, W_shape[0] * W_shape[1])
        
        # 随机选择要检查的权重
        indices = np.random.choice(W_shape[0] * W_shape[1], n_checks, replace=False)
        
        analytical_grads = []
        numerical_grads = []
        
        for idx in indices:
            i, j = idx // W_shape[1], idx % W_shape[1]
            
            # 解析梯度（反向传播）
            analytical_grad = layer.dW[i, j]
            
            # 数值梯度
            W_orig = layer.W[i, j]
            
            # f(θ + ε)
            layer.W[i, j] = W_orig + epsilon
            y_plus = network.forward(X)
            loss_plus = loss_fn.forward(y, y_plus)
            
            # f(θ - ε)
            layer.W[i, j] = W_orig - epsilon
            y_minus = network.forward(X)
            loss_minus = loss_fn.forward(y, y_minus)
            
            # 恢复原值
            layer.W[i, j] = W_orig
            
            # 数值梯度
            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            
            analytical_grads.append(analytical_grad)
            numerical_grads.append(numerical_grad)
        
        analytical_grads = np.array(analytical_grads)
        numerical_grads = np.array(numerical_grads)
        
        # 计算相对误差
        diff = np.abs(analytical_grads - numerical_grads)
        norm_sum = np.abs(analytical_grads) + np.abs(numerical_grads)
        relative_error = np.mean(diff / (norm_sum + 1e-10))
        
        results[f'layer_{layer_idx}'] = {
            'analytical': analytical_grads,
            'numerical': numerical_grads,
            'relative_error': relative_error
        }
        
        print(f"  相对误差: {relative_error:.2e}")
        
        if relative_error < 1e-5:
            print("  ✓ 梯度正确")
        elif relative_error < 1e-3:
            print("  ⚠ 梯度可能有小误差")
        else:
            print("  ✗ 梯度错误！")
    
    return results


def visualize_gradients(network: FeedForwardNetwork,
                       X: np.ndarray, y: np.ndarray,
                       loss_fn: Loss,
                       show_plot: bool = True) -> None:
    """
    可视化梯度流
    
    展示梯度如何在网络中传播。
    
    Args:
        network: 神经网络
        X: 输入数据
        y: 标签
        loss_fn: 损失函数
        show_plot: 是否显示图形
    """
    print("\n梯度流可视化")
    print("=" * 60)
    
    # 前向和反向传播
    y_pred = network.forward(X)
    dY = loss_fn.backward(y, y_pred)
    network.backward(dY)
    
    # 收集梯度信息
    gradient_norms = []
    weight_norms = []
    
    for i, layer in enumerate(network.layers):
        grad_norm = np.linalg.norm(layer.dW)
        weight_norm = np.linalg.norm(layer.W)
        
        gradient_norms.append(grad_norm)
        weight_norms.append(weight_norm)
        
        print(f"第{i+1}层:")
        print(f"  权重范数: {weight_norm:.4f}")
        print(f"  梯度范数: {grad_norm:.4f}")
        print(f"  梯度/权重比: {grad_norm/weight_norm:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        layers = list(range(1, len(network.layers) + 1))
        
        # 梯度范数
        ax1 = axes[0]
        ax1.bar(layers, gradient_norms, color='blue', alpha=0.7)
        ax1.set_xlabel('层')
        ax1.set_ylabel('梯度范数')
        ax1.set_title('各层梯度范数')
        ax1.grid(True, alpha=0.3)
        
        # 权重范数
        ax2 = axes[1]
        ax2.bar(layers, weight_norms, color='green', alpha=0.7)
        ax2.set_xlabel('层')
        ax2.set_ylabel('权重范数')
        ax2.set_title('各层权重范数')
        ax2.grid(True, alpha=0.3)
        
        # 梯度/权重比
        ax3 = axes[2]
        ratios = [g/w for g, w in zip(gradient_norms, weight_norms)]
        ax3.bar(layers, ratios, color='red', alpha=0.7)
        ax3.set_xlabel('层')
        ax3.set_ylabel('梯度/权重')
        ax3.set_title('梯度权重比')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.01, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.001, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle('梯度流分析', fontsize=14)
        plt.tight_layout()
        plt.show()


def demonstrate_backpropagation(show_plot: bool = True) -> None:
    """
    演示反向传播
    
    在简单任务上展示训练过程。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n反向传播演示")
    print("=" * 60)
    
    # 生成XOR数据（非线性可分）
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("任务：学习XOR函数")
    print("这是经典的非线性问题，需要隐藏层。")
    
    # 创建网络
    network = FeedForwardNetwork(
        layer_dims=[2, 4, 1],
        activations=['tanh', 'sigmoid'],
        weight_init='xavier'
    )
    
    # 创建训练器
    trainer = NeuralNetworkTrainer(
        network=network,
        loss_fn=MSELoss(),
        learning_rate=0.5
    )
    
    # 训练
    print("\n开始训练...")
    history = trainer.train(
        X, y,
        epochs=1000,
        batch_size=4,
        verbose=False
    )
    
    # 最终预测
    y_pred = network.forward(X)
    
    print("\n训练结果：")
    print("输入 -> 目标 -> 预测")
    for i in range(len(X)):
        print(f"{X[i]} -> {y[i][0]:.0f} -> {y_pred[i][0]:.3f}")
    
    final_loss = history['loss_history'][-1]
    print(f"\n最终损失: {final_loss:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1 = axes[0]
        ax1.plot(history['loss_history'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 决策边界
        ax2 = axes[1]
        
        # 创建网格
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        
        # 预测
        Z = network.forward(X_grid)
        Z = Z.reshape(xx.shape)
        
        # 绘制
        contour = ax2.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(contour, ax=ax2)
        ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        # 数据点
        colors = ['blue' if yi[0] < 0.5 else 'red' for yi in y]
        ax2.scatter(X[:, 0], X[:, 1], c=colors, s=200, 
                   edgecolor='black', linewidth=2)
        
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title('学习的XOR函数')
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        
        plt.suptitle('反向传播训练XOR', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 网络成功学习了XOR函数")
    print("2. 隐藏层创建了非线性决策边界")
    print("3. 损失函数单调下降")


def compare_optimization_methods(show_plot: bool = True) -> None:
    """
    比较不同的优化方法
    
    展示学习率对训练的影响。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n优化方法比较")
    print("=" * 60)
    
    # 生成回归数据
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 1))
    y = np.sin(3 * X) + 0.1 * np.random.randn(100, 1)
    
    # 不同的学习率
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    histories = {}
    
    for lr in learning_rates:
        print(f"\n学习率 = {lr}:")
        
        # 创建新网络
        network = FeedForwardNetwork(
            layer_dims=[1, 10, 1],
            activations=['tanh', 'linear'],
            weight_init='xavier'
        )
        
        # 训练
        trainer = NeuralNetworkTrainer(
            network=network,
            loss_fn=MSELoss(),
            learning_rate=lr
        )
        
        try:
            history = trainer.train(
                X, y,
                epochs=100,
                batch_size=20,
                verbose=False
            )
            
            final_loss = history['loss_history'][-1]
            print(f"  最终损失: {final_loss:.4f}")
            
            if np.isnan(final_loss):
                print("  ✗ 训练发散！")
            else:
                print("  ✓ 训练收敛")
                
            histories[lr] = history['loss_history']
        except:
            print("  ✗ 训练失败！")
            histories[lr] = None
    
    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for lr, history in histories.items():
            if history is not None and not any(np.isnan(history)):
                ax.plot(history, label=f'LR={lr}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('不同学习率的训练曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 学习率太小：收敛慢")
    print("2. 学习率太大：可能发散")
    print("3. 合适的学习率：快速稳定收敛")
    print("4. 自适应学习率方法（如Adam）可以缓解这个问题")