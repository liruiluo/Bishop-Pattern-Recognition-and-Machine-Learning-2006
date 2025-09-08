"""
5.1 前馈网络函数 (Feed-forward Network Functions)
=================================================

神经网络是一系列的函数组合：
y = f_L(...f_2(f_1(x)))

每层的计算：
a^(l) = W^(l) z^(l-1) + b^(l)  # 线性组合
z^(l) = h(a^(l))               # 激活函数

其中：
- W^(l)：第l层的权重矩阵
- b^(l)：第l层的偏置向量
- h：激活函数
- z^(l)：第l层的激活值

关键概念：
1. 万能近似定理：足够宽的单隐层网络可以近似任意连续函数
2. 深度的作用：多层网络可以更高效地表示复杂函数
3. 激活函数：引入非线性，否则多层等价于单层

激活函数选择：
- Sigmoid：σ(x) = 1/(1+e^(-x))，输出(0,1)，有梯度消失问题
- Tanh：tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))，输出(-1,1)，零中心
- ReLU：max(0,x)，计算简单，缓解梯度消失，但有死神经元问题

权重初始化：
- 随机初始化：打破对称性
- Xavier初始化：保持方差，适合sigmoid/tanh
- He初始化：适合ReLU
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


class Activation:
    """
    激活函数基类
    
    每个激活函数需要实现前向传播和反向传播。
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        raise NotImplementedError
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        反向传播：计算导数
        
        注意：x是激活前的值（不是激活后的）
        """
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Sigmoid(Activation):
    """
    Sigmoid激活函数
    
    σ(x) = 1/(1 + e^(-x))
    σ'(x) = σ(x)(1 - σ(x))
    
    优点：
    - 输出范围(0,1)，可解释为概率
    - 平滑可导
    
    缺点：
    - 梯度消失：当|x|很大时，梯度接近0
    - 非零中心：输出总是正的
    - 计算exp较慢
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 避免溢出
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    """
    Tanh激活函数
    
    tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
    tanh'(x) = 1 - tanh²(x)
    
    优点：
    - 零中心：输出范围(-1,1)
    - 比sigmoid收敛快
    
    缺点：
    - 仍有梯度消失问题
    - 计算exp较慢
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1 - t**2


class ReLU(Activation):
    """
    ReLU激活函数
    
    ReLU(x) = max(0, x)
    ReLU'(x) = 1 if x > 0 else 0
    
    优点：
    - 计算简单快速
    - 缓解梯度消失
    - 稀疏激活
    
    缺点：
    - 死神经元：负区域梯度为0
    - 非零中心
    - 无界
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Linear(Activation):
    """
    线性激活（恒等函数）
    
    f(x) = x
    f'(x) = 1
    
    用于输出层（回归任务）
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Layer:
    """
    神经网络层
    
    实现一层的前向和反向传播。
    包括线性变换和激活函数。
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 activation: Optional[Activation] = None,
                 weight_init: str = 'xavier'):
        """
        初始化层
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            activation: 激活函数
            weight_init: 权重初始化方法
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation or Linear()
        
        # 初始化权重和偏置
        self.W = self._init_weights(weight_init)
        self.b = np.zeros(output_dim)
        
        # 梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # 缓存前向传播的值（用于反向传播）
        self.cache = {}
    
    def _init_weights(self, method: str) -> np.ndarray:
        """
        初始化权重
        
        不同的初始化方法适合不同的激活函数。
        
        Args:
            method: 初始化方法
            
        Returns:
            初始化的权重矩阵
        """
        if method == 'random':
            # 简单随机初始化
            return np.random.randn(self.input_dim, self.output_dim) * 0.01
        
        elif method == 'xavier':
            # Xavier初始化：保持方差
            # 适合sigmoid和tanh
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            return np.random.randn(self.input_dim, self.output_dim) * std
        
        elif method == 'he':
            # He初始化：适合ReLU
            std = np.sqrt(2.0 / self.input_dim)
            return np.random.randn(self.input_dim, self.output_dim) * std
        
        else:
            raise ValueError(f"未知初始化方法: {method}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        z = h(Wx + b)
        
        Args:
            X: 输入，shape (batch_size, input_dim)
            
        Returns:
            激活后的输出，shape (batch_size, output_dim)
        """
        # 线性变换
        self.cache['X'] = X
        self.cache['Z'] = X @ self.W + self.b  # (batch, output)
        
        # 激活函数
        self.cache['A'] = self.activation.forward(self.cache['Z'])
        
        return self.cache['A']
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        计算梯度并传递误差。
        
        Args:
            dA: 上游梯度，shape (batch_size, output_dim)
            
        Returns:
            传递给下游的梯度，shape (batch_size, input_dim)
        """
        batch_size = dA.shape[0]
        
        # 通过激活函数反向传播
        dZ = dA * self.activation.backward(self.cache['Z'])
        
        # 计算权重和偏置的梯度
        self.dW = self.cache['X'].T @ dZ / batch_size
        self.db = np.sum(dZ, axis=0) / batch_size
        
        # 传递给下一层的梯度
        dX = dZ @ self.W.T
        
        return dX
    
    def update_weights(self, learning_rate: float) -> None:
        """
        更新权重（梯度下降）
        
        Args:
            learning_rate: 学习率
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class FeedForwardNetwork:
    """
    前馈神经网络
    
    多层感知器（MLP）的实现。
    支持任意深度和宽度。
    """
    
    def __init__(self, layer_dims: List[int],
                 activations: Optional[List[str]] = None,
                 weight_init: str = 'xavier'):
        """
        初始化网络
        
        Args:
            layer_dims: 各层维度 [input_dim, hidden1, hidden2, ..., output_dim]
            activations: 各层激活函数名称
            weight_init: 权重初始化方法
        """
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        
        # 默认激活函数
        if activations is None:
            # 隐层用ReLU，输出层用线性
            activations = ['relu'] * (self.n_layers - 1) + ['linear']
        
        # 创建激活函数对象
        activation_map = {
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'relu': ReLU,
            'linear': Linear
        }
        
        # 构建网络层
        self.layers = []
        for i in range(self.n_layers):
            activation = activation_map[activations[i]]()
            layer = Layer(layer_dims[i], layer_dims[i+1], 
                         activation, weight_init)
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        逐层计算，直到输出。
        
        Args:
            X: 输入数据，shape (batch_size, input_dim)
            
        Returns:
            网络输出，shape (batch_size, output_dim)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, dY: np.ndarray) -> None:
        """
        反向传播
        
        从输出层开始，逐层计算梯度。
        
        Args:
            dY: 输出层的梯度，shape (batch_size, output_dim)
        """
        dA = dY
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def update_weights(self, learning_rate: float) -> None:
        """
        更新所有层的权重
        
        Args:
            learning_rate: 学习率
        """
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（前向传播的别名）
        
        Args:
            X: 输入数据
            
        Returns:
            预测输出
        """
        return self.forward(X)
    
    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        获取所有层的权重和偏置
        
        Returns:
            [(W1, b1), (W2, b2), ...]
        """
        return [(layer.W, layer.b) for layer in self.layers]
    
    def set_weights(self, weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        设置所有层的权重和偏置
        
        Args:
            weights: [(W1, b1), (W2, b2), ...]
        """
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W
            layer.b = b


def visualize_activation_functions(show_plot: bool = True) -> None:
    """
    可视化不同的激活函数
    
    展示激活函数及其导数。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n激活函数比较")
    print("=" * 60)
    
    x = np.linspace(-5, 5, 200)
    
    activations = {
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ReLU': ReLU()
    }
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for idx, (name, activation) in enumerate(activations.items()):
            # 激活函数
            ax1 = axes[0, idx]
            y = activation.forward(x)
            ax1.plot(x, y, linewidth=2)
            ax1.set_title(f'{name}激活函数')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            ax1.axvline(x=0, color='k', linewidth=0.5)
            
            # 导数
            ax2 = axes[1, idx]
            dy = activation.backward(x)
            ax2.plot(x, dy, linewidth=2, color='orange')
            ax2.set_title(f'{name}导数')
            ax2.set_xlabel('x')
            ax2.set_ylabel("f'(x)")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            ax2.axvline(x=0, color='k', linewidth=0.5)
            
            # 打印特性
            print(f"\n{name}:")
            print(f"  输出范围: [{y.min():.2f}, {y.max():.2f}]")
            print(f"  f(0) = {activation.forward(0):.3f}")
            print(f"  f'(0) = {activation.backward(0):.3f}")
        
        plt.suptitle('激活函数及其导数', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n特性总结：")
    print("1. Sigmoid：平滑但有梯度消失")
    print("2. Tanh：零中心，比Sigmoid好")
    print("3. ReLU：简单高效，但有死神经元")


def demonstrate_forward_propagation(show_plot: bool = True) -> None:
    """
    演示前向传播
    
    展示数据如何通过网络流动。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n前向传播演示")
    print("=" * 60)
    
    # 创建简单网络：2 -> 3 -> 1
    network = FeedForwardNetwork(
        layer_dims=[2, 3, 1],
        activations=['tanh', 'sigmoid'],
        weight_init='xavier'
    )
    
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(5, 2)  # 5个样本，2个特征
    
    print("网络结构: 2 -> 3 -> 1")
    print(f"输入形状: {X.shape}")
    
    # 前向传播
    y = network.forward(X)
    print(f"输出形状: {y.shape}")
    
    # 显示各层激活值
    print("\n各层激活值：")
    A = X
    print(f"输入层: shape={A.shape}")
    for i, layer in enumerate(network.layers):
        A = layer.forward(A)
        print(f"第{i+1}层输出: shape={A.shape}, "
              f"范围=[{A.min():.3f}, {A.max():.3f}]")
    
    if show_plot:
        # 可视化网络响应
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 创建网格
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        
        # 计算网络输出
        Z = network.forward(X_grid)
        Z = Z.reshape(xx.shape)
        
        # 左图：网络输出
        ax1 = axes[0]
        contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax1)
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title('网络输出')
        
        # 右图：等高线
        ax2 = axes[1]
        contour2 = ax2.contour(xx, yy, Z, levels=10, colors='black', alpha=0.5)
        ax2.clabel(contour2, inline=True, fontsize=8)
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title('输出等高线')
        
        # 显示输入点
        ax2.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                   cmap='RdBu_r', s=100, edgecolor='black')
        
        plt.suptitle('前向传播可视化', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 数据逐层变换")
    print("2. 激活函数引入非线性")
    print("3. 网络可以学习复杂的决策边界")


def compare_weight_initialization(n_trials: int = 100,
                                 show_plot: bool = True) -> None:
    """
    比较不同的权重初始化方法
    
    展示初始化对激活值分布的影响。
    
    Args:
        n_trials: 试验次数
        show_plot: 是否显示图形
    """
    print("\n权重初始化比较")
    print("=" * 60)
    
    # 网络配置
    layer_dims = [100, 50, 50, 10]
    
    # 生成输入数据
    X = np.random.randn(1000, 100)
    
    methods = ['random', 'xavier', 'he']
    activations = ['sigmoid', 'tanh', 'relu']
    
    results = {}
    
    for method in methods:
        for activation in activations:
            # 创建网络
            network = FeedForwardNetwork(
                layer_dims=layer_dims,
                activations=[activation] * 3,
                weight_init=method
            )
            
            # 前向传播
            A = X
            variances = [np.var(A)]
            
            for layer in network.layers:
                A = layer.forward(A)
                variances.append(np.var(A))
            
            results[(method, activation)] = variances
            
            print(f"\n{method} + {activation}:")
            print(f"  输入方差: {variances[0]:.3f}")
            print(f"  最后层方差: {variances[-1]:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        
        for i, activation in enumerate(activations):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                variances = results[(method, activation)]
                layers = list(range(len(variances)))
                
                ax.plot(layers, variances, 'o-', linewidth=2)
                ax.set_xlabel('层')
                ax.set_ylabel('激活值方差')
                ax.set_title(f'{method} + {activation}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, max(2, max(variances) * 1.1)])
                
                # 标记理想范围
                ax.axhspan(0.5, 1.5, alpha=0.2, color='green')
        
        plt.suptitle('权重初始化对激活值方差的影响', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 随机初始化：方差快速衰减或爆炸")
    print("2. Xavier：适合sigmoid/tanh，保持方差")
    print("3. He：适合ReLU，考虑了ReLU的特性")
    print("4. 正确的初始化避免梯度消失/爆炸")