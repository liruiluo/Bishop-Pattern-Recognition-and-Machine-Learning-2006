"""
4.3 概率判别模型 (Probabilistic Discriminative Models)
======================================================

判别模型直接建模后验概率p(C_k|x)，
而不是先建模p(x|C_k)和p(C_k)。

优点：
- 参数更少（只需要决策边界）
- 通常性能更好
- 不需要对输入分布建模

主要方法：
1. 逻辑回归（Logistic Regression）
2. 多类逻辑回归（Softmax回归）

逻辑回归：
p(C_1|x) = σ(w^T x) = 1/(1 + exp(-w^T x))

对数几率（log-odds）：
ln[p(C_1|x)/p(C_2|x)] = w^T x

这是线性的！所以叫"逻辑回归"。

训练方法：
最大似然估计，但没有闭式解。
使用迭代优化：
- 梯度下降
- 牛顿法
- IRLS（迭代重加权最小二乘）

交叉熵损失：
E(w) = -Σ_n [t_n ln y_n + (1-t_n) ln(1-y_n)]

其中y_n = σ(w^T x_n)

梯度：
∇E = Σ_n (y_n - t_n) x_n

这个形式非常优雅！误差×输入。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from scipy.special import expit, softmax  # sigmoid和softmax函数
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    """
    逻辑回归（二分类）
    
    模型：
    p(y=1|x,w) = σ(w^T x) = 1/(1 + exp(-w^T x))
    
    训练：
    最大化对数似然（最小化交叉熵）
    
    三种优化方法：
    1. 梯度下降
    2. 牛顿法
    3. IRLS（迭代重加权最小二乘）
    """
    
    def __init__(self, fit_intercept: bool = True,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 learning_rate: float = 0.01,
                 method: str = 'irls'):
        """
        初始化逻辑回归
        
        Args:
            fit_intercept: 是否拟合截距
            max_iter: 最大迭代次数
            tol: 收敛容差
            learning_rate: 学习率（用于梯度下降）
            method: 优化方法 ('gd', 'newton', 'irls')
        """
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.method = method
        
        self.weights = None
        self.n_features = None
        self.loss_history = []
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """添加截距项"""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid函数
        
        σ(z) = 1/(1 + exp(-z))
        
        使用scipy.special.expit，它处理了数值稳定性。
        """
        return expit(z)
    
    def _cross_entropy_loss(self, X: np.ndarray, y: np.ndarray, 
                           w: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        E = -Σ[y*log(p) + (1-y)*log(1-p)]
        
        其中p = σ(Xw)
        
        Args:
            X: 输入数据（已添加截距）
            y: 标签（0或1）
            w: 权重
            
        Returns:
            损失值
        """
        z = X @ w
        p = self._sigmoid(z)
        
        # 避免log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    def _gradient(self, X: np.ndarray, y: np.ndarray, 
                 w: np.ndarray) -> np.ndarray:
        """
        计算梯度
        
        ∇E = X^T(p - y) / n
        
        这个公式很优雅：预测误差的加权和。
        
        Args:
            X: 输入数据
            y: 标签
            w: 权重
            
        Returns:
            梯度
        """
        z = X @ w
        p = self._sigmoid(z)
        gradient = X.T @ (p - y) / len(y)
        return gradient
    
    def _hessian(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        计算Hessian矩阵（二阶导数）
        
        H = X^T R X / n
        
        其中R是对角矩阵，R_nn = p_n(1 - p_n)
        
        Hessian是正定的，所以损失函数是凸的。
        
        Args:
            X: 输入数据
            w: 权重
            
        Returns:
            Hessian矩阵
        """
        z = X @ w
        p = self._sigmoid(z)
        
        # R矩阵：对角元素是p(1-p)
        R = p * (1 - p)
        
        # H = X^T R X
        # 使用广播避免显式构造对角矩阵
        XR = X * R.reshape(-1, 1)
        H = XR.T @ X / len(X)
        
        # 添加小的正则化避免奇异
        H += 1e-8 * np.eye(len(w))
        
        return H
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用梯度下降拟合
        
        简单但可能慢。
        w := w - α∇E
        """
        n_samples, n_features = X.shape
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # 计算梯度
            gradient = self._gradient(X, y, self.weights)
            
            # 更新权重
            self.weights -= self.learning_rate * gradient
            
            # 计算损失
            loss = self._cross_entropy_loss(X, y, self.weights)
            self.loss_history.append(loss)
            
            # 检查收敛
            if np.linalg.norm(gradient) < self.tol:
                print(f"梯度下降收敛于第{iteration}次迭代")
                break
    
    def _fit_newton(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用牛顿法拟合
        
        使用二阶信息，收敛更快。
        w := w - H^(-1)∇E
        """
        n_samples, n_features = X.shape
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # 计算梯度和Hessian
            gradient = self._gradient(X, y, self.weights)
            hessian = self._hessian(X, self.weights)
            
            # 牛顿更新
            try:
                delta = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Hessian奇异，退化到梯度下降
                delta = self.learning_rate * gradient
            
            self.weights -= delta
            
            # 计算损失
            loss = self._cross_entropy_loss(X, y, self.weights)
            self.loss_history.append(loss)
            
            # 检查收敛
            if np.linalg.norm(delta) < self.tol:
                print(f"牛顿法收敛于第{iteration}次迭代")
                break
    
    def _fit_irls(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用IRLS（迭代重加权最小二乘）拟合
        
        这是牛顿法的一个变体，特别适合逻辑回归。
        
        每次迭代解一个加权最小二乘问题：
        w_new = (X^T R X)^(-1) X^T R z
        
        其中：
        - R是权重矩阵（对角）
        - z是调整的响应
        """
        n_samples, n_features = X.shape
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # 当前预测
            z = X @ self.weights
            p = self._sigmoid(z)
            
            # 权重矩阵（对角元素）
            R = p * (1 - p)
            R = np.maximum(R, 1e-10)  # 避免除零
            
            # 调整的响应
            # z_adj = Xw + (y - p) / R
            z_adj = z + (y - p) / R
            
            # 加权最小二乘
            # w = (X^T R X)^(-1) X^T R z_adj
            XR = X * R.reshape(-1, 1)
            XRX = XR.T @ X
            XRz = XR.T @ z_adj
            
            # 添加正则化
            XRX += 1e-8 * np.eye(n_features)
            
            # 求解
            weights_new = np.linalg.solve(XRX, XRz)
            
            # 检查收敛
            if np.linalg.norm(weights_new - self.weights) < self.tol:
                print(f"IRLS收敛于第{iteration}次迭代")
                self.weights = weights_new
                break
            
            self.weights = weights_new
            
            # 计算损失
            loss = self._cross_entropy_loss(X, y, self.weights)
            self.loss_history.append(loss)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        拟合逻辑回归模型
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签（0或1），shape (n_samples,)
            
        Returns:
            self
        """
        # 添加截距
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        self.n_features = X.shape[1]
        self.loss_history = []
        
        # 选择优化方法
        if self.method == 'gd':
            self._fit_gradient_descent(X, y)
        elif self.method == 'newton':
            self._fit_newton(X, y)
        elif self.method == 'irls':
            self._fit_irls(X, y)
        else:
            raise ValueError(f"未知方法: {self.method}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入数据
            
        Returns:
            概率，shape (n_samples, 2)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        p1 = self._sigmoid(X @ self.weights)
        p0 = 1 - p1
        
        return np.column_stack([p0, p1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据
            
        Returns:
            预测类别（0或1）
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)


class SoftmaxRegression:
    """
    Softmax回归（多类逻辑回归）
    
    将逻辑回归推广到K类：
    p(C_k|x) = exp(w_k^T x) / Σ_j exp(w_j^T x)
    
    这就是softmax函数。
    
    损失函数：
    交叉熵 E = -Σ_n Σ_k t_nk ln y_nk
    
    其中t_nk是1-of-K编码。
    """
    
    def __init__(self, n_classes: int,
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 learning_rate: float = 0.01):
        """
        初始化Softmax回归
        
        Args:
            n_classes: 类别数
            fit_intercept: 是否拟合截距
            max_iter: 最大迭代次数
            tol: 收敛容差
            learning_rate: 学习率
        """
        self.n_classes = n_classes
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        
        self.weights = None
        self.loss_history = []
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """添加截距项"""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        1-of-K编码
        
        将类别标签转换为one-hot向量。
        
        Args:
            y: 类别标签，shape (n_samples,)
            
        Returns:
            One-hot编码，shape (n_samples, n_classes)
        """
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y.astype(int)] = 1
        return one_hot
    
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Softmax函数
        
        softmax(z)_k = exp(z_k) / Σ_j exp(z_j)
        
        使用scipy的实现，它处理了数值稳定性。
        
        Args:
            Z: 线性输出，shape (n_samples, n_classes)
            
        Returns:
            概率，shape (n_samples, n_classes)
        """
        return softmax(Z, axis=1)
    
    def _cross_entropy_loss(self, X: np.ndarray, T: np.ndarray, 
                           W: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        E = -Σ_n Σ_k t_nk ln y_nk
        
        Args:
            X: 输入数据
            T: One-hot标签
            W: 权重矩阵
            
        Returns:
            损失值
        """
        # 预测概率
        Z = X @ W.T
        P = self._softmax(Z)
        
        # 避免log(0)
        eps = 1e-15
        P = np.clip(P, eps, 1 - eps)
        
        # 交叉熵
        loss = -np.sum(T * np.log(P)) / len(X)
        return loss
    
    def _gradient(self, X: np.ndarray, T: np.ndarray, 
                 W: np.ndarray) -> np.ndarray:
        """
        计算梯度
        
        ∇E_k = X^T (p_k - t_k) / n
        
        Args:
            X: 输入数据
            T: One-hot标签
            W: 权重矩阵
            
        Returns:
            梯度矩阵
        """
        # 预测概率
        Z = X @ W.T
        P = self._softmax(Z)
        
        # 梯度
        gradient = (P - T).T @ X / len(X)
        
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftmaxRegression':
        """
        拟合Softmax回归
        
        Args:
            X: 训练数据
            y: 标签
            
        Returns:
            self
        """
        # 添加截距
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        n_samples, n_features = X.shape
        
        # One-hot编码
        T = self._one_hot_encode(y)
        
        # 初始化权重
        self.weights = np.zeros((self.n_classes, n_features))
        
        self.loss_history = []
        
        # 梯度下降
        for iteration in range(self.max_iter):
            # 计算梯度
            gradient = self._gradient(X, T, self.weights)
            
            # 更新权重
            self.weights -= self.learning_rate * gradient
            
            # 计算损失
            loss = self._cross_entropy_loss(X, T, self.weights)
            self.loss_history.append(loss)
            
            # 检查收敛
            if np.linalg.norm(gradient) < self.tol:
                print(f"Softmax回归收敛于第{iteration}次迭代")
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        Z = X @ self.weights.T
        return self._softmax(Z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)


def demonstrate_logistic_regression(n_samples: int = 200,
                                   show_plot: bool = True) -> None:
    """
    演示逻辑回归
    
    比较不同优化方法的性能。
    
    Args:
        n_samples: 样本数
        show_plot: 是否绘图
    """
    print("\n逻辑回归演示")
    print("=" * 60)
    
    # 生成线性可分数据
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=2,
                              n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, class_sep=2.0,
                              random_state=42)
    
    # 比较不同优化方法
    methods = ['gd', 'newton', 'irls']
    models = {}
    
    for method in methods:
        print(f"\n{method.upper()}方法:")
        model = LogisticRegression(method=method, max_iter=100)
        model.fit(X, y)
        accuracy = model.score(X, y)
        print(f"  准确率: {accuracy:.2%}")
        print(f"  迭代次数: {len(model.loss_history)}")
        print(f"  最终损失: {model.loss_history[-1]:.4f}")
        models[method] = model
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：决策边界
        for idx, (method, model) in enumerate(models.items()):
            ax = axes[0, idx]
            
            # 创建网格
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            
            # 预测概率
            proba = model.predict_proba(X_grid)[:, 1]
            proba = proba.reshape(xx.shape)
            
            # 绘制概率等高线
            contour = ax.contourf(xx, yy, proba, levels=20, 
                                 cmap='RdBu_r', alpha=0.8)
            ax.contour(xx, yy, proba, levels=[0.5], 
                      colors='black', linewidths=2)
            
            # 数据点
            ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                      s=50, edgecolor='black', label='类0')
            ax.scatter(X[y==1, 0], X[y==1, 1], c='red', 
                      s=50, edgecolor='black', label='类1')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(f'{method.upper()}方法')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 第二行：损失曲线
        for idx, (method, model) in enumerate(models.items()):
            ax = axes[1, idx]
            
            ax.plot(model.loss_history, linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('交叉熵损失')
            ax.set_title(f'{method.upper()}收敛曲线')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('逻辑回归优化方法比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 梯度下降：简单但收敛慢")
    print("2. 牛顿法：使用二阶信息，收敛快")
    print("3. IRLS：专门为逻辑回归设计，稳定高效")


def demonstrate_softmax_regression(n_classes: int = 4,
                                  n_samples: int = 300,
                                  show_plot: bool = True) -> None:
    """
    演示Softmax回归（多类分类）
    
    Args:
        n_classes: 类别数
        n_samples: 样本数
        show_plot: 是否绘图
    """
    print("\nSoftmax回归（多类分类）")
    print("=" * 60)
    
    # 生成多类数据
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=2,
                              n_informative=2, n_redundant=0,
                              n_classes=n_classes, n_clusters_per_class=1,
                              class_sep=2.0, random_state=42)
    
    # 训练Softmax回归
    model = SoftmaxRegression(n_classes=n_classes, learning_rate=0.1)
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    print(f"类别数: {n_classes}")
    print(f"训练准确率: {accuracy:.2%}")
    print(f"迭代次数: {len(model.loss_history)}")
    print(f"最终损失: {model.loss_history[-1]:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 决策边界
        ax1 = axes[0]
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        
        # 预测
        Z = model.predict(X_grid)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策区域
        ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        ax1.contour(xx, yy, Z, colors='black', linewidths=0.5)
        
        # 数据点
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        for k in range(n_classes):
            mask = y == k
            ax1.scatter(X[mask, 0], X[mask, 1], c=[colors[k]],
                       s=50, edgecolor='black', label=f'类{k}')
        
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title('决策边界')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2 = axes[1]
        ax2.plot(model.loss_history, linewidth=2)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('交叉熵损失')
        ax2.set_title('损失收敛')
        ax2.grid(True, alpha=0.3)
        
        # 概率分布
        ax3 = axes[2]
        
        # 在一条线上采样，显示概率变化
        n_points = 100
        x_line = np.linspace(x_min, x_max, n_points)
        y_line = np.zeros(n_points)
        X_line = np.column_stack([x_line, y_line])
        
        proba = model.predict_proba(X_line)
        
        for k in range(n_classes):
            ax3.plot(x_line, proba[:, k], linewidth=2, 
                    color=colors[k], label=f'P(类{k}|x)')
        
        ax3.set_xlabel('x₁ (x₂=0)')
        ax3.set_ylabel('概率')
        ax3.set_title('概率随x变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        plt.suptitle('Softmax回归（多类分类）', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. Softmax将线性输出转换为概率")
    print("2. 决策边界是线性的")
    print("3. 概率和为1（归一化）")
    print("4. 交叉熵损失适合多类分类")