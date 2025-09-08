"""
4.1 判别函数 (Discriminant Functions)
=====================================

判别函数直接将输入空间划分为不同的决策区域。
每个区域对应一个类别。

线性判别函数：
y(x) = w^T x + w₀

决策边界：
y(x) = 0

二分类：
- 单个判别函数 y(x)
- y(x) > 0 → 类别1
- y(x) < 0 → 类别2

多分类（K类）：
方法1：一对多（One-vs-Rest）
- K个判别函数 y_k(x)
- 选择最大的：k* = argmax_k y_k(x)

方法2：一对一（One-vs-One）
- K(K-1)/2个判别函数
- 投票决定

方法3：单层多类
- K个判别函数，直接比较

判别函数的几何解释：
- w：决策边界的法向量
- w₀：偏置（决定边界位置）
- |y(x)|/||w||：点到边界的距离

关键概念：
1. 决策区域：分配给某类的输入空间区域
2. 决策边界：区域之间的边界
3. 线性可分：存在线性边界完美分类
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class LinearDiscriminant:
    """
    线性判别函数
    
    实现二分类的线性判别。
    判别函数：y(x) = w^T x + w₀
    
    决策规则：
    - y(x) > 0 → 类别1
    - y(x) < 0 → 类别2
    - y(x) = 0 → 决策边界
    
    训练方法：
    这里使用最小二乘法，虽然不是最优的，
    但简单直观，适合教学。
    """
    
    def __init__(self, n_features: int, include_bias: bool = True):
        """
        初始化线性判别器
        
        Args:
            n_features: 特征维度
            include_bias: 是否包含偏置项
        """
        self.n_features = n_features
        self.include_bias = include_bias
        
        # 权重向量
        if include_bias:
            # 将偏置合并到权重中
            self.weights = np.zeros(n_features + 1)
        else:
            self.weights = np.zeros(n_features)
            
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        添加偏置项（1列）
        
        将x扩展为[1, x]，这样w₀就成为权重的一部分。
        
        Args:
            X: 输入数据，shape (n_samples, n_features)
            
        Returns:
            扩展后的数据，shape (n_samples, n_features + 1)
        """
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearDiscriminant':
        """
        使用最小二乘法训练
        
        最小化：||Xw - t||²
        解：w = (X^T X)^(-1) X^T t
        
        其中t是目标编码：
        - 类别1 → +1
        - 类别2 → -1
        
        注意：最小二乘法对分类不是最优的，
        因为它对离群点敏感，且不能保证概率解释。
        但它简单且有闭式解。
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签（0或1），shape (n_samples,)
            
        Returns:
            self
        """
        # 转换标签：0→-1, 1→+1
        t = 2 * y - 1
        
        # 添加偏置项
        if self.include_bias:
            X_extended = self._add_bias(X)
        else:
            X_extended = X
        
        # 最小二乘解
        # w = (X^T X)^(-1) X^T t
        XtX = X_extended.T @ X_extended
        
        # 添加小的正则化项避免奇异
        reg = 1e-6 * np.eye(XtX.shape[0])
        XtX += reg
        
        # 求解
        Xtt = X_extended.T @ t
        self.weights = np.linalg.solve(XtX, Xtt)
        
        return self
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        计算判别函数值
        
        y(x) = w^T x + w₀
        
        Args:
            X: 输入数据
            
        Returns:
            判别函数值
        """
        if self.include_bias:
            X_extended = self._add_bias(X)
        else:
            X_extended = X
            
        return X_extended @ self.weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据
            
        Returns:
            预测类别（0或1）
        """
        scores = self.predict_score(X)
        return (scores > 0).astype(int)
    
    def decision_function(self, x: np.ndarray) -> float:
        """
        单个样本的判别函数值
        
        用于可视化决策边界。
        
        Args:
            x: 单个样本
            
        Returns:
            判别函数值
        """
        return self.predict_score(x.reshape(1, -1))[0]
    
    def distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        计算点到决策边界的距离
        
        距离 = |y(x)| / ||w||
        
        这个距离是有符号的：
        - 正：在类别1一侧
        - 负：在类别2一侧
        
        Args:
            X: 输入数据
            
        Returns:
            到边界的距离
        """
        scores = self.predict_score(X)
        
        # 权重的范数（不包括偏置）
        if self.include_bias:
            w_norm = np.linalg.norm(self.weights[1:])
        else:
            w_norm = np.linalg.norm(self.weights)
        
        return scores / w_norm


class MultiClassDiscriminant:
    """
    多类线性判别
    
    K个类别，K个判别函数：
    y_k(x) = w_k^T x + w_{k0}
    
    决策规则：
    选择最大的判别函数值
    C(x) = argmax_k y_k(x)
    
    这种方法避免了一对多或一对一的歧义区域。
    """
    
    def __init__(self, n_features: int, n_classes: int):
        """
        初始化多类判别器
        
        Args:
            n_features: 特征维度
            n_classes: 类别数
        """
        self.n_features = n_features
        self.n_classes = n_classes
        
        # 权重矩阵：每行是一个类的权重
        # shape: (n_classes, n_features + 1)
        self.weights = np.zeros((n_classes, n_features + 1))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassDiscriminant':
        """
        训练多类判别器
        
        使用1-of-K编码和最小二乘法。
        
        对每个类别k：
        - t_k = 1 如果样本属于类k
        - t_k = 0 否则
        
        最小化：||XW - T||²_F
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签，shape (n_samples,)
            
        Returns:
            self
        """
        n_samples = X.shape[0]
        
        # 添加偏置列
        X_extended = np.column_stack([np.ones(n_samples), X])
        
        # 1-of-K编码
        T = np.zeros((n_samples, self.n_classes))
        for i in range(n_samples):
            T[i, y[i]] = 1
        
        # 最小二乘解：W = (X^T X)^(-1) X^T T
        XtX = X_extended.T @ X_extended
        
        # 正则化
        reg = 1e-6 * np.eye(XtX.shape[0])
        XtX += reg
        
        # 求解每个类的权重
        XtT = X_extended.T @ T
        W = np.linalg.solve(XtX, XtT)
        
        # 转置得到权重矩阵
        self.weights = W.T
        
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        计算所有类的判别函数值
        
        Args:
            X: 输入数据，shape (n_samples, n_features)
            
        Returns:
            判别函数值，shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        X_extended = np.column_stack([np.ones(n_samples), X])
        
        # Y = X W^T
        return X_extended @ self.weights.T
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        选择判别函数值最大的类。
        
        Args:
            X: 输入数据
            
        Returns:
            预测类别
        """
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（使用softmax）
        
        虽然判别函数不直接给出概率，
        但可以用softmax转换：
        p(C_k|x) ∝ exp(y_k(x))
        
        Args:
            X: 输入数据
            
        Returns:
            类别概率
        """
        scores = self.predict_scores(X)
        return softmax(scores, axis=1)


def generate_linear_separable_data(n_samples: int = 200,
                                  n_features: int = 2,
                                  margin: float = 1.0,
                                  noise: float = 0.1,
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成线性可分数据
    
    创建两个类，由线性边界分开。
    
    Args:
        n_samples: 样本总数
        n_features: 特征维度
        margin: 类间间隔
        noise: 噪声水平
        seed: 随机种子
        
    Returns:
        X, y: 数据和标签
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_per_class = n_samples // 2
    
    # 随机生成分离超平面
    w = np.random.randn(n_features)
    w = w / np.linalg.norm(w)
    
    # 生成类1：w^T x > margin/2
    X1 = np.random.randn(n_per_class, n_features)
    # 投影并移动到正确一侧
    proj1 = X1 @ w
    X1 += (margin/2 - proj1.min() + np.abs(np.random.randn())) * w.reshape(1, -1)
    
    # 生成类2：w^T x < -margin/2
    X2 = np.random.randn(n_per_class, n_features)
    proj2 = X2 @ w
    X2 -= (margin/2 + proj2.max() + np.abs(np.random.randn())) * w.reshape(1, -1)
    
    # 添加噪声
    X1 += noise * np.random.randn(n_per_class, n_features)
    X2 += noise * np.random.randn(n_per_class, n_features)
    
    # 合并数据
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per_class), np.zeros(n_per_class)]).astype(int)
    
    # 打乱顺序
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y


def visualize_decision_boundary(model: Union[LinearDiscriminant, MultiClassDiscriminant],
                               X: np.ndarray, y: np.ndarray,
                               title: str = "决策边界",
                               ax: Optional[plt.Axes] = None) -> None:
    """
    可视化2D决策边界
    
    展示：
    1. 数据点
    2. 决策边界
    3. 决策区域
    
    Args:
        model: 训练好的判别模型
        X: 数据点
        y: 标签
        title: 图标题
        ax: 绘图轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    
    if isinstance(model, LinearDiscriminant):
        Z = model.predict(X_grid)
        n_classes = 2
    else:
        Z = model.predict(X_grid)
        n_classes = model.n_classes
    
    Z = Z.reshape(xx.shape)
    
    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=n_classes-1)
    
    # 绘制决策边界
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    # 绘制数据点
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    for i in range(n_classes if n_classes > 2 else 2):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                  s=50, edgecolors='black', label=f'类{i}')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def demonstrate_two_class_discriminant(n_samples: int = 200,
                                      show_plot: bool = True) -> None:
    """
    演示二分类判别函数
    
    展示：
    1. 线性判别函数
    2. 决策边界
    3. 到边界的距离
    
    Args:
        n_samples: 样本数
        show_plot: 是否绘图
    """
    print("\n二分类线性判别函数")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_linear_separable_data(n_samples, n_features=2, 
                                         margin=2.0, noise=0.3, seed=42)
    
    # 训练判别器
    discriminant = LinearDiscriminant(n_features=2)
    discriminant.fit(X, y)
    
    # 预测
    y_pred = discriminant.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"训练样本: {n_samples}")
    print(f"权重向量: w = {discriminant.weights[1:]}")
    print(f"偏置: w₀ = {discriminant.weights[0]:.3f}")
    print(f"训练准确率: {accuracy:.2%}")
    
    # 计算到边界的距离
    distances = discriminant.distance_to_boundary(X)
    print(f"\n到决策边界的平均距离:")
    print(f"  类0: {np.mean(np.abs(distances[y==0])):.3f}")
    print(f"  类1: {np.mean(np.abs(distances[y==1])):.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：决策边界
        visualize_decision_boundary(discriminant, X, y, 
                                   "线性判别函数", axes[0])
        
        # 右图：判别函数值分布
        ax2 = axes[1]
        scores = discriminant.predict_score(X)
        
        ax2.hist(scores[y==0], bins=20, alpha=0.5, color='blue', 
                label='类0', density=True)
        ax2.hist(scores[y==1], bins=20, alpha=0.5, color='red', 
                label='类1', density=True)
        ax2.axvline(x=0, color='black', linestyle='--', 
                   label='决策边界')
        
        ax2.set_xlabel('判别函数值 y(x)')
        ax2.set_ylabel('密度')
        ax2.set_title('判别函数值分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('二分类线性判别', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 线性边界将空间分为两个半空间")
    print("2. 判别函数值的符号决定类别")
    print("3. 判别函数值的大小表示置信度")


def demonstrate_multi_class_discriminant(n_classes: int = 3,
                                        n_samples_per_class: int = 50,
                                        show_plot: bool = True) -> None:
    """
    演示多类判别函数
    
    展示：
    1. 多个判别函数
    2. 决策区域
    3. 歧义区域（如果有）
    
    Args:
        n_classes: 类别数
        n_samples_per_class: 每类样本数
        show_plot: 是否绘图
    """
    print("\n多类线性判别函数")
    print("=" * 60)
    print(f"类别数: {n_classes}")
    
    # 生成多类数据
    np.random.seed(42)
    X_list = []
    y_list = []
    
    # 在不同位置生成各类数据
    angles = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
    radius = 3.0
    
    for k in range(n_classes):
        # 类中心
        center = radius * np.array([np.cos(angles[k]), np.sin(angles[k])])
        
        # 生成该类数据
        X_k = np.random.randn(n_samples_per_class, 2) + center
        X_list.append(X_k)
        y_list.append(np.full(n_samples_per_class, k))
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # 训练多类判别器
    discriminant = MultiClassDiscriminant(n_features=2, n_classes=n_classes)
    discriminant.fit(X, y)
    
    # 预测
    y_pred = discriminant.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"训练样本: {len(X)}")
    print(f"训练准确率: {accuracy:.2%}")
    
    # 显示每个类的权重
    print("\n各类权重向量:")
    for k in range(n_classes):
        w = discriminant.weights[k, 1:]
        w0 = discriminant.weights[k, 0]
        print(f"  类{k}: w = [{w[0]:.2f}, {w[1]:.2f}], w₀ = {w0:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：决策区域
        visualize_decision_boundary(discriminant, X, y,
                                   f"{n_classes}类判别", axes[0])
        
        # 右图：判别函数值
        ax2 = axes[1]
        scores = discriminant.predict_scores(X)
        
        # 每个类的判别函数值
        x_plot = np.arange(len(X))
        for k in range(n_classes):
            ax2.scatter(x_plot[y==k], scores[y==k, k], 
                       alpha=0.5, s=10, label=f'类{k}')
        
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel('判别函数值')
        ax2.set_title('各类判别函数值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{n_classes}类线性判别', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. K个类需要K个判别函数")
    print("2. 决策区域是凸的")
    print("3. 不存在歧义区域")


def compare_multiclass_strategies(n_classes: int = 4,
                                 n_samples_per_class: int = 30,
                                 show_plot: bool = True) -> None:
    """
    比较多类分类策略
    
    比较：
    1. 直接多类（K个判别函数）
    2. 一对多（One-vs-Rest）
    3. 一对一（One-vs-One）
    
    Args:
        n_classes: 类别数
        n_samples_per_class: 每类样本数
        show_plot: 是否绘图
    """
    print("\n多类分类策略比较")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_classes*n_samples_per_class,
                              n_features=2, n_informative=2, n_redundant=0,
                              n_classes=n_classes, n_clusters_per_class=1,
                              class_sep=2.0, random_state=42)
    
    # 策略1：直接多类
    print("\n1. 直接多类判别:")
    multi_direct = MultiClassDiscriminant(n_features=2, n_classes=n_classes)
    multi_direct.fit(X, y)
    acc_direct = np.mean(multi_direct.predict(X) == y)
    print(f"   准确率: {acc_direct:.2%}")
    
    # 策略2：一对多（One-vs-Rest）
    print("\n2. 一对多（OvR）:")
    ovr_models = []
    for k in range(n_classes):
        # 将类k作为正类，其他作为负类
        y_binary = (y == k).astype(int)
        model = LinearDiscriminant(n_features=2)
        model.fit(X, y_binary)
        ovr_models.append(model)
    
    # OvR预测：选择得分最高的类
    ovr_scores = np.zeros((len(X), n_classes))
    for k, model in enumerate(ovr_models):
        ovr_scores[:, k] = model.predict_score(X)
    y_pred_ovr = np.argmax(ovr_scores, axis=1)
    acc_ovr = np.mean(y_pred_ovr == y)
    print(f"   准确率: {acc_ovr:.2%}")
    print(f"   二分类器数量: {n_classes}")
    
    # 策略3：一对一（One-vs-One）
    print("\n3. 一对一（OvO）:")
    ovo_models = {}
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            # 只使用类i和类j的数据
            mask = (y == i) | (y == j)
            X_pair = X[mask]
            y_pair = (y[mask] == j).astype(int)
            
            model = LinearDiscriminant(n_features=2)
            model.fit(X_pair, y_pair)
            ovo_models[(i, j)] = model
    
    # OvO预测：投票
    votes = np.zeros((len(X), n_classes))
    for (i, j), model in ovo_models.items():
        pred = model.predict(X)
        votes[pred == 0, i] += 1
        votes[pred == 1, j] += 1
    y_pred_ovo = np.argmax(votes, axis=1)
    acc_ovo = np.mean(y_pred_ovo == y)
    print(f"   准确率: {acc_ovo:.2%}")
    print(f"   二分类器数量: {len(ovo_models)}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 绘制三种策略的决策边界
        strategies = [
            (multi_direct, "直接多类"),
            (ovr_models, "一对多（OvR）"),
            (ovo_models, "一对一（OvO）")
        ]
        
        for idx, (model, name) in enumerate(strategies):
            ax = axes[idx]
            
            # 创建网格
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            
            # 预测
            if isinstance(model, MultiClassDiscriminant):
                Z = model.predict(X_grid)
            elif isinstance(model, list):  # OvR
                scores = np.zeros((len(X_grid), n_classes))
                for k, m in enumerate(model):
                    scores[:, k] = m.predict_score(X_grid)
                Z = np.argmax(scores, axis=1)
            else:  # OvO
                votes = np.zeros((len(X_grid), n_classes))
                for (i, j), m in model.items():
                    pred = m.predict(X_grid)
                    votes[pred == 0, i] += 1
                    votes[pred == 1, j] += 1
                Z = np.argmax(votes, axis=1)
            
            Z = Z.reshape(xx.shape)
            
            # 绘制
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            ax.contour(xx, yy, Z, colors='black', linewidths=0.5)
            
            # 数据点
            colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
            for k in range(n_classes):
                mask = y == k
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[k]],
                          s=30, edgecolors='black', alpha=0.7)
            
            ax.set_title(name)
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('多类分类策略比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n比较总结：")
    print("1. 直接多类：简单，无歧义区域")
    print("2. OvR：可能有歧义区域或无分类区域")
    print("3. OvO：训练快（每次只用两类数据），但分类器多")