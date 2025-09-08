"""
3.1 线性基函数模型 (Linear Basis Function Models)
==================================================

线性回归的核心思想：
虽然模型对于参数w是线性的，但通过基函数变换φ(x)，
可以对输入x建模非线性关系。

模型形式：
y(x,w) = w₀ + Σᵢ wᵢφᵢ(x) = w^T φ(x)

其中：
- φ(x) = [1, φ₁(x), φ₂(x), ..., φₘ(x)]^T 是基函数向量
- w = [w₀, w₁, ..., wₘ]^T 是权重向量
- w₀ 是偏置项（通过φ₀(x)=1实现）

基函数的选择：
1. 多项式基函数：φⱼ(x) = x^j
2. 高斯基函数：φⱼ(x) = exp(-(x-μⱼ)²/(2s²))
3. Sigmoid基函数：φⱼ(x) = σ((x-μⱼ)/s)
4. 傅里叶基函数：φⱼ(x) = sin(jπx) 或 cos(jπx)

关键洞察：
- 模型对w线性 → 凸优化问题，有全局最优解
- 基函数固定 → 特征提取与学习分离
- 不同基函数 → 不同的函数空间和归纳偏置
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Union
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')


class BasisFunction:
    """
    基函数的基类
    
    所有具体的基函数类都继承这个类。
    基函数将输入空间映射到特征空间。
    """
    
    def __init__(self, n_basis: int):
        """
        初始化基函数
        
        Args:
            n_basis: 基函数的数量（不包括偏置项）
        """
        self.n_basis = n_basis
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        计算基函数值
        
        Args:
            x: 输入，shape (n_samples,) 或 (n_samples, 1)
            
        Returns:
            Φ: 设计矩阵，shape (n_samples, n_basis + 1)
               第一列是1（偏置项）
        """
        raise NotImplementedError
    
    def plot_basis(self, x_range: Tuple[float, float] = (0, 1),
                  ax: Optional[plt.Axes] = None) -> None:
        """
        绘制基函数
        
        可视化每个基函数的形状，帮助理解
        它们如何组合成复杂的函数。
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x = np.linspace(x_range[0], x_range[1], 200)
        phi = self(x)
        
        # 绘制每个基函数（除了偏置项）
        for j in range(1, phi.shape[1]):
            ax.plot(x, phi[:, j], label=f'φ_{j}(x)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('φ(x)')
        ax.set_title(f'{self.__class__.__name__} (M={self.n_basis})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)


class PolynomialBasis(BasisFunction):
    """
    多项式基函数
    
    φⱼ(x) = x^j, j = 0, 1, ..., M
    
    优点：
    - 简单直观
    - 泰勒展开的自然选择
    
    缺点：
    - 全局支撑（改变一处影响全局）
    - 高阶时数值不稳定
    - 容易过拟合
    
    应用：
    - 小区间上的函数近似
    - 物理学中的泰勒展开
    """
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        计算多项式基函数
        
        返回范德蒙德矩阵：
        [1, x, x², x³, ..., x^M]
        """
        x = np.asarray(x).ravel()
        n_samples = len(x)
        
        # 创建设计矩阵
        phi = np.zeros((n_samples, self.n_basis + 1))
        
        # φ₀(x) = 1 (偏置项)
        # φⱼ(x) = x^j
        for j in range(self.n_basis + 1):
            phi[:, j] = x ** j
        
        return phi


class GaussianBasis(BasisFunction):
    """
    高斯基函数（径向基函数RBF）
    
    φⱼ(x) = exp(-(x - μⱼ)² / (2s²))
    
    其中：
    - μⱼ 是第j个基函数的中心
    - s 是宽度参数（所有基函数共享）
    
    优点：
    - 局部支撑（局部改变不影响远处）
    - 平滑
    - 可解释性好（每个基函数对应一个区域）
    
    缺点：
    - 需要选择中心位置和宽度
    - 维度诅咒（高维时需要指数级的基函数）
    
    应用：
    - RBF网络
    - 核方法的基础
    - 函数插值
    """
    
    def __init__(self, n_basis: int, x_min: float = 0, x_max: float = 1,
                 width: Optional[float] = None):
        """
        初始化高斯基函数
        
        Args:
            n_basis: 基函数数量
            x_min, x_max: 输入范围
            width: 基函数宽度，None则自动计算
        """
        super().__init__(n_basis)
        
        # 均匀分布基函数中心
        self.centers = np.linspace(x_min, x_max, n_basis)
        
        # 设置宽度（如果未指定，使用中心间距）
        if width is None:
            if n_basis > 1:
                self.width = (x_max - x_min) / (n_basis - 1) * 0.5
            else:
                self.width = (x_max - x_min) * 0.5
        else:
            self.width = width
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        计算高斯基函数
        
        每个基函数是一个高斯"山峰"，
        组合起来可以近似任意连续函数。
        """
        x = np.asarray(x).ravel()
        n_samples = len(x)
        
        # 创建设计矩阵
        phi = np.zeros((n_samples, self.n_basis + 1))
        
        # φ₀(x) = 1 (偏置项)
        phi[:, 0] = 1
        
        # φⱼ(x) = exp(-(x - μⱼ)² / (2s²))
        for j in range(self.n_basis):
            phi[:, j + 1] = np.exp(-0.5 * ((x - self.centers[j]) / self.width) ** 2)
        
        return phi


class SigmoidBasis(BasisFunction):
    """
    Sigmoid基函数
    
    φⱼ(x) = σ((x - μⱼ) / s)
    
    其中σ(a) = 1 / (1 + exp(-a))
    
    特点：
    - S形曲线
    - 单调递增
    - 渐近行为
    
    应用：
    - 神经网络的激活函数
    - 分段近似
    """
    
    def __init__(self, n_basis: int, x_min: float = 0, x_max: float = 1,
                 width: float = 0.1):
        super().__init__(n_basis)
        self.centers = np.linspace(x_min, x_max, n_basis)
        self.width = width
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        n_samples = len(x)
        
        phi = np.zeros((n_samples, self.n_basis + 1))
        phi[:, 0] = 1
        
        for j in range(self.n_basis):
            phi[:, j + 1] = 1 / (1 + np.exp(-(x - self.centers[j]) / self.width))
        
        return phi


class FourierBasis(BasisFunction):
    """
    傅里叶基函数
    
    φⱼ(x) = sin(jπx/L) 或 cos(jπx/L)
    
    特点：
    - 全局支撑
    - 正交基
    - 适合周期函数
    
    应用：
    - 信号处理
    - 周期函数近似
    - 谱方法
    """
    
    def __init__(self, n_basis: int, period: float = 1.0):
        # n_basis应该是偶数（sin和cos成对）
        super().__init__(n_basis)
        self.period = period
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        n_samples = len(x)
        
        phi = np.zeros((n_samples, self.n_basis + 1))
        phi[:, 0] = 1
        
        idx = 1
        for j in range(1, (self.n_basis // 2) + 1):
            # sin项
            if idx <= self.n_basis:
                phi[:, idx] = np.sin(2 * np.pi * j * x / self.period)
                idx += 1
            # cos项
            if idx <= self.n_basis:
                phi[:, idx] = np.cos(2 * np.pi * j * x / self.period)
                idx += 1
        
        return phi


class LinearRegression:
    """
    线性回归模型（使用基函数）
    
    最小化平方损失：
    E(w) = (1/2) Σₙ (tₙ - w^T φ(xₙ))²
    
    加上L2正则化：
    E(w) = (1/2) Σₙ (tₙ - w^T φ(xₙ))² + (λ/2) ||w||²
    
    解析解（正规方程）：
    w = (Φ^T Φ + λI)^(-1) Φ^T t
    
    其中Φ是设计矩阵，每行是φ(xₙ)^T
    """
    
    def __init__(self, basis_function: BasisFunction, 
                 regularization: float = 0.0):
        """
        初始化线性回归模型
        
        Args:
            basis_function: 基函数对象
            regularization: L2正则化系数λ
        """
        self.basis_function = basis_function
        self.regularization = regularization
        self.weights = None
        self.training_error = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        拟合模型
        
        使用正规方程求解：
        w = (Φ^T Φ + λI)^(-1) Φ^T y
        
        这是凸优化问题，有唯一全局最优解。
        
        Args:
            X: 输入数据，shape (n_samples,)
            y: 目标值，shape (n_samples,)
            
        Returns:
            self
        """
        # 计算设计矩阵
        Phi = self.basis_function(X)
        n_features = Phi.shape[1]
        
        # 正规方程
        # A = Φ^T Φ + λI
        A = Phi.T @ Phi
        
        # 添加正则化（注意：通常不正则化偏置项）
        if self.regularization > 0:
            reg_matrix = self.regularization * np.eye(n_features)
            # 不正则化偏置项
            reg_matrix[0, 0] = 0
            A += reg_matrix
        
        # b = Φ^T y
        b = Phi.T @ y
        
        # 求解 Aw = b
        try:
            self.weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            print("警告：矩阵奇异，使用伪逆求解")
            self.weights = np.linalg.pinv(A) @ b
        
        # 计算训练误差
        predictions = self.predict(X)
        self.training_error = np.sqrt(np.mean((y - predictions) ** 2))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        y = w^T φ(x)
        
        Args:
            X: 输入数据
            
        Returns:
            预测值
        """
        if self.weights is None:
            raise ValueError("模型还未拟合")
        
        Phi = self.basis_function(X)
        return Phi @ self.weights
    
    def compute_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算均方根误差"""
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))


def generate_synthetic_data(n_samples: int, 
                           noise_std: float = 0.1,
                           x_min: float = 0,
                           x_max: float = 1,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成数据
    
    使用sin函数加噪声：
    t = sin(2πx) + ε
    
    Args:
        n_samples: 样本数量
        noise_std: 噪声标准差
        x_min, x_max: x的范围
        seed: 随机种子
        
    Returns:
        X, y: 输入和目标值
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.uniform(x_min, x_max, n_samples)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, noise_std, n_samples)
    
    return X, y


def compare_basis_functions(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          n_basis: int = 9,
                          regularization: float = 0.0,
                          show_plot: bool = True) -> None:
    """
    比较不同基函数的性能
    
    展示不同基函数如何影响：
    1. 拟合能力
    2. 泛化性能
    3. 函数形状
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        n_basis: 基函数数量
        regularization: 正则化系数
        show_plot: 是否绘图
    """
    print("\n基函数比较")
    print("=" * 60)
    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"基函数数量: {n_basis}, 正则化: λ={regularization}")
    print("-" * 60)
    
    # 创建不同的基函数
    basis_functions = {
        '多项式': PolynomialBasis(n_basis),
        '高斯': GaussianBasis(n_basis, X_train.min(), X_train.max()),
        'Sigmoid': SigmoidBasis(n_basis, X_train.min(), X_train.max()),
        '傅里叶': FourierBasis(n_basis)
    }
    
    results = {}
    models = {}
    
    for name, basis in basis_functions.items():
        # 训练模型
        model = LinearRegression(basis, regularization)
        model.fit(X_train, y_train)
        
        # 计算误差
        train_error = model.compute_error(X_train, y_train)
        test_error = model.compute_error(X_test, y_test)
        
        results[name] = {
            'train_error': train_error,
            'test_error': test_error,
            'weights': model.weights
        }
        models[name] = model
        
        print(f"{name:10} - 训练RMSE: {train_error:.4f}, 测试RMSE: {test_error:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 绘制基函数
        for idx, (name, basis) in enumerate(basis_functions.items()):
            ax = axes[0, idx]
            basis.plot_basis((X_train.min(), X_train.max()), ax)
            ax.set_title(f'{name}基函数')
        
        # 绘制拟合结果
        x_plot = np.linspace(X_train.min(), X_train.max(), 200)
        y_true = np.sin(2 * np.pi * x_plot)
        
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[1, idx]
            
            y_pred = model.predict(x_plot)
            
            # 真实函数
            ax.plot(x_plot, y_true, 'g-', label='真实函数', alpha=0.5)
            # 拟合函数
            ax.plot(x_plot, y_pred, 'r-', label=f'{name}拟合', linewidth=2)
            # 训练数据
            ax.scatter(X_train, y_train, s=20, c='blue', alpha=0.5, label='训练数据')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{name} (测试RMSE={results[name]["test_error"]:.3f})')
            ax.legend(fontsize=8)
            ax.set_ylim([-2, 2])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'不同基函数的比较 (M={n_basis}, λ={regularization})', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 多项式基函数：全局支撑，高阶时可能振荡")
    print("2. 高斯基函数：局部支撑，平滑插值")
    print("3. Sigmoid基函数：单调变化，适合阶跃函数")
    print("4. 傅里叶基函数：周期性，适合周期函数")


def demonstrate_regularization_effect(X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray,
                                     basis_type: str = 'polynomial',
                                     n_basis: int = 9,
                                     lambda_values: List[float] = [0, 0.001, 0.01, 0.1],
                                     show_plot: bool = True) -> None:
    """
    演示正则化的效果
    
    展示L2正则化如何：
    1. 控制模型复杂度
    2. 减少过拟合
    3. 影响权重大小
    
    正则化的几何解释：
    - 在权重空间中添加一个"惩罚区域"
    - λ越大，惩罚越强，权重越接近0
    - 等价于对权重施加高斯先验（贝叶斯观点）
    """
    print("\n正则化效果演示")
    print("=" * 60)
    print(f"基函数类型: {basis_type}, M={n_basis}")
    print("-" * 60)
    
    # 选择基函数
    if basis_type == 'polynomial':
        basis = PolynomialBasis(n_basis)
    elif basis_type == 'gaussian':
        basis = GaussianBasis(n_basis, X_train.min(), X_train.max())
    else:
        basis = PolynomialBasis(n_basis)
    
    results = []
    
    for lambda_val in lambda_values:
        model = LinearRegression(basis, lambda_val)
        model.fit(X_train, y_train)
        
        train_error = model.compute_error(X_train, y_train)
        test_error = model.compute_error(X_test, y_test)
        weight_norm = np.linalg.norm(model.weights[1:])  # 不包括偏置
        
        results.append({
            'lambda': lambda_val,
            'train_error': train_error,
            'test_error': test_error,
            'weight_norm': weight_norm,
            'model': model
        })
        
        print(f"λ={lambda_val:8.4f}: 训练RMSE={train_error:.4f}, "
              f"测试RMSE={test_error:.4f}, ||w||={weight_norm:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, len(lambda_values), 
                                figsize=(4*len(lambda_values), 8))
        
        x_plot = np.linspace(X_train.min(), X_train.max(), 200)
        y_true = np.sin(2 * np.pi * x_plot)
        
        for idx, result in enumerate(results):
            # 上图：拟合结果
            ax1 = axes[0, idx] if len(lambda_values) > 1 else axes[0]
            
            y_pred = result['model'].predict(x_plot)
            
            ax1.plot(x_plot, y_true, 'g-', label='真实', alpha=0.5)
            ax1.plot(x_plot, y_pred, 'r-', label='拟合', linewidth=2)
            ax1.scatter(X_train, y_train, s=20, c='blue', alpha=0.5)
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'λ={result["lambda"]:.4f}')
            ax1.set_ylim([-2, 2])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下图：权重分布
            ax2 = axes[1, idx] if len(lambda_values) > 1 else axes[1]
            
            weights = result['model'].weights
            ax2.bar(range(len(weights)), weights, color='blue', alpha=0.7)
            ax2.set_xlabel('权重索引')
            ax2.set_ylabel('权重值')
            ax2.set_title(f'权重分布 (||w||={result["weight_norm"]:.3f})')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'正则化效果 ({basis_type}基函数, M={n_basis})', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. λ=0：无正则化，可能过拟合")
    print("2. λ增大：权重变小，函数变平滑")
    print("3. λ过大：欠拟合，丢失重要特征")
    print("4. 最优λ：在偏差和方差之间平衡")