"""
6.1-6.2 核函数 (Kernel Functions)
==================================

核方法的核心思想：
通过核函数隐式地在高维（甚至无限维）特征空间中计算内积，
而不需要显式地计算特征映射。

核技巧（Kernel Trick）：
k(x, x') = φ(x)^T φ(x')

其中φ是从输入空间到特征空间的映射。

核函数的条件（Mercer定理）：
核函数k必须是正定的，即对任意x₁,...,xₙ，
Gram矩阵K（其中K_ij = k(x_i, x_j)）是半正定的。

常用核函数：
1. 线性核：k(x,x') = x^T x'
2. 多项式核：k(x,x') = (x^T x' + c)^d
3. 高斯RBF核：k(x,x') = exp(-γ||x-x'||²)
4. Laplace核：k(x,x') = exp(-γ||x-x'||₁)
5. Sigmoid核：k(x,x') = tanh(αx^T x' + c)

核的构造：
1. 核的线性组合
2. 核的乘积
3. 函数作用后的核

应用：
- SVM（支持向量机）
- 核PCA
- 核岭回归
- 高斯过程
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union, Tuple, List
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')


class Kernel:
    """
    核函数基类
    
    所有核函数都需要实现__call__方法。
    """
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算核矩阵
        
        Args:
            X: 第一组数据点，shape (n_samples_X, n_features)
            Y: 第二组数据点，shape (n_samples_Y, n_features)
               如果为None，则Y=X
               
        Returns:
            核矩阵K，其中K[i,j] = k(X[i], Y[j])
            shape (n_samples_X, n_samples_Y)
        """
        raise NotImplementedError
    
    def is_stationary(self) -> bool:
        """是否是平稳核（只依赖于x-x'）"""
        return False
    
    def is_positive_definite(self) -> bool:
        """是否保证正定"""
        return True


class LinearKernel(Kernel):
    """
    线性核
    
    k(x, x') = x^T x'
    
    最简单的核，等价于在原始空间中的内积。
    适用于线性可分的问题。
    """
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X
        return X @ Y.T


class PolynomialKernel(Kernel):
    """
    多项式核
    
    k(x, x') = (γ x^T x' + c)^d
    
    隐式地计算d次多项式特征的内积。
    
    参数：
    - degree: 多项式次数d
    - gamma: 缩放参数γ
    - coef0: 常数项c
    
    当d=1, c=0时退化为线性核。
    """
    
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X
        return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """
    径向基函数（RBF）核 / 高斯核
    
    k(x, x') = exp(-γ ||x - x'||²)
    
    最常用的核函数之一。
    
    特点：
    - 局部性：远处的点影响小
    - 平滑性：无限可微
    - 万能性：可以近似任意函数
    - 对应无限维特征空间
    
    参数：
    - gamma: 控制影响范围，γ = 1/(2σ²)
      γ大：影响范围小，容易过拟合
      γ小：影响范围大，容易欠拟合
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma: 核宽度参数，gamma = 1/(2*sigma^2)
        """
        self.gamma = gamma
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X
        
        # 计算欧氏距离的平方
        # ||x - y||² = ||x||² + ||y||² - 2x^T y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        distances_sq = X_norm + Y_norm - 2 * (X @ Y.T)
        
        # 避免数值问题
        distances_sq = np.maximum(distances_sq, 0)
        
        return np.exp(-self.gamma * distances_sq)
    
    def is_stationary(self) -> bool:
        return True


class LaplacianKernel(Kernel):
    """
    Laplace核
    
    k(x, x') = exp(-γ ||x - x'||₁)
    
    使用L1距离而不是L2距离。
    比RBF核更尖锐，对离群点更鲁棒。
    """
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X
        
        # 计算L1距离
        distances = cdist(X, Y, metric='cityblock')
        
        return np.exp(-self.gamma * distances)
    
    def is_stationary(self) -> bool:
        return True


class SigmoidKernel(Kernel):
    """
    Sigmoid核
    
    k(x, x') = tanh(α x^T x' + c)
    
    也称为双曲正切核。
    注意：不总是正定的，需要合适的参数。
    
    与神经网络的联系：
    可以看作是两层神经网络的输出。
    """
    
    def __init__(self, alpha: float = 1.0, coef0: float = 0.0):
        self.alpha = alpha
        self.coef0 = coef0
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X
        return np.tanh(self.alpha * (X @ Y.T) + self.coef0)
    
    def is_positive_definite(self) -> bool:
        return False  # 不保证正定


class CompositeKernel(Kernel):
    """
    组合核
    
    通过组合简单核构造复杂核。
    
    有效的组合方式：
    1. 线性组合（系数非负）：k = α₁k₁ + α₂k₂
    2. 乘积：k = k₁ × k₂
    3. 函数复合：k(x,x') = f(x)^T f(x')
    """
    
    def __init__(self, kernels: List[Kernel], 
                 weights: Optional[List[float]] = None,
                 operation: str = 'sum'):
        """
        Args:
            kernels: 核函数列表
            weights: 权重（用于加权和）
            operation: 'sum', 'product'
        """
        self.kernels = kernels
        self.operation = operation
        
        if weights is None:
            self.weights = [1.0] * len(kernels)
        else:
            self.weights = weights
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.operation == 'sum':
            K = np.zeros((len(X), len(Y) if Y is not None else len(X)))
            for kernel, weight in zip(self.kernels, self.weights):
                K += weight * kernel(X, Y)
            return K
        
        elif self.operation == 'product':
            K = np.ones((len(X), len(Y) if Y is not None else len(X)))
            for kernel in self.kernels:
                K *= kernel(X, Y)
            return K
        
        else:
            raise ValueError(f"未知操作: {self.operation}")


def gram_matrix(kernel: Kernel, X: np.ndarray) -> np.ndarray:
    """
    计算Gram矩阵
    
    Gram矩阵K的元素K_ij = k(x_i, x_j)
    
    性质：
    - 对称：K = K^T
    - 半正定：所有特征值 ≥ 0
    
    Args:
        kernel: 核函数
        X: 数据点
        
    Returns:
        Gram矩阵
    """
    return kernel(X, X)


def center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """
    中心化核矩阵
    
    在特征空间中中心化，使得数据均值为0。
    
    K_centered = K - 1_n K - K 1_n + 1_n K 1_n
    
    其中1_n是所有元素为1/n的n×n矩阵。
    
    Args:
        K: 原始核矩阵
        
    Returns:
        中心化的核矩阵
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    return K_centered


def check_positive_definite(K: np.ndarray, tol: float = 1e-6) -> bool:
    """
    检查矩阵是否正定
    
    通过检查所有特征值是否非负。
    
    Args:
        K: 核矩阵
        tol: 容差
        
    Returns:
        是否正定
    """
    # 确保对称
    K = (K + K.T) / 2
    
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(K)
    
    return np.all(eigenvalues >= -tol)


def visualize_kernels(show_plot: bool = True) -> None:
    """
    可视化不同核函数
    
    展示一维输入下不同核函数的形状。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n核函数可视化")
    print("=" * 60)
    
    # 创建一维数据
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    x0 = np.array([[0]])  # 参考点
    
    # 不同的核函数
    kernels = {
        'Linear': LinearKernel(),
        'Polynomial (d=2)': PolynomialKernel(degree=2, gamma=1, coef0=1),
        'RBF (γ=0.5)': RBFKernel(gamma=0.5),
        'RBF (γ=2.0)': RBFKernel(gamma=2.0),
        'Laplacian': LaplacianKernel(gamma=1.0),
        'Sigmoid': SigmoidKernel(alpha=0.5, coef0=0)
    }
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, (name, kernel) in enumerate(kernels.items()):
            ax = axes[idx]
            
            # 计算k(x, 0)
            k_values = kernel(x, x0).ravel()
            
            ax.plot(x.ravel(), k_values, linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('k(x, 0)')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            
            # 标记最大值
            max_idx = np.argmax(k_values)
            ax.plot(x[max_idx], k_values[max_idx], 'ro', markersize=8)
        
        plt.suptitle('不同核函数 k(x, 0) 的形状', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n特点总结：")
    print("1. 线性核：简单内积，随x线性变化")
    print("2. 多项式核：非线性，但全局支撑")
    print("3. RBF核：局部支撑，γ控制宽度")
    print("4. Laplace核：比RBF更尖锐")
    print("5. Sigmoid核：S形曲线，可能非正定")


def visualize_gram_matrix(kernel: Kernel, X: np.ndarray,
                         title: str = "Gram矩阵",
                         show_plot: bool = True) -> None:
    """
    可视化Gram矩阵
    
    展示核矩阵的结构。
    
    Args:
        kernel: 核函数
        X: 数据点
        title: 图标题
        show_plot: 是否显示图形
    """
    K = gram_matrix(kernel, X)
    
    # 检查正定性
    is_pd = check_positive_definite(K)
    print(f"\n{title}")
    print(f"  矩阵大小: {K.shape}")
    print(f"  正定性: {'是' if is_pd else '否'}")
    print(f"  条件数: {np.linalg.cond(K):.2e}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Gram矩阵热图
        ax1 = axes[0]
        im = ax1.imshow(K, cmap='coolwarm', aspect='auto')
        ax1.set_xlabel('样本 j')
        ax1.set_ylabel('样本 i')
        ax1.set_title('Gram矩阵 K[i,j]')
        plt.colorbar(im, ax=ax1)
        
        # 特征值分布
        ax2 = axes[1]
        eigenvalues = np.linalg.eigvalsh(K)
        ax2.plot(eigenvalues, 'o-')
        ax2.set_xlabel('索引')
        ax2.set_ylabel('特征值')
        ax2.set_title('特征值谱')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


def demonstrate_kernel_trick(show_plot: bool = True) -> None:
    """
    演示核技巧
    
    展示如何在高维空间中隐式计算内积。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n核技巧演示")
    print("=" * 60)
    
    # 二维输入数据
    np.random.seed(42)
    X = np.random.randn(5, 2)
    
    print("原始数据 (2维):")
    print(X)
    
    # 显式特征映射：2次多项式
    # φ(x) = [1, x₁, x₂, x₁², x₁x₂, x₂²]
    def explicit_polynomial_features(X):
        n = len(X)
        phi = np.zeros((n, 6))
        phi[:, 0] = 1
        phi[:, 1] = X[:, 0]
        phi[:, 2] = X[:, 1]
        phi[:, 3] = X[:, 0] ** 2
        phi[:, 4] = X[:, 0] * X[:, 1]
        phi[:, 5] = X[:, 1] ** 2
        return phi
    
    # 显式计算
    phi_X = explicit_polynomial_features(X)
    K_explicit = phi_X @ phi_X.T
    
    print(f"\n显式特征映射后 (6维):")
    print(f"  特征维度: {phi_X.shape[1]}")
    
    # 使用核函数
    kernel = PolynomialKernel(degree=2, gamma=1, coef0=1)
    K_kernel = kernel(X)
    
    print(f"\n核函数直接计算:")
    print(f"  计算复杂度: O(n²d) vs O(n²D)")
    print(f"  其中d=2(原始维度), D=6(特征维度)")
    
    # 比较结果
    difference = np.abs(K_explicit - K_kernel).max()
    print(f"\n两种方法的最大差异: {difference:.2e}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 显式计算的Gram矩阵
        ax1 = axes[0]
        im1 = ax1.imshow(K_explicit, cmap='coolwarm')
        ax1.set_title('显式特征映射\nφ(x)ᵀφ(x\')')
        ax1.set_xlabel('样本')
        ax1.set_ylabel('样本')
        plt.colorbar(im1, ax=ax1)
        
        # 核函数计算的Gram矩阵
        ax2 = axes[1]
        im2 = ax2.imshow(K_kernel, cmap='coolwarm')
        ax2.set_title('核函数\nk(x, x\')')
        ax2.set_xlabel('样本')
        ax2.set_ylabel('样本')
        plt.colorbar(im2, ax=ax2)
        
        # 差异
        ax3 = axes[2]
        im3 = ax3.imshow(K_explicit - K_kernel, cmap='coolwarm')
        ax3.set_title('差异')
        ax3.set_xlabel('样本')
        ax3.set_ylabel('样本')
        plt.colorbar(im3, ax=ax3)
        
        plt.suptitle('核技巧：隐式 vs 显式特征映射', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n核技巧的优势：")
    print("1. 避免显式计算高维（甚至无限维）特征")
    print("2. 计算效率高")
    print("3. 存储效率高")
    print("4. 可以处理无限维特征空间（如RBF核）")


def compare_kernel_properties(n_samples: int = 50,
                             show_plot: bool = True) -> None:
    """
    比较不同核的性质
    
    展示不同核函数的特性。
    
    Args:
        n_samples: 样本数
        show_plot: 是否显示图形
    """
    print("\n核函数性质比较")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    
    # 不同的核
    kernels = {
        'Linear': LinearKernel(),
        'Poly(d=3)': PolynomialKernel(degree=3),
        'RBF(γ=0.1)': RBFKernel(gamma=0.1),
        'RBF(γ=10)': RBFKernel(gamma=10),
        'Laplacian': LaplacianKernel(gamma=1.0),
        'Sigmoid': SigmoidKernel()
    }
    
    print("\n核矩阵性质：")
    print("-" * 60)
    print(f"{'核函数':<15} {'正定性':<10} {'条件数':<15} {'秩':<10}")
    print("-" * 60)
    
    gram_matrices = {}
    
    for name, kernel in kernels.items():
        K = gram_matrix(kernel, X)
        gram_matrices[name] = K
        
        is_pd = check_positive_definite(K)
        cond = np.linalg.cond(K)
        rank = np.linalg.matrix_rank(K)
        
        print(f"{name:<15} {'是' if is_pd else '否':<10} {cond:<15.2e} {rank:<10}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, (name, K) in enumerate(gram_matrices.items()):
            ax = axes[idx]
            
            # 绘制Gram矩阵
            im = ax.imshow(K, cmap='coolwarm', aspect='auto')
            ax.set_title(name)
            ax.set_xlabel('样本')
            ax.set_ylabel('样本')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('不同核函数的Gram矩阵', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. RBF核总是正定的")
    print("2. γ大的RBF核矩阵接近对角阵（局部性强）")
    print("3. Sigmoid核可能不正定")
    print("4. 条件数大表示数值稳定性差")