"""
Chapter 6: Kernel Methods (核方法)
==================================

本章介绍核方法，这是机器学习中的重要技术。

主要内容：
1. 对偶表示 (6.1)
   - 将线性模型转换为对偶形式
   - 预测只依赖于内积

2. 构建核 (6.2)
   - 核函数的性质
   - 常用核函数
   - 核的组合

3. 径向基函数网络 (6.3)
   - RBF网络
   - Nadaraya-Watson模型

4. 高斯过程 (6.4)
   - 函数的贝叶斯处理
   - GP回归
   - GP分类

核心思想：
核技巧允许我们在高维（甚至无限维）特征空间中
隐式地进行计算，而不需要显式的特征映射。

这使得：
1. 非线性模型的构建变得简单
2. 计算效率大大提高
3. 可以处理结构化数据（字符串、图等）

高斯过程是核方法的贝叶斯视角，
提供了原理性的不确定性量化。

本章为理解SVM、核PCA等方法奠定基础。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .kernel_functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    LaplacianKernel,
    SigmoidKernel,
    CompositeKernel,
    gram_matrix,
    center_kernel_matrix,
    check_positive_definite,
    visualize_kernels,
    visualize_gram_matrix,
    demonstrate_kernel_trick,
    compare_kernel_properties
)

from .gaussian_processes import (
    GaussianProcessRegressor,
    demonstrate_gaussian_process,
    compare_gp_kernels,
    demonstrate_gp_hyperparameter_optimization
)


def run_chapter06(cfg: DictConfig) -> None:
    """
    运行第6章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第6章：核方法 (Kernel Methods)")
    print("="*80)
    
    # 6.1-6.2 核函数
    print("\n" + "-"*60)
    print("6.1-6.2 核函数与核技巧")
    print("-"*60)
    
    # 可视化核函数
    visualize_kernels(
        show_plot=cfg.visualization.show_plots
    )
    
    # 演示核技巧
    demonstrate_kernel_trick(
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较核的性质
    compare_kernel_properties(
        n_samples=50,
        show_plot=cfg.visualization.show_plots
    )
    
    # 可视化Gram矩阵
    if cfg.visualization.show_plots:
        print("\nGram矩阵可视化")
        np.random.seed(42)
        X = np.random.randn(30, 2)
        
        # RBF核的Gram矩阵
        rbf_kernel = RBFKernel(gamma=1.0)
        visualize_gram_matrix(rbf_kernel, X, "RBF核的Gram矩阵", 
                            show_plot=True)
        
        # 多项式核的Gram矩阵
        poly_kernel = PolynomialKernel(degree=3)
        visualize_gram_matrix(poly_kernel, X, "多项式核的Gram矩阵",
                            show_plot=True)
    
    # 6.3 径向基函数网络
    print("\n" + "-"*60)
    print("6.3 径向基函数网络")
    print("-"*60)
    
    demonstrate_rbf_network(
        show_plot=cfg.visualization.show_plots
    )
    
    # 6.4 高斯过程
    print("\n" + "-"*60)
    print("6.4 高斯过程 (Gaussian Processes)")
    print("-"*60)
    
    # 基本GP演示
    demonstrate_gaussian_process(
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较不同核的GP
    compare_gp_kernels(
        show_plot=cfg.visualization.show_plots
    )
    
    # 超参数优化
    demonstrate_gp_hyperparameter_optimization(
        show_plot=cfg.visualization.show_plots
    )
    
    # 核方法应用示例
    print("\n" + "-"*60)
    print("核方法应用")
    print("-"*60)
    
    demonstrate_kernel_ridge_regression(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第6章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 核技巧允许在高维空间隐式计算")
    print("2. 不同核函数对应不同的特征空间")
    print("3. Mercer定理保证核的有效性")
    print("4. 高斯过程提供贝叶斯非参数方法")
    print("5. GP自动提供不确定性量化")
    print("6. 核方法是SVM等算法的基础")


def demonstrate_rbf_network(show_plot: bool = True) -> None:
    """
    演示径向基函数网络
    
    RBF网络是使用径向基函数作为激活函数的神经网络。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n径向基函数网络演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    def true_function(x):
        return np.sin(3 * x) + 0.5 * np.cos(5 * x)
    
    # 训练数据
    n_train = 50
    X_train = np.random.uniform(-2, 2, n_train).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + 0.1 * np.random.randn(n_train)
    
    # 测试数据
    X_test = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # RBF网络（简化版）
    class SimpleRBFNetwork:
        def __init__(self, n_centers: int = 10, width: float = 0.3):
            self.n_centers = n_centers
            self.width = width
            self.centers = None
            self.weights = None
            
        def _select_centers(self, X):
            """使用K-means选择中心"""
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_
            
        def _compute_features(self, X):
            """计算RBF特征"""
            # 计算到各中心的距离
            from scipy.spatial.distance import cdist
            distances = cdist(X, self.centers)
            # RBF激活
            features = np.exp(-distances**2 / (2 * self.width**2))
            # 添加偏置
            features = np.column_stack([np.ones(len(X)), features])
            return features
            
        def fit(self, X, y):
            # 选择中心
            self._select_centers(X)
            # 计算特征
            Phi = self._compute_features(X)
            # 最小二乘求解权重
            self.weights = np.linalg.lstsq(Phi, y, rcond=None)[0]
            return self
            
        def predict(self, X):
            Phi = self._compute_features(X)
            return Phi @ self.weights
    
    # 训练RBF网络
    rbf_net = SimpleRBFNetwork(n_centers=15, width=0.3)
    rbf_net.fit(X_train, y_train)
    y_pred_rbf = rbf_net.predict(X_test)
    
    # 使用高斯过程比较
    from .gaussian_processes import GaussianProcessRegressor
    gp = GaussianProcessRegressor(kernel=RBFKernel(gamma=1.0), noise_variance=0.01)
    gp.fit(X_train, y_train)
    y_pred_gp, y_std_gp = gp.predict(X_test, return_std=True)
    
    # 计算误差
    mse_rbf = np.mean((y_pred_rbf - y_true)**2)
    mse_gp = np.mean((y_pred_gp - y_true)**2)
    
    print(f"RBF网络：")
    print(f"  中心数: {rbf_net.n_centers}")
    print(f"  宽度: {rbf_net.width}")
    print(f"  测试MSE: {mse_rbf:.4f}")
    
    print(f"\n高斯过程：")
    print(f"  测试MSE: {mse_gp:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RBF网络
        ax1 = axes[0]
        ax1.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        ax1.plot(X_test, y_pred_rbf, 'r-', label='RBF网络', linewidth=2)
        ax1.scatter(X_train, y_train, c='blue', s=20, alpha=0.5)
        # 显示中心
        ax1.scatter(rbf_net.centers, np.zeros(rbf_net.n_centers), 
                   c='red', s=100, marker='^', label='RBF中心')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'RBF网络 (MSE={mse_rbf:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 高斯过程
        ax2 = axes[1]
        ax2.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        ax2.plot(X_test, y_pred_gp, 'r-', label='GP均值', linewidth=2)
        ax2.fill_between(X_test.ravel(),
                        y_pred_gp - 2*y_std_gp,
                        y_pred_gp + 2*y_std_gp,
                        alpha=0.3, color='red')
        ax2.scatter(X_train, y_train, c='blue', s=20, alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title(f'高斯过程 (MSE={mse_gp:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('RBF网络 vs 高斯过程', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. RBF网络需要选择中心位置")
    print("2. GP自动确定'等效中心'")
    print("3. GP提供不确定性估计")
    print("4. 两者都使用径向基函数")


def demonstrate_kernel_ridge_regression(show_plot: bool = True) -> None:
    """
    演示核岭回归
    
    核岭回归是岭回归的核化版本。
    
    预测函数：f(x) = Σᵢ αᵢ k(x, xᵢ)
    其中 α = (K + λI)⁻¹ y
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n核岭回归演示")
    print("=" * 60)
    
    # 生成非线性数据
    np.random.seed(42)
    n_train = 30
    X_train = np.random.uniform(-3, 3, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(n_train)
    
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()
    
    # 核岭回归类
    class KernelRidgeRegression:
        def __init__(self, kernel: Kernel, lambda_reg: float = 0.1):
            self.kernel = kernel
            self.lambda_reg = lambda_reg
            self.X_train = None
            self.alpha = None
            
        def fit(self, X, y):
            self.X_train = X
            # 计算Gram矩阵
            K = self.kernel(X, X)
            # 求解 α = (K + λI)⁻¹ y
            n = len(X)
            self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(n), y)
            return self
            
        def predict(self, X):
            # f(x) = Σᵢ αᵢ k(x, xᵢ)
            K_test = self.kernel(X, self.X_train)
            return K_test @ self.alpha
    
    # 不同的正则化强度
    lambdas = [0.001, 0.01, 0.1, 1.0]
    
    results = {}
    
    for lam in lambdas:
        krr = KernelRidgeRegression(
            kernel=RBFKernel(gamma=1.0),
            lambda_reg=lam
        )
        krr.fit(X_train, y_train)
        y_pred = krr.predict(X_test)
        
        mse = np.mean((y_pred - y_true)**2)
        results[lam] = {
            'y_pred': y_pred,
            'mse': mse
        }
        
        print(f"λ = {lam:6.3f}: MSE = {mse:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, lam in enumerate(lambdas):
            ax = axes[idx]
            
            ax.plot(X_test, y_true, 'g-', label='真实函数', 
                   linewidth=2, alpha=0.5)
            ax.plot(X_test, results[lam]['y_pred'], 'r-', 
                   label=f'预测 (λ={lam})', linewidth=2)
            ax.scatter(X_train, y_train, c='blue', s=30, alpha=0.5)
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'λ = {lam}, MSE = {results[lam]["mse"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1.5, 1.5])
        
        plt.suptitle('核岭回归：正则化的影响', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. λ小：过拟合，函数振荡")
    print("2. λ大：欠拟合，过度平滑")
    print("3. 合适的λ平衡拟合和泛化")
    print("4. 核岭回归等价于具有特定先验的GP")