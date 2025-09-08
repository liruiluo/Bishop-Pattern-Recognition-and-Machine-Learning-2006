"""
6.4 高斯过程 (Gaussian Processes)
==================================

高斯过程是函数的概率分布。
它是贝叶斯非参数方法的典型代表。

定义：
高斯过程是一个随机过程，其中任意有限个点的联合分布都是高斯分布。

GP由均值函数和协方差函数（核函数）完全确定：
f(x) ~ GP(m(x), k(x, x'))

其中：
- m(x)：均值函数（通常设为0）
- k(x, x')：协方差函数（核函数）

高斯过程回归：
给定训练数据(X, y)，预测新点x*的分布：
- 均值：μ* = k*ᵀ(K + σ²I)⁻¹y
- 方差：σ*² = k** - k*ᵀ(K + σ²I)⁻¹k*

其中：
- K：训练数据的Gram矩阵
- k*：测试点与训练点的协方差向量
- k**：测试点的自协方差
- σ²：观测噪声方差

优点：
1. 提供不确定性估计
2. 灵活的非参数模型
3. 自动复杂度控制（奥卡姆剃刀）

缺点：
1. 计算复杂度O(n³)
2. 存储复杂度O(n²)
3. 核函数选择

与其他方法的联系：
- 核岭回归的贝叶斯解释
- 无限宽神经网络的极限
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, Dict, List, Union
from scipy.linalg import solve_triangular, cholesky, cho_solve
from scipy.optimize import minimize
from .kernel_functions import Kernel, RBFKernel
import warnings
warnings.filterwarnings('ignore')


class GaussianProcessRegressor:
    """
    高斯过程回归
    
    贝叶斯非参数回归方法。
    通过核函数定义函数的先验分布。
    """
    
    def __init__(self, kernel: Optional[Kernel] = None,
                 noise_variance: float = 1e-3,
                 optimize_hyperparameters: bool = False,
                 n_restarts: int = 5):
        """
        初始化高斯过程
        
        Args:
            kernel: 核函数（协方差函数）
            noise_variance: 观测噪声方差σ²
            optimize_hyperparameters: 是否优化超参数
            n_restarts: 优化重启次数
        """
        self.kernel = kernel or RBFKernel(gamma=1.0)
        self.noise_variance = noise_variance
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_restarts = n_restarts
        
        # 训练数据
        self.X_train = None
        self.y_train = None
        
        # 预计算的矩阵
        self.K = None  # Gram矩阵
        self.L = None  # Cholesky分解
        self.alpha = None  # K^(-1) y
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegressor':
        """
        拟合高斯过程
        
        主要是存储训练数据和预计算一些矩阵。
        
        Args:
            X: 训练输入，shape (n_samples, n_features)
            y: 训练目标，shape (n_samples,)
            
        Returns:
            self
        """
        self.X_train = X
        self.y_train = y.ravel()
        
        # 优化超参数
        if self.optimize_hyperparameters:
            self._optimize_hyperparameters()
        
        # 计算Gram矩阵
        self.K = self.kernel(X, X)
        
        # 添加噪声项（数值稳定性）
        K_noise = self.K + self.noise_variance * np.eye(len(X))
        
        # Cholesky分解：K = L L^T
        # 用于高效求解线性系统
        try:
            self.L = cholesky(K_noise, lower=True)
        except np.linalg.LinAlgError:
            # 添加更多正则化
            K_noise += 1e-6 * np.eye(len(X))
            self.L = cholesky(K_noise, lower=True)
        
        # 计算 alpha = K^(-1) y
        # 通过求解 L L^T alpha = y
        self.alpha = cho_solve((self.L, True), self.y_train)
        
        return self
    
    def predict(self, X: np.ndarray, 
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测
        
        计算后验分布的均值和方差。
        
        均值：μ* = k*ᵀ α
        方差：σ*² = k** - v^T v
        其中 v = L^(-1) k*
        
        Args:
            X: 测试输入，shape (n_samples, n_features)
            return_std: 是否返回标准差
            
        Returns:
            预测均值，或(均值, 标准差)
        """
        if self.X_train is None:
            raise RuntimeError("需要先调用fit方法")
        
        # 计算测试点与训练点的协方差
        K_star = self.kernel(X, self.X_train)
        
        # 预测均值
        y_mean = K_star @ self.alpha
        
        if return_std:
            # 计算预测方差
            # v = L^(-1) k*
            v = solve_triangular(self.L, K_star.T, lower=True)
            
            # 测试点的自协方差
            K_star_star = self.kernel(X, X)
            
            # 方差：σ*² = k** - v^T v
            y_var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
            
            # 确保方差非负
            y_var = np.maximum(y_var, 0)
            
            y_std = np.sqrt(y_var + self.noise_variance)
            
            return y_mean, y_std
        
        return y_mean
    
    def sample_prior(self, X: np.ndarray, n_samples: int = 5) -> np.ndarray:
        """
        从先验分布采样
        
        展示没有数据时的函数分布。
        
        Args:
            X: 采样点
            n_samples: 采样数量
            
        Returns:
            采样的函数值，shape (n_samples, n_points)
        """
        # 计算协方差矩阵
        K = self.kernel(X, X)
        
        # 添加小的正则化
        K += 1e-10 * np.eye(len(X))
        
        # Cholesky分解
        L = cholesky(K, lower=True)
        
        # 采样：f = L z，其中z ~ N(0, I)
        samples = np.zeros((n_samples, len(X)))
        for i in range(n_samples):
            z = np.random.randn(len(X))
            samples[i] = L @ z
        
        return samples
    
    def sample_posterior(self, X: np.ndarray, n_samples: int = 5) -> np.ndarray:
        """
        从后验分布采样
        
        展示给定数据后的函数分布。
        
        Args:
            X: 采样点
            n_samples: 采样数量
            
        Returns:
            采样的函数值
        """
        if self.X_train is None:
            return self.sample_prior(X, n_samples)
        
        # 预测均值和协方差
        mean, std = self.predict(X, return_std=True)
        
        # 计算条件协方差
        K_star = self.kernel(X, self.X_train)
        K_star_star = self.kernel(X, X)
        
        v = solve_triangular(self.L, K_star.T, lower=True)
        cov = K_star_star - v.T @ v
        
        # 确保对称和正定
        cov = (cov + cov.T) / 2
        cov += 1e-6 * np.eye(len(X))
        
        # Cholesky分解
        L_cond = cholesky(cov, lower=True)
        
        # 采样
        samples = np.zeros((n_samples, len(X)))
        for i in range(n_samples):
            z = np.random.randn(len(X))
            samples[i] = mean + L_cond @ z
        
        return samples
    
    def log_marginal_likelihood(self) -> float:
        """
        计算对数边际似然
        
        log p(y|X) = -0.5 * [y^T K^(-1) y + log|K| + n log(2π)]
        
        用于超参数优化和模型选择。
        
        Returns:
            对数边际似然
        """
        if self.X_train is None:
            raise RuntimeError("需要先调用fit方法")
        
        n = len(self.y_train)
        
        # 第一项：y^T K^(-1) y = y^T alpha
        term1 = 0.5 * self.y_train @ self.alpha
        
        # 第二项：log|K| = 2 * sum(log(diag(L)))
        term2 = np.sum(np.log(np.diag(self.L)))
        
        # 第三项：常数项
        term3 = 0.5 * n * np.log(2 * np.pi)
        
        return -(term1 + term2 + term3)
    
    def _optimize_hyperparameters(self) -> None:
        """
        优化超参数
        
        通过最大化边际似然来选择最优超参数。
        这实现了自动相关性确定（ARD）。
        """
        print("优化超参数...")
        
        # 保存原始数据
        X_train = self.X_train
        y_train = self.y_train
        
        # 定义目标函数（负对数边际似然）
        def objective(params):
            # 更新超参数
            if isinstance(self.kernel, RBFKernel):
                gamma = np.exp(params[0])
                noise_var = np.exp(params[1])
            else:
                gamma = None
                noise_var = np.exp(params[0])
            
            # 直接计算边际似然，不调用fit
            if gamma is not None:
                # 临时更新kernel参数
                old_gamma = self.kernel.gamma
                self.kernel.gamma = gamma
                K = self.kernel(X_train, X_train)
                self.kernel.gamma = old_gamma  # 恢复原值
            else:
                K = self.kernel(X_train, X_train)
            
            # 添加噪声项
            K_noise = K + noise_var * np.eye(len(X_train))
            
            # Cholesky分解
            try:
                L = cholesky(K_noise, lower=True)
            except np.linalg.LinAlgError:
                return 1e10  # 返回大的惩罚值
            
            # 计算 alpha = K^(-1) y
            alpha = cho_solve((L, True), y_train)
            
            # 计算负对数边际似然
            n = len(y_train)
            term1 = 0.5 * y_train @ alpha
            term2 = np.sum(np.log(np.diag(L)))
            term3 = 0.5 * n * np.log(2 * np.pi)
            
            return term1 + term2 + term3
        
        # 初始参数
        if isinstance(self.kernel, RBFKernel):
            x0 = [np.log(self.kernel.gamma), np.log(self.noise_variance)]
        else:
            x0 = [np.log(self.noise_variance)]
        
        # 多次重启优化
        best_params = None
        best_value = np.inf
        
        for _ in range(self.n_restarts):
            # 随机初始化
            x0_random = x0 + 0.5 * np.random.randn(len(x0))
            
            # 优化
            result = minimize(objective, x0_random, method='L-BFGS-B')
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x
        
        # 设置最优参数
        if isinstance(self.kernel, RBFKernel):
            self.kernel.gamma = np.exp(best_params[0])
            self.noise_variance = np.exp(best_params[1])
            print(f"  最优参数: γ={self.kernel.gamma:.4f}, σ²={self.noise_variance:.4f}")
        else:
            self.noise_variance = np.exp(best_params[0])
            print(f"  最优噪声方差: σ²={self.noise_variance:.4f}")


def demonstrate_gaussian_process(show_plot: bool = True) -> None:
    """
    演示高斯过程回归
    
    展示GP的关键特性：
    1. 先验和后验
    2. 不确定性量化
    3. 数据点的影响
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n高斯过程回归演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    # 真实函数
    def true_function(x):
        return np.sin(3 * x) * np.exp(-0.5 * x)
    
    # 训练数据（少量）
    n_train = 7
    X_train = np.random.uniform(-3, 3, n_train).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + 0.1 * np.random.randn(n_train)
    
    # 测试数据
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # 创建高斯过程
    kernel = RBFKernel(gamma=0.5)
    gp = GaussianProcessRegressor(kernel=kernel, noise_variance=0.01)
    
    print("训练高斯过程...")
    print(f"  训练样本数: {n_train}")
    print(f"  核函数: RBF (γ=0.5)")
    print(f"  噪声方差: 0.01")
    
    # 拟合
    gp.fit(X_train, y_train)
    
    # 预测
    y_mean, y_std = gp.predict(X_test, return_std=True)
    
    # 计算对数边际似然
    log_ml = gp.log_marginal_likelihood()
    print(f"  对数边际似然: {log_ml:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 先验采样
        ax1 = axes[0, 0]
        prior_samples = gp.sample_prior(X_test, n_samples=5)
        for sample in prior_samples:
            ax1.plot(X_test, sample, alpha=0.5, linewidth=1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('先验分布（无数据）')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-3, 3])
        
        # 后验均值和不确定性
        ax2 = axes[0, 1]
        ax2.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        ax2.plot(X_test, y_mean, 'r-', label='GP均值', linewidth=2)
        ax2.fill_between(X_test.ravel(),
                         y_mean - 2*y_std,
                         y_mean + 2*y_std,
                         alpha=0.3, color='red',
                         label='95%置信区间')
        ax2.scatter(X_train, y_train, c='blue', s=50, 
                   zorder=5, label='训练数据')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('后验分布（有数据）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 后验采样
        ax3 = axes[1, 0]
        posterior_samples = gp.sample_posterior(X_test, n_samples=5)
        for sample in posterior_samples:
            ax3.plot(X_test, sample, alpha=0.5, linewidth=1)
        ax3.scatter(X_train, y_train, c='blue', s=50, zorder=5)
        ax3.set_xlabel('x')
        ax3.set_ylabel('f(x)')
        ax3.set_title('后验采样')
        ax3.grid(True, alpha=0.3)
        
        # 不确定性
        ax4 = axes[1, 1]
        ax4.plot(X_test, y_std, 'b-', linewidth=2)
        ax4.scatter(X_train, np.zeros(n_train), c='red', s=50, 
                   marker='v', label='训练点位置')
        ax4.set_xlabel('x')
        ax4.set_ylabel('标准差')
        ax4.set_title('预测不确定性')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('高斯过程回归', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 数据点处不确定性最小（插值）")
    print("2. 远离数据的地方不确定性增大")
    print("3. 后验采样都通过训练数据附近")
    print("4. GP自动平衡拟合和平滑")


def compare_gp_kernels(show_plot: bool = True) -> None:
    """
    比较不同核函数的高斯过程
    
    展示核函数如何影响GP的行为。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n不同核函数的高斯过程")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X_train = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)
    y_train = np.array([0.5, -0.5, 1.0, -0.5, 0.5])
    
    X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    
    # 不同的核函数
    from .kernel_functions import LinearKernel, PolynomialKernel, LaplacianKernel
    
    kernels = {
        'Linear': LinearKernel(),
        'Polynomial': PolynomialKernel(degree=3, gamma=1, coef0=1),
        'RBF (γ=0.5)': RBFKernel(gamma=0.5),
        'RBF (γ=5.0)': RBFKernel(gamma=5.0),
        'Laplacian': LaplacianKernel(gamma=1.0)
    }
    
    results = {}
    
    for name, kernel in kernels.items():
        gp = GaussianProcessRegressor(kernel=kernel, noise_variance=0.01)
        gp.fit(X_train, y_train)
        y_mean, y_std = gp.predict(X_test, return_std=True)
        
        results[name] = {
            'mean': y_mean,
            'std': y_std,
            'log_ml': gp.log_marginal_likelihood()
        }
        
        print(f"{name:15} - log p(y|X) = {results[name]['log_ml']:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            if idx >= 5:
                break
            
            ax = axes[idx]
            
            # 绘制预测
            ax.plot(X_test, result['mean'], 'r-', linewidth=2, label='GP均值')
            ax.fill_between(X_test.ravel(),
                           result['mean'] - 2*result['std'],
                           result['mean'] + 2*result['std'],
                           alpha=0.3, color='red')
            ax.scatter(X_train, y_train, c='blue', s=50, zorder=5)
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'{name}\nlog ML = {result["log_ml"]:.2f}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-2, 2])
        
        # 移除多余的子图
        if len(results) < 6:
            fig.delaxes(axes[5])
        
        plt.suptitle('不同核函数的高斯过程', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 线性核：只能拟合线性函数")
    print("2. 多项式核：全局影响，可能振荡")
    print("3. RBF核：局部影响，γ控制平滑度")
    print("4. Laplace核：比RBF更尖锐的转折")
    print("5. 边际似然可用于模型选择")


def demonstrate_gp_hyperparameter_optimization(show_plot: bool = True) -> None:
    """
    演示高斯过程超参数优化
    
    展示如何通过最大化边际似然自动选择超参数。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n高斯过程超参数优化")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    n_train = 10
    X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + 0.05 * np.random.randn(n_train)
    
    X_test = np.linspace(-0.2, 1.2, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # 创建两个GP：一个固定参数，一个优化参数
    print("1. 固定超参数的GP:")
    gp_fixed = GaussianProcessRegressor(
        kernel=RBFKernel(gamma=1.0),
        noise_variance=0.1,
        optimize_hyperparameters=False
    )
    gp_fixed.fit(X_train, y_train)
    y_mean_fixed, y_std_fixed = gp_fixed.predict(X_test, return_std=True)
    log_ml_fixed = gp_fixed.log_marginal_likelihood()
    print(f"   γ=1.0, σ²=0.1")
    print(f"   log p(y|X) = {log_ml_fixed:.2f}")
    
    print("\n2. 优化超参数的GP:")
    gp_optimized = GaussianProcessRegressor(
        kernel=RBFKernel(gamma=1.0),
        noise_variance=0.1,
        optimize_hyperparameters=True,
        n_restarts=5
    )
    gp_optimized.fit(X_train, y_train)
    y_mean_opt, y_std_opt = gp_optimized.predict(X_test, return_std=True)
    log_ml_opt = gp_optimized.log_marginal_likelihood()
    print(f"   log p(y|X) = {log_ml_opt:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 固定超参数
        ax1 = axes[0]
        ax1.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        ax1.plot(X_test, y_mean_fixed, 'r-', label='GP均值', linewidth=2)
        ax1.fill_between(X_test.ravel(),
                        y_mean_fixed - 2*y_std_fixed,
                        y_mean_fixed + 2*y_std_fixed,
                        alpha=0.3, color='red')
        ax1.scatter(X_train, y_train, c='blue', s=50, zorder=5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'固定超参数\nlog ML = {log_ml_fixed:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 优化超参数
        ax2 = axes[1]
        ax2.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        ax2.plot(X_test, y_mean_opt, 'r-', label='GP均值', linewidth=2)
        ax2.fill_between(X_test.ravel(),
                        y_mean_opt - 2*y_std_opt,
                        y_mean_opt + 2*y_std_opt,
                        alpha=0.3, color='red')
        ax2.scatter(X_train, y_train, c='blue', s=50, zorder=5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title(f'优化超参数\nlog ML = {log_ml_opt:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('超参数优化的效果', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 优化后的边际似然更高")
    print("2. 优化后的预测更准确")
    print("3. 自动调整了函数的平滑度")
    print("4. 自动调整了噪声水平")