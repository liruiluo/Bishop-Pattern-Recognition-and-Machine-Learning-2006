"""
2.3 高斯分布 (The Gaussian Distribution)
=========================================

高斯分布（正态分布）是机器学习中最重要的连续分布。

为什么高斯分布如此重要？
1. 中心极限定理：大量独立随机变量的和趋向于正态分布
2. 最大熵原理：在给定均值和方差下，高斯分布有最大熵
3. 数学性质优美：线性变换后仍是高斯分布
4. 计算方便：许多操作有解析解

本节内容：
1. 一维高斯分布
2. 多维高斯分布
3. 最大似然估计
4. 贝叶斯推断
5. 高斯混合模型（GMM）预览
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')


class UnivariateGaussian:
    """
    一维高斯分布
    
    N(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    
    其中：
    - μ: 均值（位置参数）
    - σ²: 方差（尺度参数）
    - σ: 标准差
    
    性质：
    - 期望：E[x] = μ
    - 方差：Var[x] = σ²
    - 众数：mode = μ
    - 熵：H = (1/2)log(2πeσ²)
    
    精度参数化：
    有时使用精度λ = 1/σ²更方便：
    N(x|μ,λ⁻¹) = √(λ/(2π)) * exp(-(λ/2)(x-μ)²)
    """
    
    def __init__(self, mu: float = 0.0, sigma2: float = 1.0, 
                 use_precision: bool = False):
        """
        初始化高斯分布
        
        Args:
            mu: 均值
            sigma2: 方差（如果use_precision=True，则为精度）
            use_precision: 是否使用精度参数化
        """
        self.mu = mu
        
        if use_precision:
            # sigma2实际上是精度λ
            if sigma2 <= 0:
                raise ValueError(f"精度必须为正数，得到{sigma2}")
            self.precision = sigma2
            self.sigma2 = 1.0 / sigma2
        else:
            if sigma2 <= 0:
                raise ValueError(f"方差必须为正数，得到{sigma2}")
            self.sigma2 = sigma2
            self.precision = 1.0 / sigma2
        
        self.sigma = np.sqrt(self.sigma2)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        概率密度函数
        
        使用对数技巧避免数值下溢：
        log p(x) = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
        """
        x = np.asarray(x)
        # 使用scipy的实现
        return stats.norm.pdf(x, self.mu, self.sigma)
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """对数概率密度"""
        x = np.asarray(x)
        return stats.norm.logpdf(x, self.mu, self.sigma)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """累积分布函数"""
        x = np.asarray(x)
        return stats.norm.cdf(x, self.mu, self.sigma)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """采样"""
        return np.random.normal(self.mu, self.sigma, size)
    
    def entropy(self) -> float:
        """
        熵
        H = (1/2)log(2πeσ²)
        """
        return 0.5 * np.log(2 * np.pi * np.e * self.sigma2)
    
    @staticmethod
    def mle(data: np.ndarray) -> 'UnivariateGaussian':
        """
        最大似然估计
        
        给定数据，估计高斯分布的参数：
        μ_ML = (1/N)Σx_i
        σ²_ML = (1/N)Σ(x_i - μ_ML)²
        
        注意：方差的MLE是有偏的（低估了真实方差）
        无偏估计应该使用N-1而不是N
        
        Args:
            data: 观测数据
            
        Returns:
            估计的高斯分布
        """
        data = np.asarray(data)
        mu_ml = np.mean(data)
        sigma2_ml = np.mean((data - mu_ml) ** 2)
        return UnivariateGaussian(mu_ml, sigma2_ml)
    
    @staticmethod
    def mle_sequential(data: np.ndarray) -> List[Tuple[float, float]]:
        """
        顺序最大似然估计
        
        展示参数估计如何随数据增加而演化
        
        Returns:
            每个数据点后的(μ, σ²)估计
        """
        estimates = []
        for n in range(1, len(data) + 1):
            subset = data[:n]
            mu = np.mean(subset)
            sigma2 = np.var(subset)
            estimates.append((mu, sigma2))
        return estimates


class MultivariateGaussian:
    """
    多维高斯分布
    
    N(x|μ,Σ) = (1/((2π)^(D/2)||Σ||^(1/2))) * exp(-(1/2)(x-μ)^TΣ^(-1)(x-μ))
    
    其中：
    - x: D维向量
    - μ: D维均值向量
    - Σ: D×D协方差矩阵（必须是对称正定的）
    - ||Σ||: 协方差矩阵的行列式
    
    精度矩阵参数化：
    Λ = Σ^(-1) 是精度矩阵
    
    协方差矩阵的类型：
    1. 球形（Spherical）：Σ = σ²I，各方向方差相同
    2. 对角（Diagonal）：Σ = diag(σ²_1,...,σ²_D)，轴对齐
    3. 完全（Full）：任意对称正定矩阵
    
    性质：
    - 期望：E[x] = μ
    - 协方差：Cov[x] = Σ
    - 熵：H = (D/2)(1 + log(2π)) + (1/2)log||Σ||
    - 边缘分布和条件分布仍是高斯分布
    """
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, 
                 covariance_type: str = 'full'):
        """
        初始化多维高斯分布
        
        Args:
            mu: 均值向量
            sigma: 协方差矩阵或方差向量（取决于covariance_type）
            covariance_type: 'full', 'diagonal', 'spherical'
        """
        self.mu = np.asarray(mu)
        self.D = len(self.mu)  # 维度
        self.covariance_type = covariance_type
        
        # 根据类型构建协方差矩阵
        if covariance_type == 'spherical':
            # 球形：单一标量方差
            if np.isscalar(sigma):
                self.sigma = sigma * np.eye(self.D)
            else:
                raise ValueError("球形协方差需要标量参数")
        elif covariance_type == 'diagonal':
            # 对角：方差向量
            if len(sigma) == self.D:
                self.sigma = np.diag(sigma)
            else:
                raise ValueError("对角协方差需要D维向量")
        else:  # full
            # 完全协方差矩阵
            self.sigma = np.asarray(sigma)
            if self.sigma.shape != (self.D, self.D):
                raise ValueError(f"协方差矩阵形状必须为({self.D}, {self.D})")
        
        # 检查对称性和正定性
        if not np.allclose(self.sigma, self.sigma.T):
            raise ValueError("协方差矩阵必须对称")
        
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        if np.min(eigenvalues) <= 0:
            raise ValueError("协方差矩阵必须正定")
        
        # 计算精度矩阵（逆矩阵）
        self.precision = np.linalg.inv(self.sigma)
        
        # 计算行列式的对数（用于pdf计算）
        self.log_det_sigma = np.linalg.slogdet(self.sigma)[1]
    
    def pdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        概率密度函数
        
        Args:
            x: 输入点，shape (D,) 或 (N, D)
            
        Returns:
            概率密度值
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # 使用scipy的实现
        return stats.multivariate_normal.pdf(x, self.mu, self.sigma)
    
    def log_pdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        对数概率密度
        
        log p(x) = -D/2*log(2π) - 1/2*log||Σ|| - 1/2*(x-μ)^TΣ^(-1)(x-μ)
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        return stats.multivariate_normal.logpdf(x, self.mu, self.sigma)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        采样
        
        Returns:
            样本，shape (size, D)
        """
        return np.random.multivariate_normal(self.mu, self.sigma, size)
    
    def mahalanobis_distance(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        计算马氏距离
        
        马氏距离考虑了协方差结构：
        Δ = √((x-μ)^TΣ^(-1)(x-μ))
        
        这是一种“标准化”的距离，在异常检测中很有用。
        
        Args:
            x: 输入点
            
        Returns:
            马氏距离
        """
        x = np.asarray(x)
        if x.ndim == 1:
            diff = x - self.mu
            return np.sqrt(diff @ self.precision @ diff)
        else:
            diff = x - self.mu
            return np.sqrt(np.sum(diff @ self.precision * diff, axis=1))
    
    @staticmethod
    def mle(data: np.ndarray, covariance_type: str = 'full') -> 'MultivariateGaussian':
        """
        最大似然估计
        
        μ_ML = (1/N)Σx_i
        Σ_ML = (1/N)Σ(x_i - μ_ML)(x_i - μ_ML)^T
        
        Args:
            data: 观测数据，shape (N, D)
            covariance_type: 协方差类型
            
        Returns:
            估计的高斯分布
        """
        data = np.asarray(data)
        N, D = data.shape
        
        # 估计均值
        mu_ml = np.mean(data, axis=0)
        
        # 估计协方差
        centered = data - mu_ml
        
        if covariance_type == 'spherical':
            # 球形：单一方差
            sigma_ml = np.mean(centered ** 2)
        elif covariance_type == 'diagonal':
            # 对角：各维独立方差
            sigma_ml = np.mean(centered ** 2, axis=0)
        else:  # full
            # 完全协方差
            sigma_ml = (centered.T @ centered) / N
        
        return MultivariateGaussian(mu_ml, sigma_ml, covariance_type)


def demonstrate_univariate_gaussian(mean_values: List[float],
                                  variance_values: List[float],
                                  n_samples: int = 1000,
                                  show_plot: bool = True) -> None:
    """
    演示一维高斯分布的性质
    
    展示均值和方差如何影响分布形状：
    - 均值控制位置
    - 方差控制分散程度
    """
    print("\n一维高斯分布演示")
    print("=" * 60)
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1：不同均值
        ax1 = axes[0, 0]
        x_range = np.linspace(-8, 8, 200)
        for mu in mean_values:
            dist = UnivariateGaussian(mu, 1.0)
            pdf_values = dist.pdf(x_range)
            ax1.plot(x_range, pdf_values, linewidth=2, 
                    label=f'μ={mu}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('概率密度')
        ax1.set_title('不同均值的高斯分布 (σ²=1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：不同方差
        ax2 = axes[0, 1]
        for sigma2 in variance_values:
            dist = UnivariateGaussian(0, sigma2)
            pdf_values = dist.pdf(x_range)
            ax2.plot(x_range, pdf_values, linewidth=2,
                    label=f'σ²={sigma2}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('概率密度')
        ax2.set_title('不同方差的高斯分布 (μ=0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：累积分布函数
        ax3 = axes[1, 0]
        for sigma2 in variance_values:
            dist = UnivariateGaussian(0, sigma2)
            cdf_values = dist.cdf(x_range)
            ax3.plot(x_range, cdf_values, linewidth=2,
                    label=f'σ²={sigma2}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('累积概率')
        ax3.set_title('累积分布函数 CDF')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # 子图4：采样和直方图
        ax4 = axes[1, 1]
        dist = UnivariateGaussian(0, 1)
        samples = dist.sample(n_samples)
        ax4.hist(samples, bins=50, density=True, alpha=0.7,
                color='blue', edgecolor='black', label='采样直方图')
        pdf_values = dist.pdf(x_range)
        ax4.plot(x_range, pdf_values, 'r-', linewidth=2,
                label='理论密度')
        ax4.set_xlabel('x')
        ax4.set_ylabel('概率密度')
        ax4.set_title(f'N(0,1)的{n_samples}个样本')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('一维高斯分布的性质', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 打印统计信息
    print("\n标准正态分布N(0,1)的重要数值：")
    print("-" * 40)
    dist = UnivariateGaussian(0, 1)
    print(f"P(|X| ≤ 1σ) = {dist.cdf(1) - dist.cdf(-1):.4f} (约68.3%)")
    print(f"P(|X| ≤ 2σ) = {dist.cdf(2) - dist.cdf(-2):.4f} (约95.5%)")
    print(f"P(|X| ≤ 3σ) = {dist.cdf(3) - dist.cdf(-3):.4f} (约99.7%)")
    print(f"熵 H = {dist.entropy():.4f}")


def demonstrate_multivariate_gaussian(dimension: int = 2,
                                    covariance_types: List[str] = ['spherical', 'diagonal', 'full'],
                                    n_samples: int = 500,
                                    show_plot: bool = True) -> None:
    """
    演示二维高斯分布的不同协方差结构
    
    协方差矩阵决定了分布的形状和方向：
    - 球形：圆形等高线
    - 对角：轴对齐的椭圆
    - 完全：任意方向的椭圆
    """
    if dimension != 2:
        print("可视化只支持2维情况")
        return
    
    print("\n二维高斯分布的协方差结构")
    print("=" * 60)
    
    if show_plot:
        fig, axes = plt.subplots(1, len(covariance_types), 
                                figsize=(5*len(covariance_types), 5))
        
        mu = np.array([0, 0])
        
        for idx, (cov_type, ax) in enumerate(zip(covariance_types, axes)):
            # 构建协方差矩阵
            if cov_type == 'spherical':
                sigma = 0.5  # 单一方差
                title = '球形协方差\nΣ = σ²I'
            elif cov_type == 'diagonal':
                sigma = np.array([1.0, 0.3])  # 不同方向的方差
                title = '对角协方差\nΣ = diag(σ²ᵢ)'
            else:  # full
                sigma = np.array([[1.0, 0.5],
                                 [0.5, 0.6]])  # 完全协方差
                title = '完全协方差\n有相关性'
            
            # 创建分布
            dist = MultivariateGaussian(mu, sigma, cov_type)
            
            # 采样
            samples = dist.sample(n_samples)
            
            # 绘制散点图
            ax.scatter(samples[:, 0], samples[:, 1], 
                      alpha=0.5, s=10, c='blue')
            
            # 绘制等高线
            x_range = np.linspace(-4, 4, 100)
            y_range = np.linspace(-4, 4, 100)
            X, Y = np.meshgrid(x_range, y_range)
            pos = np.dstack((X, Y))
            
            # 计算概率密度
            pdf_values = dist.pdf(pos.reshape(-1, 2)).reshape(X.shape)
            
            # 绘制等高线
            contours = ax.contour(X, Y, pdf_values, levels=5, 
                                 colors='red', linewidths=2)
            ax.clabel(contours, inline=True, fontsize=8)
            
            # 标记中心
            ax.plot(mu[0], mu[1], 'r*', markersize=15)
            
            # 设置图形属性
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(title)
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 打印协方差信息
            print(f"\n{cov_type}协方差:")
            print(f"  协方差矩阵:\n{dist.sigma}")
            print(f"  行列式: {np.linalg.det(dist.sigma):.4f}")
        
        plt.suptitle('不同协方差结构的二维高斯分布', fontsize=14)
        plt.tight_layout()
        plt.show()


def demonstrate_mle_convergence(true_mean: float = 0.5,
                               true_variance: float = 1.0,
                               sample_sizes: List[int] = [10, 50, 100, 500],
                               show_plot: bool = True) -> None:
    """
    演示最大似然估计的收敛性
    
    随着数据增加，MLE估计：
    1. 逐渐接近真实参数（一致性）
    2. 估计的不确定性减小
    3. 方差估计是有偏的（低估）
    """
    print("\n最大似然估计的收敛性")
    print("=" * 60)
    print(f"真实参数: μ={true_mean}, σ²={true_variance}")
    print("-" * 60)
    
    np.random.seed(42)
    
    # 生成最大数据集
    max_n = max(sample_sizes)
    all_data = np.random.normal(true_mean, np.sqrt(true_variance), max_n)
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1：参数估计随数据量的变化
        ax1 = axes[0, 0]
        
        # 顺序估计
        estimates = UnivariateGaussian.mle_sequential(all_data[:max(sample_sizes)])
        n_range = np.arange(1, len(estimates) + 1)
        
        mu_estimates = [e[0] for e in estimates]
        sigma2_estimates = [e[1] for e in estimates]
        
        ax1.plot(n_range, mu_estimates, 'b-', alpha=0.7, label='μ估计')
        ax1.axhline(y=true_mean, color='b', linestyle='--', 
                   label=f'真实μ={true_mean}')
        ax1.set_xlabel('数据点数')
        ax1.set_ylabel('均值估计')
        ax1.set_title('均值估计的收敛')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：方差估计
        ax2 = axes[0, 1]
        ax2.plot(n_range, sigma2_estimates, 'r-', alpha=0.7, 
                label='σ²估计(有偏)')
        ax2.axhline(y=true_variance, color='r', linestyle='--',
                   label=f'真实σ²={true_variance}')
        
        # 无偏估计
        sigma2_unbiased = [e * n / (n - 1) if n > 1 else e 
                          for n, e in enumerate(sigma2_estimates, 1)]
        ax2.plot(n_range, sigma2_unbiased, 'g-', alpha=0.7,
                label='σ²估计(无偏)')
        
        ax2.set_xlabel('数据点数')
        ax2.set_ylabel('方差估计')
        ax2.set_title('方差估计的收敛')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3-4：不同数据量下的分布估计
        x_range = np.linspace(true_mean - 4*np.sqrt(true_variance),
                             true_mean + 4*np.sqrt(true_variance), 200)
        
        for idx, n in enumerate(sample_sizes[:2]):
            ax = axes[1, idx]
            
            # 获取数据子集
            data_subset = all_data[:n]
            
            # MLE估计
            dist_mle = UnivariateGaussian.mle(data_subset)
            
            # 绘制直方图
            ax.hist(data_subset, bins=20, density=True, alpha=0.7,
                   color='blue', edgecolor='black', label='数据')
            
            # 绘制估计的分布
            pdf_mle = dist_mle.pdf(x_range)
            ax.plot(x_range, pdf_mle, 'r-', linewidth=2,
                   label=f'MLE: μ={dist_mle.mu:.2f}, σ²={dist_mle.sigma2:.2f}')
            
            # 绘制真实分布
            dist_true = UnivariateGaussian(true_mean, true_variance)
            pdf_true = dist_true.pdf(x_range)
            ax.plot(x_range, pdf_true, 'g--', linewidth=2,
                   label=f'真实: μ={true_mean}, σ²={true_variance}')
            
            ax.set_xlabel('x')
            ax.set_ylabel('概率密度')
            ax.set_title(f'n={n}个数据点')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('MLE估计随数据增加的收敛', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 打印统计结果
    for n in sample_sizes:
        data_subset = all_data[:n]
        dist_mle = UnivariateGaussian.mle(data_subset)
        
        mu_error = abs(dist_mle.mu - true_mean)
        sigma2_error = abs(dist_mle.sigma2 - true_variance)
        sigma2_unbiased = dist_mle.sigma2 * n / (n - 1) if n > 1 else dist_mle.sigma2
        
        print(f"\nn={n}:")
        print(f"  μ估计: {dist_mle.mu:.4f} (误差: {mu_error:.4f})")
        print(f"  σ²估计(有偏): {dist_mle.sigma2:.4f} (误差: {sigma2_error:.4f})")
        print(f"  σ²估计(无偏): {sigma2_unbiased:.4f}")
    
    print("\n观察：")
    print("1. 均值估计是无偏的，快速收敛")
    print("2. 方差的MLE是有偏的，系统性低估")
    print("3. 使用n-1而不是n可以得到无偏估计")
    print("4. 随着数据增加，估计越来越准确")