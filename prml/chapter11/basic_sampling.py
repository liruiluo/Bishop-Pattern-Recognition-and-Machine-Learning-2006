"""
11.1-11.2 基础采样方法 (Basic Sampling Methods)
================================================

采样是从概率分布中生成样本的过程，是贝叶斯推理和蒙特卡罗方法的基础。

基础采样方法：

1. 拒绝采样 (Rejection Sampling)：
   - 使用提议分布q(z)采样
   - 以概率p(z)/(Mq(z))接受样本
   - 要求：p(z) ≤ Mq(z)对所有z成立
   - 效率取决于M的大小

2. 重要性采样 (Importance Sampling)：
   - 不拒绝样本，而是赋予权重
   - 权重：w(z) = p(z)/q(z)
   - 期望估计：E[f] ≈ Σ w_i f(z_i) / Σ w_i
   - 避免了拒绝，但方差可能很大

3. 采样-重要性-重采样 (SIR)：
   - 先用重要性采样生成加权样本
   - 然后根据权重重采样
   - 得到近似来自目标分布的无权样本

逆变换采样 (Inverse Transform Sampling)：
对于一维分布，如果CDF可逆：
1. 生成U ~ Uniform(0,1)
2. 返回F^(-1)(U)

Box-Muller变换：
从均匀分布生成标准正态分布：
- Z₀ = √(-2ln U₁) cos(2πU₂)
- Z₁ = √(-2ln U₁) sin(2πU₂)

应用：
- 贝叶斯推理
- 蒙特卡罗积分
- 粒子滤波
- 随机优化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
from typing import Callable, Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class RejectionSampler:
    """
    拒绝采样器
    
    从目标分布p(x)采样，使用提议分布q(x)。
    """
    
    def __init__(self, target_pdf: Callable,
                 proposal_sampler: Callable,
                 proposal_pdf: Callable,
                 M: float,
                 bounds: Optional[Tuple[float, float]] = None):
        """
        初始化拒绝采样器
        
        Args:
            target_pdf: 目标概率密度函数（未归一化也可）
            proposal_sampler: 提议分布采样器
            proposal_pdf: 提议分布概率密度
            M: 界常数，满足p(x) ≤ M*q(x)
            bounds: 采样区间
        """
        self.target_pdf = target_pdf
        self.proposal_sampler = proposal_sampler
        self.proposal_pdf = proposal_pdf
        self.M = M
        self.bounds = bounds
        
        # 统计信息
        self.n_accepted = 0
        self.n_proposed = 0
        
    def sample(self, n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """
        生成样本
        
        Args:
            n_samples: 样本数量
            random_state: 随机种子
            
        Returns:
            样本数组
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        
        while len(samples) < n_samples:
            # 从提议分布采样
            x = self.proposal_sampler()
            
            # 计算接受概率
            accept_prob = self.target_pdf(x) / (self.M * self.proposal_pdf(x))
            
            # 确保接受概率有效
            if accept_prob > 1:
                print(f"警告：接受概率{accept_prob:.3f} > 1，M可能太小")
                accept_prob = 1
            
            self.n_proposed += 1
            
            # 接受/拒绝决策
            if np.random.rand() < accept_prob:
                samples.append(x)
                self.n_accepted += 1
        
        return np.array(samples)
    
    def get_acceptance_rate(self) -> float:
        """获取接受率"""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


class ImportanceSampler:
    """
    重要性采样器
    
    使用加权样本近似目标分布。
    """
    
    def __init__(self, target_pdf: Callable,
                 proposal_sampler: Callable,
                 proposal_pdf: Callable):
        """
        初始化重要性采样器
        
        Args:
            target_pdf: 目标概率密度函数
            proposal_sampler: 提议分布采样器
            proposal_pdf: 提议分布概率密度
        """
        self.target_pdf = target_pdf
        self.proposal_sampler = proposal_sampler
        self.proposal_pdf = proposal_pdf
        
        # 存储样本和权重
        self.samples = None
        self.weights = None
        self.normalized_weights = None
        
    def sample(self, n_samples: int,
               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成加权样本
        
        Args:
            n_samples: 样本数量
            random_state: 随机种子
            
        Returns:
            (samples, weights): 样本和归一化权重
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 从提议分布采样
        samples = []
        for _ in range(n_samples):
            samples.append(self.proposal_sampler())
        samples = np.array(samples)
        
        # 计算重要性权重
        weights = np.zeros(n_samples)
        for i, x in enumerate(samples):
            weights[i] = self.target_pdf(x) / self.proposal_pdf(x)
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        self.samples = samples
        self.weights = weights
        self.normalized_weights = normalized_weights
        
        return samples, normalized_weights
    
    def estimate_expectation(self, f: Callable) -> float:
        """
        估计期望值E[f(X)]
        
        Args:
            f: 函数f
            
        Returns:
            期望估计
        """
        if self.samples is None:
            raise ValueError("请先调用sample()生成样本")
        
        # 加权平均
        f_values = np.array([f(x) for x in self.samples])
        expectation = np.sum(self.normalized_weights * f_values)
        
        return expectation
    
    def effective_sample_size(self) -> float:
        """
        计算有效样本大小
        
        ESS = 1 / Σ w_i^2
        
        Returns:
            有效样本大小
        """
        if self.normalized_weights is None:
            raise ValueError("请先调用sample()生成样本")
        
        ess = 1.0 / np.sum(self.normalized_weights ** 2)
        return ess
    
    def resample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        根据权重重采样（SIR）
        
        Args:
            n_samples: 重采样数量
            
        Returns:
            重采样后的样本
        """
        if self.samples is None:
            raise ValueError("请先调用sample()生成样本")
        
        if n_samples is None:
            n_samples = len(self.samples)
        
        # 根据权重重采样
        indices = np.random.choice(
            len(self.samples),
            size=n_samples,
            p=self.normalized_weights
        )
        
        return self.samples[indices]


def box_muller_transform(n_samples: int) -> np.ndarray:
    """
    Box-Muller变换
    
    从均匀分布生成标准正态分布。
    
    Args:
        n_samples: 样本数量
        
    Returns:
        标准正态样本
    """
    # 生成均匀分布样本
    u1 = np.random.uniform(0, 1, n_samples // 2 + 1)
    u2 = np.random.uniform(0, 1, n_samples // 2 + 1)
    
    # Box-Muller变换
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    
    # 合并并返回所需数量
    samples = np.concatenate([z0, z1])
    return samples[:n_samples]


def inverse_transform_sampling(cdf_inverse: Callable, 
                              n_samples: int) -> np.ndarray:
    """
    逆变换采样
    
    Args:
        cdf_inverse: CDF的逆函数
        n_samples: 样本数量
        
    Returns:
        样本数组
    """
    # 生成均匀分布样本
    u = np.random.uniform(0, 1, n_samples)
    
    # 应用逆CDF
    samples = cdf_inverse(u)
    
    return samples


def demonstrate_rejection_sampling(show_plot: bool = True) -> None:
    """
    演示拒绝采样
    """
    print("\n拒绝采样演示")
    print("=" * 60)
    
    # 定义目标分布：混合高斯
    def target_pdf(x):
        return 0.3 * stats.norm.pdf(x, -2, 0.5) + \
               0.7 * stats.norm.pdf(x, 2, 1.0)
    
    # 定义提议分布：均匀分布
    bounds = (-5, 5)
    def proposal_sampler():
        return np.random.uniform(bounds[0], bounds[1])
    
    def proposal_pdf(x):
        return 1.0 / (bounds[1] - bounds[0])
    
    # 找到M（通过网格搜索）
    x_grid = np.linspace(bounds[0], bounds[1], 1000)
    ratios = target_pdf(x_grid) / proposal_pdf(x_grid)
    M = np.max(ratios) * 1.1  # 添加10%余量
    
    print(f"界常数M: {M:.2f}")
    
    # 创建拒绝采样器
    sampler = RejectionSampler(
        target_pdf, proposal_sampler, proposal_pdf, M, bounds
    )
    
    # 生成样本
    n_samples = 5000
    samples = sampler.sample(n_samples, random_state=42)
    
    print(f"接受率: {sampler.get_acceptance_rate():.3f}")
    print(f"生成{n_samples}个样本需要{sampler.n_proposed}次提议")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 目标分布 vs 提议分布
        ax1 = axes[0, 0]
        x = np.linspace(bounds[0], bounds[1], 1000)
        ax1.plot(x, target_pdf(x), 'b-', linewidth=2, label='目标分布')
        ax1.plot(x, M * proposal_pdf(x) * np.ones_like(x), 'r--', 
                linewidth=2, label=f'M×提议分布')
        ax1.fill_between(x, 0, target_pdf(x), alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('概率密度')
        ax1.set_title('拒绝采样原理')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 样本直方图
        ax2 = axes[0, 1]
        ax2.hist(samples, bins=50, density=True, alpha=0.6, 
                color='green', label='样本')
        ax2.plot(x, target_pdf(x), 'b-', linewidth=2, label='目标分布')
        ax2.set_xlabel('x')
        ax2.set_ylabel('概率密度')
        ax2.set_title(f'采样结果 (n={n_samples})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 接受/拒绝可视化
        ax3 = axes[1, 0]
        # 生成一些提议点进行可视化
        n_vis = 100
        x_prop = np.random.uniform(bounds[0], bounds[1], n_vis)
        y_prop = np.random.uniform(0, M * proposal_pdf(x_prop[0]), n_vis)
        
        # 判断接受/拒绝
        accept_mask = y_prop < target_pdf(x_prop)
        
        ax3.scatter(x_prop[accept_mask], y_prop[accept_mask], 
                   c='green', s=20, alpha=0.6, label='接受')
        ax3.scatter(x_prop[~accept_mask], y_prop[~accept_mask],
                   c='red', s=20, alpha=0.6, label='拒绝')
        ax3.plot(x, target_pdf(x), 'b-', linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('接受/拒绝区域')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 累积平均
        ax4 = axes[1, 1]
        cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        true_mean = -2 * 0.3 + 2 * 0.7  # 理论均值
        ax4.plot(cumulative_mean, 'b-', linewidth=2, label='累积平均')
        ax4.axhline(y=true_mean, color='r', linestyle='--', 
                   linewidth=2, label=f'真实均值={true_mean:.2f}')
        ax4.set_xlabel('样本数')
        ax4.set_ylabel('累积平均')
        ax4.set_title('收敛性')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('拒绝采样', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 接受率取决于M的选择")
    print("2. M越小，接受率越高")
    print("3. 高维空间中效率急剧下降")


def demonstrate_importance_sampling(show_plot: bool = True) -> None:
    """
    演示重要性采样
    """
    print("\n重要性采样演示")
    print("=" * 60)
    
    # 目标分布：标准正态的尾部
    def target_pdf(x):
        # 只关注尾部x > 3
        if x < 3:
            return 0
        return stats.norm.pdf(x, 0, 1)
    
    # 提议分布：指数分布（平移）
    def proposal_sampler():
        return 3 + np.random.exponential(1)
    
    def proposal_pdf(x):
        if x < 3:
            return 0
        return np.exp(-(x - 3))
    
    # 创建重要性采样器
    sampler = ImportanceSampler(target_pdf, proposal_sampler, proposal_pdf)
    
    # 生成样本
    n_samples = 5000
    samples, weights = sampler.sample(n_samples, random_state=42)
    
    # 计算有效样本大小
    ess = sampler.effective_sample_size()
    print(f"有效样本大小: {ess:.1f} / {n_samples} = {ess/n_samples:.3f}")
    
    # 估计期望
    def f(x):
        return x  # 估计E[X]
    
    expectation = sampler.estimate_expectation(f)
    print(f"E[X|X>3]估计: {expectation:.3f}")
    
    # 理论值（截断正态分布）
    from scipy.stats import truncnorm
    true_mean = truncnorm.mean(a=3, b=np.inf, loc=0, scale=1)
    print(f"E[X|X>3]真实: {true_mean:.3f}")
    
    # SIR重采样
    resampled = sampler.resample()
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 目标分布 vs 提议分布
        ax1 = axes[0, 0]
        x = np.linspace(3, 8, 1000)
        ax1.plot(x, [target_pdf(xi) for xi in x], 'b-', 
                linewidth=2, label='目标分布')
        ax1.plot(x, [proposal_pdf(xi) for xi in x], 'r--',
                linewidth=2, label='提议分布')
        ax1.set_xlabel('x')
        ax1.set_ylabel('概率密度')
        ax1.set_title('分布比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 权重分布
        ax2 = axes[0, 1]
        ax2.hist(weights, bins=50, alpha=0.6, color='orange')
        ax2.set_xlabel('权重')
        ax2.set_ylabel('频数')
        ax2.set_title(f'权重分布 (ESS={ess:.1f})')
        ax2.grid(True, alpha=0.3)
        
        # 加权样本
        ax3 = axes[0, 2]
        # 按权重大小着色
        scatter = ax3.scatter(samples, weights, c=weights,
                            cmap='YlOrRd', s=20, alpha=0.6)
        ax3.set_xlabel('样本值')
        ax3.set_ylabel('权重')
        ax3.set_title('样本-权重关系')
        plt.colorbar(scatter, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # 原始样本直方图
        ax4 = axes[1, 0]
        ax4.hist(samples, bins=50, density=True, alpha=0.6,
                color='red', label='提议样本')
        ax4.plot(x, [target_pdf(xi)/0.00135 for xi in x], 'b-',
                linewidth=2, label='目标分布（归一化）')
        ax4.set_xlabel('x')
        ax4.set_ylabel('概率密度')
        ax4.set_title('原始样本')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 重采样结果
        ax5 = axes[1, 1]
        ax5.hist(resampled, bins=50, density=True, alpha=0.6,
                color='green', label='SIR样本')
        ax5.plot(x, [target_pdf(xi)/0.00135 for xi in x], 'b-',
                linewidth=2, label='目标分布（归一化）')
        ax5.set_xlabel('x')
        ax5.set_ylabel('概率密度')
        ax5.set_title('SIR重采样')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 期望估计收敛
        ax6 = axes[1, 2]
        cumulative_estimates = []
        for i in range(1, len(samples)):
            est = np.sum(weights[:i] * samples[:i]) / np.sum(weights[:i])
            cumulative_estimates.append(est)
        
        ax6.plot(cumulative_estimates, 'b-', linewidth=2, label='IS估计')
        ax6.axhline(y=true_mean, color='r', linestyle='--',
                   linewidth=2, label=f'真实值={true_mean:.3f}')
        ax6.set_xlabel('样本数')
        ax6.set_ylabel('E[X|X>3]')
        ax6.set_title('期望估计收敛')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('重要性采样', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 重要性采样避免了拒绝")
    print("2. 权重方差影响估计精度")
    print("3. ESS衡量样本效率")
    print("4. SIR将加权样本转为无权样本")


def demonstrate_box_muller(show_plot: bool = True) -> None:
    """
    演示Box-Muller变换
    """
    print("\nBox-Muller变换演示")
    print("=" * 60)
    
    n_samples = 10000
    
    # 使用Box-Muller生成样本
    samples_bm = box_muller_transform(n_samples)
    
    # 对比：NumPy的正态分布
    samples_np = np.random.randn(n_samples)
    
    print(f"Box-Muller样本统计：")
    print(f"  均值: {np.mean(samples_bm):.4f}")
    print(f"  标准差: {np.std(samples_bm):.4f}")
    print(f"\nNumPy样本统计：")
    print(f"  均值: {np.mean(samples_np):.4f}")
    print(f"  标准差: {np.std(samples_np):.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Box-Muller直方图
        ax1 = axes[0, 0]
        ax1.hist(samples_bm, bins=50, density=True, alpha=0.6,
                color='blue', label='Box-Muller')
        x = np.linspace(-4, 4, 100)
        ax1.plot(x, stats.norm.pdf(x), 'r-', linewidth=2,
                label='标准正态')
        ax1.set_xlabel('值')
        ax1.set_ylabel('概率密度')
        ax1.set_title('Box-Muller变换')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        ax2 = axes[0, 1]
        stats.probplot(samples_bm, dist="norm", plot=ax2)
        ax2.set_title('Q-Q图（Box-Muller）')
        ax2.grid(True, alpha=0.3)
        
        # 变换过程可视化
        ax3 = axes[1, 0]
        # 显示均匀分布到正态分布的映射
        u1 = np.random.uniform(0, 1, 1000)
        u2 = np.random.uniform(0, 1, 1000)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        
        scatter = ax3.scatter(u1, u2, c=z, cmap='coolwarm', s=10, alpha=0.6)
        ax3.set_xlabel('U₁')
        ax3.set_ylabel('U₂')
        ax3.set_title('均匀分布输入（颜色=输出值）')
        plt.colorbar(scatter, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # 2D正态分布
        ax4 = axes[1, 1]
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        ax4.hexbin(z0, z1, gridsize=30, cmap='YlOrRd')
        ax4.set_xlabel('Z₀')
        ax4.set_ylabel('Z₁')
        ax4.set_title('生成的2D正态分布')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Box-Muller变换', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. Box-Muller精确生成正态分布")
    print("2. 一次生成两个独立样本")
    print("3. 计算效率高")
    print("4. 适合需要大量正态样本的场景")