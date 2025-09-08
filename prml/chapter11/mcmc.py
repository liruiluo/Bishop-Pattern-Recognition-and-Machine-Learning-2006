"""
11.2-11.3 马尔可夫链蒙特卡罗 (Markov Chain Monte Carlo, MCMC)
=============================================================

MCMC是一类通过构建马尔可夫链来从复杂分布采样的方法。

核心思想：
构建一个以目标分布π(x)为平稳分布的马尔可夫链，
经过足够长时间后，链的状态分布收敛到目标分布。

主要算法：

1. Metropolis-Hastings算法：
   - 提议新状态：x' ~ q(x'|x)
   - 接受概率：α = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
   - 如果q对称，简化为Metropolis算法

2. Gibbs采样：
   - 轮流从条件分布采样
   - x_i ~ p(x_i | x_{-i})
   - 是MH的特例（接受率总是1）

3. Hamiltonian Monte Carlo (HMC)：
   - 引入辅助动量变量
   - 使用哈密顿动力学演化
   - 大步长，高接受率

详细平衡条件：
π(x)T(x'|x) = π(x')T(x|x')
保证π是平稳分布。

收敛诊断：
- R̂统计量（Gelman-Rubin）
- 有效样本大小（ESS）
- 自相关函数

应用：
- 贝叶斯推理
- 统计物理
- 机器学习中的后验采样
- 概率编程
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, Tuple, Optional, Dict, List, Union
import warnings
warnings.filterwarnings('ignore')


class MetropolisHastings:
    """
    Metropolis-Hastings采样器
    
    通用的MCMC算法，可以处理任意目标分布。
    """
    
    def __init__(self, target_log_pdf: Callable,
                 proposal_sampler: Callable,
                 proposal_log_pdf: Optional[Callable] = None,
                 initial_state: Optional[np.ndarray] = None):
        """
        初始化MH采样器
        
        Args:
            target_log_pdf: 目标分布的对数概率密度
            proposal_sampler: 提议分布采样器 q(x'|x)
            proposal_log_pdf: 提议分布的对数概率密度（非对称时需要）
            initial_state: 初始状态
        """
        self.target_log_pdf = target_log_pdf
        self.proposal_sampler = proposal_sampler
        self.proposal_log_pdf = proposal_log_pdf
        self.initial_state = initial_state
        
        # 采样链
        self.chain = []
        self.accepted = []
        self.n_accepted = 0
        self.n_proposed = 0
        
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        执行一步MH采样
        
        Args:
            current_state: 当前状态
            
        Returns:
            (new_state, accepted): 新状态和是否接受
        """
        # 提议新状态
        proposed_state = self.proposal_sampler(current_state)
        
        # 计算接受率
        log_ratio = self.target_log_pdf(proposed_state) - self.target_log_pdf(current_state)
        
        # 如果提议分布非对称，需要校正
        if self.proposal_log_pdf is not None:
            log_ratio += self.proposal_log_pdf(current_state, proposed_state)
            log_ratio -= self.proposal_log_pdf(proposed_state, current_state)
        
        log_accept_prob = min(0, log_ratio)
        
        # 接受/拒绝
        if np.log(np.random.rand()) < log_accept_prob:
            return proposed_state, True
        else:
            return current_state, False
    
    def sample(self, n_samples: int,
               burn_in: int = 0,
               thin: int = 1,
               random_state: Optional[int] = None) -> np.ndarray:
        """
        生成MCMC样本
        
        Args:
            n_samples: 样本数量
            burn_in: 预烧期
            thin: 细化因子
            random_state: 随机种子
            
        Returns:
            样本链
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 初始化
        if self.initial_state is None:
            current_state = np.random.randn()
        else:
            current_state = self.initial_state.copy()
        
        self.chain = []
        self.accepted = []
        self.n_accepted = 0
        self.n_proposed = 0
        
        # 预烧期
        for _ in range(burn_in):
            current_state, _ = self.step(current_state)
        
        # 主采样
        for i in range(n_samples * thin):
            current_state, accepted = self.step(current_state)
            self.n_proposed += 1
            
            if accepted:
                self.n_accepted += 1
            
            # 细化
            if i % thin == 0:
                self.chain.append(current_state.copy())
                self.accepted.append(accepted)
        
        return np.array(self.chain)
    
    def get_acceptance_rate(self) -> float:
        """获取接受率"""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


class GibbsSampler:
    """
    Gibbs采样器
    
    从条件分布轮流采样各个维度。
    """
    
    def __init__(self, conditional_samplers: List[Callable],
                 initial_state: Optional[np.ndarray] = None):
        """
        初始化Gibbs采样器
        
        Args:
            conditional_samplers: 条件分布采样器列表
            initial_state: 初始状态
        """
        self.conditional_samplers = conditional_samplers
        self.n_dimensions = len(conditional_samplers)
        self.initial_state = initial_state
        
        # 采样链
        self.chain = []
        
    def step(self, current_state: np.ndarray) -> np.ndarray:
        """
        执行一轮Gibbs采样
        
        Args:
            current_state: 当前状态
            
        Returns:
            新状态
        """
        new_state = current_state.copy()
        
        # 轮流更新每个维度
        for i in range(self.n_dimensions):
            # 从条件分布采样
            new_state[i] = self.conditional_samplers[i](new_state, i)
        
        return new_state
    
    def sample(self, n_samples: int,
               burn_in: int = 0,
               thin: int = 1,
               random_state: Optional[int] = None) -> np.ndarray:
        """
        生成Gibbs样本
        
        Args:
            n_samples: 样本数量
            burn_in: 预烧期
            thin: 细化因子
            random_state: 随机种子
            
        Returns:
            样本链
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 初始化
        if self.initial_state is None:
            current_state = np.random.randn(self.n_dimensions)
        else:
            current_state = self.initial_state.copy()
        
        self.chain = []
        
        # 预烧期
        for _ in range(burn_in):
            current_state = self.step(current_state)
        
        # 主采样
        for i in range(n_samples * thin):
            current_state = self.step(current_state)
            
            # 细化
            if i % thin == 0:
                self.chain.append(current_state.copy())
        
        return np.array(self.chain)


class HamiltonianMC:
    """
    哈密顿蒙特卡罗 (HMC)
    
    使用哈密顿动力学在相空间中演化。
    """
    
    def __init__(self, target_log_pdf: Callable,
                 grad_log_pdf: Callable,
                 step_size: float = 0.01,
                 n_leapfrog: int = 10,
                 mass_matrix: Optional[np.ndarray] = None):
        """
        初始化HMC采样器
        
        Args:
            target_log_pdf: 目标分布的对数概率密度
            grad_log_pdf: 对数概率密度的梯度
            step_size: Leapfrog积分步长
            n_leapfrog: Leapfrog步数
            mass_matrix: 质量矩阵（默认为单位矩阵）
        """
        self.target_log_pdf = target_log_pdf
        self.grad_log_pdf = grad_log_pdf
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.mass_matrix = mass_matrix
        
        # 采样链
        self.chain = []
        self.accepted = []
        self.n_accepted = 0
        self.n_proposed = 0
        
    def leapfrog(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog积分器
        
        Args:
            q: 位置
            p: 动量
            
        Returns:
            (new_q, new_p): 新的位置和动量
        """
        q = q.copy()
        p = p.copy()
        
        # 半步动量更新
        p = p + 0.5 * self.step_size * self.grad_log_pdf(q)
        
        # Leapfrog步
        for _ in range(self.n_leapfrog - 1):
            q = q + self.step_size * p  # 如果有质量矩阵，需要p/m
            p = p + self.step_size * self.grad_log_pdf(q)
        
        # 最后的位置更新
        q = q + self.step_size * p
        
        # 最后的半步动量更新
        p = p + 0.5 * self.step_size * self.grad_log_pdf(q)
        
        # 反转动量（保持详细平衡）
        p = -p
        
        return q, p
    
    def step(self, current_q: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        执行一步HMC采样
        
        Args:
            current_q: 当前位置
            
        Returns:
            (new_q, accepted): 新位置和是否接受
        """
        # 采样动量
        current_p = np.random.randn(*current_q.shape)
        
        # Leapfrog积分
        proposed_q, proposed_p = self.leapfrog(current_q, current_p)
        
        # 计算哈密顿量
        current_H = -self.target_log_pdf(current_q) + 0.5 * np.sum(current_p**2)
        proposed_H = -self.target_log_pdf(proposed_q) + 0.5 * np.sum(proposed_p**2)
        
        # 接受率
        log_accept_prob = min(0, current_H - proposed_H)
        
        # 接受/拒绝
        if np.log(np.random.rand()) < log_accept_prob:
            return proposed_q, True
        else:
            return current_q, False
    
    def sample(self, n_samples: int,
               initial_state: Optional[np.ndarray] = None,
               burn_in: int = 0,
               thin: int = 1,
               random_state: Optional[int] = None) -> np.ndarray:
        """
        生成HMC样本
        
        Args:
            n_samples: 样本数量
            initial_state: 初始状态
            burn_in: 预烧期
            thin: 细化因子
            random_state: 随机种子
            
        Returns:
            样本链
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 初始化
        if initial_state is None:
            current_q = np.random.randn()
        else:
            current_q = initial_state.copy()
        
        self.chain = []
        self.accepted = []
        self.n_accepted = 0
        self.n_proposed = 0
        
        # 预烧期
        for _ in range(burn_in):
            current_q, _ = self.step(current_q)
        
        # 主采样
        for i in range(n_samples * thin):
            current_q, accepted = self.step(current_q)
            self.n_proposed += 1
            
            if accepted:
                self.n_accepted += 1
            
            # 细化
            if i % thin == 0:
                self.chain.append(current_q.copy())
                self.accepted.append(accepted)
        
        return np.array(self.chain)
    
    def get_acceptance_rate(self) -> float:
        """获取接受率"""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


def compute_effective_sample_size(chain: np.ndarray) -> float:
    """
    计算有效样本大小(ESS)
    
    Args:
        chain: MCMC链
        
    Returns:
        ESS
    """
    n_samples = len(chain)
    
    # 计算自相关
    mean = np.mean(chain)
    var = np.var(chain)
    
    if var == 0:
        return n_samples
    
    # 计算自相关函数
    autocorr = []
    for lag in range(min(n_samples // 4, 100)):  # 最多计算1/4长度
        if lag == 0:
            autocorr.append(1.0)
        else:
            c = np.mean((chain[:-lag] - mean) * (chain[lag:] - mean)) / var
            autocorr.append(c)
            
            # 如果自相关变小，停止
            if abs(c) < 0.05:
                break
    
    # 积分自相关时间
    tau = 1 + 2 * sum(autocorr[1:])
    
    # ESS = n / tau
    ess = n_samples / tau
    
    return ess


def compute_rhat(chains: List[np.ndarray]) -> float:
    """
    计算R̂统计量（Gelman-Rubin）
    
    Args:
        chains: 多条MCMC链
        
    Returns:
        R̂值
    """
    m = len(chains)  # 链数
    n = len(chains[0])  # 每条链长度
    
    # 链间方差
    chain_means = [np.mean(chain) for chain in chains]
    B = n * np.var(chain_means)
    
    # 链内方差
    W = np.mean([np.var(chain) for chain in chains])
    
    # 后验方差估计
    var_plus = ((n - 1) * W + B) / n
    
    # R̂
    rhat = np.sqrt(var_plus / W)
    
    return rhat


def demonstrate_metropolis_hastings(show_plot: bool = True) -> None:
    """
    演示Metropolis-Hastings算法
    """
    print("\nMetropolis-Hastings算法演示")
    print("=" * 60)
    
    # 目标分布：双峰混合高斯
    def target_log_pdf(x):
        p1 = stats.norm.pdf(x, -2, 0.5)
        p2 = stats.norm.pdf(x, 2, 0.8)
        return np.log(0.3 * p1 + 0.7 * p2)
    
    # 提议分布：随机游走
    def proposal_sampler(x):
        return x + np.random.normal(0, 0.5)
    
    # 创建MH采样器
    mh_sampler = MetropolisHastings(
        target_log_pdf=target_log_pdf,
        proposal_sampler=proposal_sampler
    )
    
    # 生成样本
    n_samples = 10000
    samples = mh_sampler.sample(n_samples, burn_in=1000, random_state=42)
    
    print(f"接受率: {mh_sampler.get_acceptance_rate():.3f}")
    
    # 计算ESS
    ess = compute_effective_sample_size(samples)
    print(f"有效样本大小: {ess:.1f} / {n_samples} = {ess/n_samples:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 轨迹图
        ax1 = axes[0, 0]
        ax1.plot(samples[:1000], 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('值')
        ax1.set_title('MCMC轨迹（前1000步）')
        ax1.grid(True, alpha=0.3)
        
        # 样本直方图
        ax2 = axes[0, 1]
        ax2.hist(samples, bins=50, density=True, alpha=0.6,
                color='green', label='MCMC样本')
        x = np.linspace(-5, 5, 1000)
        true_pdf = 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 2, 0.8)
        ax2.plot(x, true_pdf, 'b-', linewidth=2, label='目标分布')
        ax2.set_xlabel('值')
        ax2.set_ylabel('概率密度')
        ax2.set_title('样本分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 自相关函数
        ax3 = axes[0, 2]
        lags = range(50)
        autocorr = [1.0]
        for lag in range(1, 50):
            c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
            autocorr.append(c)
        ax3.bar(lags, autocorr, alpha=0.7, color='blue')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('滞后')
        ax3.set_ylabel('自相关')
        ax3.set_title(f'自相关函数 (ESS={ess:.1f})')
        ax3.grid(True, alpha=0.3)
        
        # 累积均值
        ax4 = axes[1, 0]
        cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        true_mean = -2 * 0.3 + 2 * 0.7
        ax4.plot(cumulative_mean, 'b-', linewidth=1, label='累积均值')
        ax4.axhline(y=true_mean, color='r', linestyle='--',
                   linewidth=2, label=f'真实均值={true_mean:.2f}')
        ax4.set_xlabel('样本数')
        ax4.set_ylabel('累积均值')
        ax4.set_title('收敛诊断')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 接受率随时间变化
        ax5 = axes[1, 1]
        accept_rate = []
        window = 100
        for i in range(window, len(mh_sampler.accepted)):
            rate = np.mean(mh_sampler.accepted[i-window:i])
            accept_rate.append(rate)
        ax5.plot(accept_rate, 'g-', linewidth=1)
        ax5.set_xlabel('迭代')
        ax5.set_ylabel('接受率')
        ax5.set_title('局部接受率（窗口=100）')
        ax5.grid(True, alpha=0.3)
        
        # 2D轨迹（连续两个样本）
        ax6 = axes[1, 2]
        ax6.scatter(samples[:-1], samples[1:], s=1, alpha=0.3, c='blue')
        ax6.set_xlabel('x(t)')
        ax6.set_ylabel('x(t+1)')
        ax6.set_title('相空间轨迹')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Metropolis-Hastings采样', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. MH可以采样复杂分布")
    print("2. 接受率影响混合速度")
    print("3. 自相关决定有效样本大小")
    print("4. 需要预烧期达到平稳分布")


def demonstrate_gibbs_sampling(show_plot: bool = True) -> None:
    """
    演示Gibbs采样（二维高斯）
    """
    print("\nGibbs采样演示")
    print("=" * 60)
    
    # 目标：二维高斯分布
    mean = np.array([1, 2])
    cov = np.array([[1, 0.8], [0.8, 2]])
    
    # 条件分布采样器
    def sample_x_given_y(state, dim):
        if dim == 0:  # 采样x|y
            y = state[1]
            # 条件分布参数
            cond_mean = mean[0] + cov[0, 1] / cov[1, 1] * (y - mean[1])
            cond_var = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]
            return np.random.normal(cond_mean, np.sqrt(cond_var))
        else:  # 采样y|x
            x = state[0]
            cond_mean = mean[1] + cov[1, 0] / cov[0, 0] * (x - mean[0])
            cond_var = cov[1, 1] - cov[1, 0]**2 / cov[0, 0]
            return np.random.normal(cond_mean, np.sqrt(cond_var))
    
    # 创建Gibbs采样器
    gibbs = GibbsSampler(
        conditional_samplers=[sample_x_given_y, sample_x_given_y],
        initial_state=np.array([0, 0])
    )
    
    # 生成样本
    n_samples = 5000
    samples = gibbs.sample(n_samples, burn_in=500, random_state=42)
    
    print(f"样本均值: {np.mean(samples, axis=0)}")
    print(f"真实均值: {mean}")
    print(f"样本协方差:\n{np.cov(samples.T)}")
    print(f"真实协方差:\n{cov}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Gibbs采样路径
        ax1 = axes[0, 0]
        ax1.plot(samples[:100, 0], samples[:100, 1], 'b-', 
                linewidth=1, alpha=0.7, marker='o', markersize=3)
        ax1.scatter(samples[0, 0], samples[0, 1], c='red', s=100, 
                   marker='s', label='起点')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Gibbs采样路径（前100步）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 样本散点图
        ax2 = axes[0, 1]
        ax2.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='blue')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Gibbs样本分布')
        ax2.grid(True, alpha=0.3)
        
        # 边缘分布
        ax3 = axes[0, 2]
        ax3.hist(samples[:, 0], bins=50, density=True, alpha=0.5,
                color='blue', label='X边缘')
        ax3.hist(samples[:, 1], bins=50, density=True, alpha=0.5,
                color='red', label='Y边缘')
        x_range = np.linspace(-3, 5, 100)
        ax3.plot(x_range, stats.norm.pdf(x_range, mean[0], np.sqrt(cov[0, 0])),
                'b-', linewidth=2, label='X理论')
        ax3.plot(x_range, stats.norm.pdf(x_range, mean[1], np.sqrt(cov[1, 1])),
                'r-', linewidth=2, label='Y理论')
        ax3.set_xlabel('值')
        ax3.set_ylabel('概率密度')
        ax3.set_title('边缘分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 轨迹图
        ax4 = axes[1, 0]
        ax4.plot(samples[:500, 0], 'b-', linewidth=1, alpha=0.7, label='X')
        ax4.plot(samples[:500, 1], 'r-', linewidth=1, alpha=0.7, label='Y')
        ax4.set_xlabel('迭代')
        ax4.set_ylabel('值')
        ax4.set_title('Gibbs采样轨迹')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 条件采样可视化
        ax5 = axes[1, 1]
        # 显示条件分布
        y_fixed = 2.0
        x_range = np.linspace(-2, 4, 100)
        cond_mean_x = mean[0] + cov[0, 1] / cov[1, 1] * (y_fixed - mean[1])
        cond_std_x = np.sqrt(cov[0, 0] - cov[0, 1]**2 / cov[1, 1])
        ax5.plot(x_range, stats.norm.pdf(x_range, cond_mean_x, cond_std_x),
                'b-', linewidth=2, label=f'p(X|Y={y_fixed})')
        
        x_fixed = 1.0
        y_range = np.linspace(-2, 6, 100)
        cond_mean_y = mean[1] + cov[1, 0] / cov[0, 0] * (x_fixed - mean[0])
        cond_std_y = np.sqrt(cov[1, 1] - cov[1, 0]**2 / cov[0, 0])
        ax5.plot(y_range, stats.norm.pdf(y_range, cond_mean_y, cond_std_y),
                'r-', linewidth=2, label=f'p(Y|X={x_fixed})')
        ax5.set_xlabel('值')
        ax5.set_ylabel('概率密度')
        ax5.set_title('条件分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 收敛诊断
        ax6 = axes[1, 2]
        # 运行多条链
        chains = []
        for _ in range(4):
            chain = gibbs.sample(1000, burn_in=100)
            chains.append(chain[:, 0])  # 只看X维度
        
        for i, chain in enumerate(chains):
            ax6.plot(chain[:200], linewidth=1, alpha=0.7, label=f'链{i+1}')
        
        rhat = compute_rhat(chains)
        ax6.set_xlabel('迭代')
        ax6.set_ylabel('X值')
        ax6.set_title(f'多链诊断 (R̂={rhat:.3f})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Gibbs采样', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print(f"\nR̂统计量: {rhat:.3f} (接近1表示收敛)")
    
    print("\n观察：")
    print("1. Gibbs采样无需调节接受率")
    print("2. 沿坐标轴移动（可能慢）")
    print("3. 需要能够从条件分布采样")
    print("4. 高相关性时效率低")


def demonstrate_hmc(show_plot: bool = True) -> None:
    """
    演示哈密顿蒙特卡罗
    """
    print("\n哈密顿蒙特卡罗(HMC)演示")
    print("=" * 60)
    
    # 目标分布：香蕉形分布(Banana distribution)
    def target_log_pdf(x):
        if len(x.shape) == 0:  # 标量
            x = np.array([x, 0])
        return -0.5 * (x[0]**2 + (x[1] - x[0]**2)**2)
    
    def grad_log_pdf(x):
        if len(x.shape) == 0:  # 标量情况
            return np.array([-x])
        # 香蕉分布的梯度
        grad = np.zeros_like(x)
        grad[0] = -x[0] - 2 * x[0] * (x[1] - x[0]**2)
        grad[1] = -(x[1] - x[0]**2)
        return grad
    
    # 为了演示，使用1D高斯
    def target_log_pdf_1d(x):
        return -0.5 * x**2
    
    def grad_log_pdf_1d(x):
        return -x
    
    # 创建HMC采样器
    hmc = HamiltonianMC(
        target_log_pdf=target_log_pdf_1d,
        grad_log_pdf=grad_log_pdf_1d,
        step_size=0.1,
        n_leapfrog=10
    )
    
    # 生成样本
    n_samples = 5000
    samples = hmc.sample(n_samples, burn_in=500, random_state=42)
    
    print(f"接受率: {hmc.get_acceptance_rate():.3f}")
    
    # 计算ESS
    ess = compute_effective_sample_size(samples)
    print(f"有效样本大小: {ess:.1f} / {n_samples} = {ess/n_samples:.3f}")
    
    # 比较with随机游走MH
    mh_sampler = MetropolisHastings(
        target_log_pdf=target_log_pdf_1d,
        proposal_sampler=lambda x: x + np.random.normal(0, 0.5)
    )
    mh_samples = mh_sampler.sample(n_samples, burn_in=500, random_state=42)
    mh_ess = compute_effective_sample_size(mh_samples)
    
    print(f"\n比较：")
    print(f"  HMC ESS/n: {ess/n_samples:.3f}")
    print(f"  MH ESS/n: {mh_ess/n_samples:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # HMC轨迹
        ax1 = axes[0, 0]
        ax1.plot(samples[:500], 'b-', linewidth=1, alpha=0.7, label='HMC')
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('值')
        ax1.set_title('HMC轨迹')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MH轨迹（对比）
        ax2 = axes[0, 1]
        ax2.plot(mh_samples[:500], 'r-', linewidth=1, alpha=0.7, label='MH')
        ax2.set_xlabel('迭代')
        ax2.set_ylabel('值')
        ax2.set_title('MH轨迹（对比）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 自相关比较
        ax3 = axes[0, 2]
        lags = range(30)
        
        # HMC自相关
        hmc_autocorr = [1.0]
        for lag in range(1, 30):
            c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
            hmc_autocorr.append(c)
        
        # MH自相关
        mh_autocorr = [1.0]
        for lag in range(1, 30):
            c = np.corrcoef(mh_samples[:-lag], mh_samples[lag:])[0, 1]
            mh_autocorr.append(c)
        
        ax3.plot(lags, hmc_autocorr, 'b-', linewidth=2, label='HMC', marker='o')
        ax3.plot(lags, mh_autocorr, 'r-', linewidth=2, label='MH', marker='s')
        ax3.axhline(y=0, color='gray', linestyle='--')
        ax3.set_xlabel('滞后')
        ax3.set_ylabel('自相关')
        ax3.set_title('自相关比较')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 样本分布
        ax4 = axes[1, 0]
        ax4.hist(samples, bins=50, density=True, alpha=0.5,
                color='blue', label='HMC')
        ax4.hist(mh_samples, bins=50, density=True, alpha=0.5,
                color='red', label='MH')
        x = np.linspace(-4, 4, 100)
        ax4.plot(x, stats.norm.pdf(x), 'k-', linewidth=2, label='真实')
        ax4.set_xlabel('值')
        ax4.set_ylabel('概率密度')
        ax4.set_title('样本分布比较')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Leapfrog轨迹可视化
        ax5 = axes[1, 1]
        # 显示一个Leapfrog轨迹
        q0 = np.array([0.0])
        p0 = np.array([1.0])
        
        # 记录轨迹
        q_traj = [q0[0]]
        p_traj = [p0[0]]
        
        q = q0.copy()
        p = p0.copy()
        
        # 简化的Leapfrog（用于可视化）
        for _ in range(20):
            p = p - 0.1 * q  # 势能梯度
            q = q + 0.1 * p
            q_traj.append(q[0])
            p_traj.append(p[0])
        
        ax5.plot(q_traj, p_traj, 'b-', linewidth=2, marker='o', markersize=4)
        ax5.scatter(q_traj[0], p_traj[0], c='red', s=100, marker='s', label='起点')
        ax5.scatter(q_traj[-1], p_traj[-1], c='green', s=100, marker='^', label='终点')
        ax5.set_xlabel('位置 q')
        ax5.set_ylabel('动量 p')
        ax5.set_title('Leapfrog轨迹（相空间）')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 能量守恒
        ax6 = axes[1, 2]
        # 计算哈密顿量
        H_traj = []
        for q, p in zip(q_traj, p_traj):
            H = 0.5 * q**2 + 0.5 * p**2  # 势能 + 动能
            H_traj.append(H)
        
        ax6.plot(H_traj, 'g-', linewidth=2, marker='o')
        ax6.axhline(y=H_traj[0], color='r', linestyle='--',
                   label=f'初始能量={H_traj[0]:.3f}')
        ax6.set_xlabel('Leapfrog步')
        ax6.set_ylabel('哈密顿量')
        ax6.set_title('能量守恒')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('哈密顿蒙特卡罗(HMC)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. HMC有更低的自相关")
    print("2. 更高的有效样本大小")
    print("3. Leapfrog保持能量近似守恒")
    print("4. 适合高维连续分布")