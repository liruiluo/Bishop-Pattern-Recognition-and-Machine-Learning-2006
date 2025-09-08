"""
2.1 二元变量 (Binary Variables)
===============================

二元变量是机器学习中最基础的随机变量类型，它只有两个可能的状态。
比如：硬币的正反面、分类问题中的是否、医疗诊断中的患病与健康等。

本节涵盖的分布：
1. 伯努利分布 (Bernoulli)：单次试验
2. 二项分布 (Binomial)：多次独立试验
3. 贝塔分布 (Beta)：伯努利和二项分布的共轭先验

共轭先验的重要性：
当先验分布和后验分布属于同一分布族时，我们称该先验为共轭先验。
这使得贝叶斯推断在数学上变得优雅且易于计算。

对于二项分布：
- 似然：二项分布
- 共轭先验：贝塔分布
- 后验：贝塔分布（参数更新）

这种共轭关系让我们能够用解析方法进行贝叶斯推断，而不需要复杂的数值计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, beta as beta_function
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class BernoulliDistribution:
    """
    伯努利分布：二元随机变量的单次试验
    
    P(x=1|μ) = μ
    P(x=0|μ) = 1 - μ
    
    可以写成统一形式：
    P(x|μ) = μ^x * (1-μ)^(1-x)
    
    其中：
    - x ∈ {0, 1}
    - μ ∈ [0, 1] 是成功概率
    
    性质：
    - 期望：E[x] = μ
    - 方差：Var[x] = μ(1-μ)
    - 最大方差出现在μ=0.5时
    """
    
    def __init__(self, mu: float):
        """
        初始化伯努利分布
        
        Args:
            mu: 成功概率，必须在[0,1]之间
        """
        if not 0 <= mu <= 1:
            raise ValueError(f"参数mu必须在[0,1]之间，得到{mu}")
        self.mu = mu
    
    def pmf(self, x: int) -> float:
        """
        概率质量函数 (Probability Mass Function)
        
        Args:
            x: 取值，必须是0或1
            
        Returns:
            P(X=x)
        """
        if x not in {0, 1}:
            return 0.0
        return self.mu if x == 1 else (1 - self.mu)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        从伯努利分布中采样
        
        Args:
            size: 样本数量
            
        Returns:
            样本数组
        """
        return np.random.binomial(1, self.mu, size)
    
    def mean(self) -> float:
        """期望值"""
        return self.mu
    
    def variance(self) -> float:
        """方差"""
        return self.mu * (1 - self.mu)
    
    def entropy(self) -> float:
        """
        信息熵
        H = -μ*log(μ) - (1-μ)*log(1-μ)
        """
        if self.mu == 0 or self.mu == 1:
            return 0.0
        return -self.mu * np.log(self.mu) - (1 - self.mu) * np.log(1 - self.mu)


class BinomialDistribution:
    """
    二项分布：N次独立伯努利试验中成功的次数
    
    P(m|N,μ) = C(N,m) * μ^m * (1-μ)^(N-m)
    
    其中：
    - m: 成功次数，m ∈ {0, 1, ..., N}
    - N: 试验总次数
    - μ: 单次试验成功概率
    - C(N,m) = N!/(m!(N-m)!) 是组合数
    
    性质：
    - 期望：E[m] = Nμ
    - 方差：Var[m] = Nμ(1-μ)
    - 当N很大时，趋向于正态分布（中心极限定理）
    """
    
    def __init__(self, n: int, mu: float):
        """
        初始化二项分布
        
        Args:
            n: 试验次数
            mu: 单次成功概率
        """
        if n <= 0:
            raise ValueError(f"试验次数n必须为正整数，得到{n}")
        if not 0 <= mu <= 1:
            raise ValueError(f"概率mu必须在[0,1]之间，得到{mu}")
        
        self.n = n
        self.mu = mu
    
    def pmf(self, m: int) -> float:
        """
        概率质量函数
        
        使用对数技巧避免数值溢出：
        log P(m) = log C(N,m) + m*log(μ) + (N-m)*log(1-μ)
        """
        if not 0 <= m <= self.n:
            return 0.0
        
        # 使用scipy的二项分布
        return stats.binom.pmf(m, self.n, self.mu)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """从二项分布中采样"""
        return np.random.binomial(self.n, self.mu, size)
    
    def mean(self) -> float:
        """期望值"""
        return self.n * self.mu
    
    def variance(self) -> float:
        """方差"""
        return self.n * self.mu * (1 - self.mu)
    
    def plot(self, ax: Optional[plt.Axes] = None) -> None:
        """绘制二项分布"""
        if ax is None:
            fig, ax = plt.subplots()
        
        m_values = np.arange(0, self.n + 1)
        pmf_values = [self.pmf(m) for m in m_values]
        
        ax.bar(m_values, pmf_values, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('成功次数 m')
        ax.set_ylabel('概率 P(m)')
        ax.set_title(f'二项分布 B({self.n}, {self.mu})')
        ax.grid(True, alpha=0.3)
        
        # 标记期望值
        mean_val = self.mean()
        ax.axvline(x=mean_val, color='red', linestyle='--', 
                  label=f'期望={mean_val:.1f}')
        ax.legend()


class BetaDistribution:
    """
    贝塔分布：二项分布的共轭先验
    
    P(μ|a,b) = Γ(a+b)/(Γ(a)Γ(b)) * μ^(a-1) * (1-μ)^(b-1)
           = (1/B(a,b)) * μ^(a-1) * (1-μ)^(b-1)
    
    其中：
    - μ ∈ [0,1]
    - a, b > 0 是形状参数
    - B(a,b) = Γ(a)Γ(b)/Γ(a+b) 是贝塔函数
    - Γ是伽马函数
    
    特殊情况：
    - a=b=1: 均匀分布U(0,1)
    - a=b<1: U形分布
    - a=b>1: 钟形分布
    - a≠b: 偏斜分布
    
    性质：
    - 期望：E[μ] = a/(a+b)
    - 方差：Var[μ] = ab/((a+b)²(a+b+1))
    - 众数：(a-1)/(a+b-2) 当a,b>1时
    
    作为共轭先验的优势：
    贝塔分布是二项分布的共轭先验，这意味着：
    - 先验：Beta(a, b)
    - 似然：Binomial(n, μ)
    - 后验：Beta(a + m, b + n - m)
    其中m是n次试验中的成功次数
    """
    
    def __init__(self, a: float, b: float):
        """
        初始化贝塔分布
        
        Args:
            a: 第一个形状参数（成功的伪计数+1）
            b: 第二个形状参数（失败的伪计数+1）
        """
        if a <= 0 or b <= 0:
            raise ValueError(f"参数a和b必须为正数，得到a={a}, b={b}")
        
        self.a = a
        self.b = b
    
    def pdf(self, mu: np.ndarray) -> np.ndarray:
        """
        概率密度函数
        
        Args:
            mu: 取值点，必须在[0,1]之间
            
        Returns:
            概率密度值
        """
        # 确保mu是数组
        mu = np.asarray(mu)
        
        # 处理边界情况
        pdf_values = np.zeros_like(mu)
        valid_mask = (mu >= 0) & (mu <= 1)
        
        if np.any(valid_mask):
            # 使用scipy的贝塔分布
            pdf_values[valid_mask] = stats.beta.pdf(mu[valid_mask], self.a, self.b)
        
        return pdf_values
    
    def sample(self, size: int = 1) -> np.ndarray:
        """从贝塔分布中采样"""
        return np.random.beta(self.a, self.b, size)
    
    def mean(self) -> float:
        """期望值"""
        return self.a / (self.a + self.b)
    
    def variance(self) -> float:
        """方差"""
        ab_sum = self.a + self.b
        return (self.a * self.b) / (ab_sum ** 2 * (ab_sum + 1))
    
    def mode(self) -> Optional[float]:
        """
        众数（概率密度最大的点）
        
        只有当a>1且b>1时才存在内部众数
        """
        if self.a > 1 and self.b > 1:
            return (self.a - 1) / (self.a + self.b - 2)
        elif self.a < 1 and self.b < 1:
            # U形分布，两个众数在0和1
            return None
        elif self.a < 1:
            return 0.0
        elif self.b < 1:
            return 1.0
        else:
            # a=1或b=1的情况
            return None
    
    def update(self, m: int, n: int) -> 'BetaDistribution':
        """
        贝叶斯更新：给定二项观测数据，更新贝塔分布参数
        
        这展示了共轭先验的优美之处：
        后验 = Beta(a + m, b + n - m)
        
        Args:
            m: 成功次数
            n: 试验总次数
            
        Returns:
            更新后的贝塔分布
        """
        return BetaDistribution(self.a + m, self.b + n - m)


def demonstrate_bernoulli_distribution(mu_values: List[float], 
                                      n_samples: int = 1000,
                                      show_plot: bool = True) -> None:
    """
    演示伯努利分布的性质
    
    展示不同参数μ下的伯努利分布特性，
    包括采样、期望、方差和熵的变化。
    
    Args:
        mu_values: 要测试的μ值列表
        n_samples: 每个分布的采样数量
        show_plot: 是否显示图形
    """
    print("\n伯努利分布演示")
    print("=" * 60)
    print("伯努利分布是最简单的离散分布，描述单次二元试验的结果。")
    print("-" * 60)
    
    results = []
    for mu in mu_values:
        dist = BernoulliDistribution(mu)
        samples = dist.sample(n_samples)
        
        # 计算经验统计量
        empirical_mean = np.mean(samples)
        empirical_var = np.var(samples)
        
        # 理论值
        theoretical_mean = dist.mean()
        theoretical_var = dist.variance()
        entropy = dist.entropy()
        
        results.append({
            'mu': mu,
            'empirical_mean': empirical_mean,
            'theoretical_mean': theoretical_mean,
            'empirical_var': empirical_var,
            'theoretical_var': theoretical_var,
            'entropy': entropy
        })
        
        print(f"\nμ = {mu}:")
        print(f"  期望: 经验={empirical_mean:.3f}, 理论={theoretical_mean:.3f}")
        print(f"  方差: 经验={empirical_var:.3f}, 理论={theoretical_var:.3f}")
        print(f"  熵: {entropy:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1：期望随μ的变化
        ax1 = axes[0]
        mus = [r['mu'] for r in results]
        means = [r['theoretical_mean'] for r in results]
        ax1.plot(mus, means, 'b-o', linewidth=2)
        ax1.set_xlabel('参数 μ')
        ax1.set_ylabel('期望 E[x]')
        ax1.set_title('伯努利分布的期望')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # 子图2：方差随μ的变化
        ax2 = axes[1]
        vars = [r['theoretical_var'] for r in results]
        ax2.plot(mus, vars, 'r-o', linewidth=2)
        ax2.set_xlabel('参数 μ')
        ax2.set_ylabel('方差 Var[x]')
        ax2.set_title('伯努利分布的方差')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.5, color='green', linestyle='--', 
                   label='最大方差位置')
        ax2.legend()
        ax2.set_xlim([0, 1])
        
        # 子图3：熵随μ的变化
        ax3 = axes[2]
        entropies = [r['entropy'] for r in results]
        
        # 绘制完整的熵曲线
        mu_fine = np.linspace(0.001, 0.999, 100)
        entropy_fine = [-m*np.log(m) - (1-m)*np.log(1-m) for m in mu_fine]
        ax3.plot(mu_fine, entropy_fine, 'g-', linewidth=2, alpha=0.5)
        ax3.plot(mus, entropies, 'go', markersize=8)
        ax3.set_xlabel('参数 μ')
        ax3.set_ylabel('熵 H(x)')
        ax3.set_title('伯努利分布的熵')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0.5, color='red', linestyle='--', 
                   label='最大熵位置')
        ax3.legend()
        ax3.set_xlim([0, 1])
        
        plt.suptitle('伯努利分布的性质', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 期望E[x] = μ，线性增长")
    print("2. 方差在μ=0.5时达到最大值0.25")
    print("3. 熵在μ=0.5时达到最大（最大不确定性）")


def demonstrate_beta_distribution(param_pairs: List[Tuple[float, float]],
                                 n_points: int = 200,
                                 show_plot: bool = True) -> None:
    """
    演示贝塔分布的不同形状
    
    贝塔分布是非常灵活的分布，可以表示多种形状：
    - U形、均匀、钟形、偏斜等
    
    Args:
        param_pairs: (a, b)参数对列表
        n_points: 绘图点数
        show_plot: 是否显示图形
    """
    print("\n贝塔分布演示")
    print("=" * 60)
    print("贝塔分布Beta(a,b)是[0,1]区间上的连续分布，")
    print("是二项分布的共轭先验，在贝叶斯推断中极其重要。")
    print("-" * 60)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        mu = np.linspace(0.001, 0.999, n_points)
        
        for idx, (a, b) in enumerate(param_pairs):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dist = BetaDistribution(a, b)
            
            # 计算PDF
            pdf_values = dist.pdf(mu)
            
            # 绘制分布
            ax.plot(mu, pdf_values, 'b-', linewidth=2)
            ax.fill_between(mu, pdf_values, alpha=0.3)
            
            # 标记统计量
            mean_val = dist.mean()
            ax.axvline(x=mean_val, color='red', linestyle='--', 
                      linewidth=1, label=f'均值={mean_val:.2f}')
            
            mode_val = dist.mode()
            if mode_val is not None and 0 < mode_val < 1:
                ax.axvline(x=mode_val, color='green', linestyle='--', 
                          linewidth=1, label=f'众数={mode_val:.2f}')
            
            # 设置图形属性
            ax.set_xlabel('μ')
            ax.set_ylabel('概率密度')
            ax.set_title(f'Beta({a}, {b})')
            ax.set_xlim([0, 1])
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # 打印统计信息
            print(f"\nBeta({a}, {b}):")
            print(f"  均值: {mean_val:.3f}")
            print(f"  方差: {dist.variance():.3f}")
            if mode_val is not None:
                print(f"  众数: {mode_val:.3f}")
            
            # 描述分布形状
            if a == b == 1:
                print("  形状: 均匀分布")
            elif a == b < 1:
                print("  形状: U形分布")
            elif a == b > 1:
                print("  形状: 对称钟形")
            elif a > b:
                print("  形状: 右偏")
            elif a < b:
                print("  形状: 左偏")
        
        # 隐藏多余的子图
        for idx in range(len(param_pairs), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('贝塔分布的不同形状', fontsize=14)
        plt.tight_layout()
        plt.show()


def demonstrate_bayesian_inference(true_mu: float = 0.7,
                                  prior_a: float = 2,
                                  prior_b: float = 2,
                                  data_sizes: List[int] = [1, 5, 10, 50],
                                  show_plot: bool = True) -> None:
    """
    演示贝叶斯推断：使用贝塔-二项共轭
    
    这是贝叶斯推断的经典例子：
    1. 从先验分布Beta(a, b)开始
    2. 观察二项数据（如抛硬币结果）
    3. 使用贝叶斯定理更新为后验分布Beta(a+m, b+n-m)
    
    随着数据增加，后验分布会：
    - 变得更加集中（方差减小）
    - 逐渐接近真实参数
    - 受先验影响越来越小
    
    Args:
        true_mu: 真实的成功概率
        prior_a: 先验贝塔分布的a参数
        prior_b: 先验贝塔分布的b参数
        data_sizes: 不同的数据量
        show_plot: 是否显示图形
    """
    print("\n贝叶斯推断演示：贝塔-二项共轭")
    print("=" * 60)
    print(f"真实参数: μ = {true_mu}")
    print(f"先验分布: Beta({prior_a}, {prior_b})")
    print("-" * 60)
    
    # 生成数据
    np.random.seed(42)
    max_n = max(data_sizes)
    all_data = np.random.binomial(1, true_mu, max_n)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        mu_range = np.linspace(0.001, 0.999, 200)
        
        # 先验分布
        prior = BetaDistribution(prior_a, prior_b)
        prior_pdf = prior.pdf(mu_range)
        
        for idx, n in enumerate([0] + data_sizes):
            ax = axes[idx]
            
            if n == 0:
                # 绘制先验
                ax.plot(mu_range, prior_pdf, 'b-', linewidth=2, label='先验')
                ax.fill_between(mu_range, prior_pdf, alpha=0.3)
                title = f'先验分布\nBeta({prior_a}, {prior_b})'
                current_dist = prior
            else:
                # 获取数据
                data = all_data[:n]
                m = np.sum(data)  # 成功次数
                
                # 计算后验
                posterior = prior.update(m, n)
                posterior_pdf = posterior.pdf(mu_range)
                
                # 绘制
                ax.plot(mu_range, prior_pdf, 'b--', linewidth=1, 
                       alpha=0.5, label='先验')
                ax.plot(mu_range, posterior_pdf, 'r-', linewidth=2, 
                       label='后验')
                ax.fill_between(mu_range, posterior_pdf, alpha=0.3, color='red')
                
                title = f'n={n}次观测后\n成功{m}次，失败{n-m}次\n' \
                       f'后验: Beta({posterior.a:.0f}, {posterior.b:.0f})'
                current_dist = posterior
            
            # 标记真实值
            ax.axvline(x=true_mu, color='green', linestyle='-', 
                      linewidth=2, alpha=0.7, label=f'真实μ={true_mu}')
            
            # 标记后验均值
            mean_val = current_dist.mean()
            ax.axvline(x=mean_val, color='orange', linestyle='--', 
                      linewidth=1, label=f'后验均值={mean_val:.3f}')
            
            # 设置图形
            ax.set_xlabel('μ')
            ax.set_ylabel('概率密度')
            ax.set_title(title)
            ax.set_xlim([0, 1])
            ax.set_ylim(bottom=0)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # 打印统计信息
            if n > 0:
                print(f"\n观测{n}个数据点后:")
                print(f"  数据: {m}次成功，{n-m}次失败")
                print(f"  后验: Beta({posterior.a}, {posterior.b})")
                print(f"  后验均值: {mean_val:.3f}")
                print(f"  后验方差: {current_dist.variance():.4f}")
                print(f"  与真实值的误差: {abs(mean_val - true_mu):.3f}")
        
        # 隐藏多余的子图
        for idx in range(len(data_sizes) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('贝叶斯推断：从先验到后验的演化', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n关键观察：")
    print("1. 后验分布随数据增加而变窄（更确定）")
    print("2. 后验均值逐渐接近真实参数")
    print("3. 先验的影响随数据增加而减弱")
    print("4. 共轭先验使得后验有封闭形式解")