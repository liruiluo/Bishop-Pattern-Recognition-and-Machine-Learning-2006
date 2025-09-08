"""
2.2 多项式变量 (Multinomial Variables)
======================================

多项式变量是二元变量的推广，可以取K个可能的状态之一。
例如：骰子（K=6）、多类分类（K个类别）、词袋模型（K个词）等。

本节涵盖：
1. 多项分布 (Multinomial)：多次试验的结果
2. 狄利克雷分布 (Dirichlet)：多项分布的共轭先验

共轭关系：
- 似然：多项分布
- 先验：狄利克雷分布
- 后验：狄利克雷分布（参数更新）

狄利克雷分布是贝塔分布在多维情况下的推广，
在主题模型（如LDA）、贝叶斯网络等领域广泛应用。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, loggamma
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def one_hot_encoding(x: int, K: int) -> np.ndarray:
    """
    将类别索引转换为one-hot向量
    
    例如：x=2, K=5 -> [0, 0, 1, 0, 0]
    
    这种编码在神经网络和概率模型中非常常见，
    因为它将离散类别转换为向量形式，便于数学处理。
    
    Args:
        x: 类别索引，范围[0, K-1]
        K: 类别总数
        
    Returns:
        K维one-hot向量
    """
    vec = np.zeros(K)
    vec[x] = 1
    return vec


class MultinomialDistribution:
    """
    多项分布：N次独立试验，每次试验有K个可能结果
    
    这是二项分布的多类推广。
    
    P(m1,...,mK|N,μ) = (N!)/(m1!...mK!) * ∏(μk^mk)
    
    其中：
    - N: 试验总次数
    - K: 可能结果的数量
    - mk: 第k个结果出现的次数，∑mk = N
    - μk: 第k个结果的概率，∑μk = 1
    
    性质：
    - 期望：E[mk] = N*μk
    - 方差：Var[mk] = N*μk*(1-μk)
    - 协方差：Cov[mi,mj] = -N*μi*μj (i≠j)
    
    应用：
    - 文本分析：词频统计
    - 图像处理：颜色直方图
    - 生物信息：DNA序列分析
    """
    
    def __init__(self, n: int, probs: np.ndarray):
        """
        初始化多项分布
        
        Args:
            n: 试验次数
            probs: 各类别的概率，必须和为1
        """
        if n <= 0:
            raise ValueError(f"试验次数n必须为正整数，得到{n}")
        
        probs = np.asarray(probs)
        if not np.allclose(np.sum(probs), 1.0):
            raise ValueError(f"概率必须和为1，得到{np.sum(probs)}")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("概率必须在[0,1]之间")
        
        self.n = n
        self.probs = probs
        self.K = len(probs)
    
    def pmf(self, counts: np.ndarray) -> float:
        """
        概率质量函数
        
        使用对数技巧避免数值问题：
        log P = log(N!) - ∑log(mk!) + ∑(mk*log(μk))
        
        Args:
            counts: 各类别出现次数，必须和为n
            
        Returns:
            概率值
        """
        counts = np.asarray(counts)
        
        # 检查约束
        if len(counts) != self.K:
            raise ValueError(f"计数向量长度必须为{self.K}")
        if np.sum(counts) != self.n:
            return 0.0
        if np.any(counts < 0):
            return 0.0
        
        # 使用scipy的多项分布
        return stats.multinomial.pmf(counts, self.n, self.probs)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        从多项分布中采样
        
        Returns:
            采样结果，shape (size, K)
        """
        return np.random.multinomial(self.n, self.probs, size)
    
    def mean(self) -> np.ndarray:
        """期望向量"""
        return self.n * self.probs
    
    def covariance(self) -> np.ndarray:
        """
        协方差矩阵
        
        对角元素：Var[mk] = N*μk*(1-μk)
        非对角元素：Cov[mi,mj] = -N*μi*μj
        """
        cov = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    cov[i, j] = self.n * self.probs[i] * (1 - self.probs[i])
                else:
                    cov[i, j] = -self.n * self.probs[i] * self.probs[j]
        return cov


class DirichletDistribution:
    """
    狄利克雷分布：多项分布的共轭先验
    
    这是贝塔分布在K维单纯形上的推广。
    
    P(μ|α) = (Γ(∑αk)/∏Γ(αk)) * ∏(μk^(αk-1))
           = (1/B(α)) * ∏(μk^(αk-1))
    
    其中：
    - μ = (μ1,...,μK): 概率向量，∑μk = 1, μk ≥ 0
    - α = (α1,...,αK): 浓度参数，αk > 0
    - B(α): 多元贝塔函数
    
    性质：
    - 期望：E[μk] = αk/α0，其中α0 = ∑αk
    - 方差：Var[μk] = αk(α0-αk)/(α0²(α0+1))
    - 浓度参数α0控制分布的集中程度
    
    特殊情况：
    - α = (1,1,...,1): 均匀分布
    - α很大: 集中在均值附近
    - α很小: 集中在单纯形的角落
    - α不等: 偏向某些类别
    
    应用：
    - LDA主题模型：文档-主题分布、主题-词分布的先验
    - 贝叶斯多项回归
    - 混合模型的混合权重先验
    """
    
    def __init__(self, alpha: np.ndarray):
        """
        初始化狄利克雷分布
        
        Args:
            alpha: 浓度参数向量，所有元素必须为正
        """
        alpha = np.asarray(alpha)
        if np.any(alpha <= 0):
            raise ValueError("所有alpha参数必须为正数")
        
        self.alpha = alpha
        self.K = len(alpha)
        self.alpha0 = np.sum(alpha)  # 总浓度
    
    def pdf(self, mu: np.ndarray) -> float:
        """
        概率密度函数
        
        只在单纯形上有定义：∑μk = 1, μk ≥ 0
        
        Args:
            mu: K维概率向量
            
        Returns:
            概率密度值
        """
        mu = np.asarray(mu)
        
        # 检查是否在单纯形上
        if len(mu) != self.K:
            raise ValueError(f"向量维度必须为{self.K}")
        if not np.allclose(np.sum(mu), 1.0):
            return 0.0
        if np.any(mu < 0) or np.any(mu > 1):
            return 0.0
        
        # 使用对数避免数值问题
        # log p = log(Γ(α0)) - ∑log(Γ(αk)) + ∑((αk-1)*log(μk))
        log_p = loggamma(self.alpha0)
        log_p -= np.sum(loggamma(self.alpha))
        log_p += np.sum((self.alpha - 1) * np.log(mu + 1e-10))
        
        return np.exp(log_p)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        从狄利克雷分布中采样
        
        使用伽马分布的性质：
        如果Xk ~ Gamma(αk, 1)，那么
        μk = Xk / ∑Xj ~ Dirichlet(α)
        
        Returns:
            采样结果，shape (size, K)
        """
        # 从伽马分布采样
        gamma_samples = np.random.gamma(self.alpha, 1, (size, self.K))
        # 归一化得到狄利克雷样本
        return gamma_samples / gamma_samples.sum(axis=1, keepdims=True)
    
    def mean(self) -> np.ndarray:
        """期望向量"""
        return self.alpha / self.alpha0
    
    def variance(self) -> np.ndarray:
        """方差向量（对角元素）"""
        mean = self.mean()
        return mean * (1 - mean) / (self.alpha0 + 1)
    
    def mode(self) -> Optional[np.ndarray]:
        """
        众数（密度最大的点）
        
        只有当所有αk > 1时才存在内部众数
        """
        if np.all(self.alpha > 1):
            return (self.alpha - 1) / (self.alpha0 - self.K)
        return None
    
    def update(self, counts: np.ndarray) -> 'DirichletDistribution':
        """
        贝叶斯更新：给定多项观测数据，更新狄利克雷分布
        
        后验 = Dirichlet(α + m)
        其中m是观测计数向量
        
        Args:
            counts: 各类别的观测次数
            
        Returns:
            更新后的狄利克雷分布
        """
        return DirichletDistribution(self.alpha + counts)
    
    def entropy(self) -> float:
        """
        计算分布的熵
        
        H = log B(α) + (α0 - K)ψ(α0) - ∑(αk - 1)ψ(αk)
        其中ψ是digamma函数
        """
        from scipy.special import digamma
        
        # log B(α)
        log_beta = np.sum(loggamma(self.alpha)) - loggamma(self.alpha0)
        
        # 熵计算
        entropy = log_beta
        entropy += (self.alpha0 - self.K) * digamma(self.alpha0)
        entropy -= np.sum((self.alpha - 1) * digamma(self.alpha))
        
        return entropy


def visualize_dirichlet_on_simplex(alpha_values: List[np.ndarray],
                                   n_samples: int = 1000,
                                   show_plot: bool = True) -> None:
    """
    在2-单纯形（三角形）上可视化3维狄利克雷分布
    
    2-单纯形是满足x+y+z=1, x,y,z≥0的点集，
    可以投影到2D平面上进行可视化。
    
    使用重心坐标系：
    - 三个顶点代表三个纯类别
    - 中心点代表均匀分布
    - 点的位置反映了三个类别的相对概率
    
    Args:
        alpha_values: 不同的α参数组合
        n_samples: 每个分布的采样数
        show_plot: 是否显示图形
    """
    print("\n狄利克雷分布在单纯形上的可视化")
    print("=" * 60)
    
    if not show_plot:
        return
    
    # 只处理3维情况（K=3）
    alpha_values = [a for a in alpha_values if len(a) == 3]
    
    if not alpha_values:
        print("需要3维的alpha参数才能在2D单纯形上可视化")
        return
    
    n_plots = len(alpha_values)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 三角形顶点的2D坐标
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    def to_barycentric_2d(points):
        """将概率向量转换为重心坐标的2D投影"""
        # points shape: (n, 3)
        # 每个点是三个顶点的加权平均
        return points @ vertices
    
    for idx, alpha in enumerate(alpha_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 从狄利克雷分布采样
        dist = DirichletDistribution(alpha)
        samples = dist.sample(n_samples)
        
        # 转换为2D坐标
        points_2d = to_barycentric_2d(samples)
        
        # 绘制三角形边界
        triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # 绘制采样点
        ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                  alpha=0.5, s=5, c='blue')
        
        # 标记顶点
        labels = ['类1', '类2', '类3']
        for i, (vertex, label) in enumerate(zip(vertices, labels)):
            ax.plot(vertex[0], vertex[1], 'ro', markersize=10)
            # 调整标签位置使其在三角形外
            offset = vertex - np.array([0.5, np.sqrt(3)/6])
            offset = offset / np.linalg.norm(offset) * 0.1
            ax.text(vertex[0] + offset[0], vertex[1] + offset[1], 
                   label, fontsize=12, ha='center')
        
        # 标记均值
        mean = dist.mean()
        mean_2d = to_barycentric_2d(mean.reshape(1, -1))[0]
        ax.plot(mean_2d[0], mean_2d[1], 'r*', markersize=15, 
               label='均值')
        
        # 设置标题和属性
        ax.set_title(f'Dirichlet({alpha[0]}, {alpha[1]}, {alpha[2]})')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(loc='upper right')
        
        # 打印统计信息
        print(f"\nDirichlet({alpha}):")
        print(f"  均值: {mean}")
        print(f"  总浓度α0: {dist.alpha0:.1f}")
        print(f"  熵: {dist.entropy():.3f}")
        
        # 描述分布特征
        if np.allclose(alpha, alpha[0]):
            if alpha[0] == 1:
                print("  特征: 均匀分布在整个单纯形")
            elif alpha[0] < 1:
                print("  特征: 集中在顶点（稀疏）")
            else:
                print("  特征: 集中在中心（密集）")
        else:
            max_idx = np.argmax(alpha)
            print(f"  特征: 偏向类{max_idx+1}")
    
    # 隐藏多余的子图
    for idx in range(len(alpha_values), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('狄利克雷分布在2-单纯形上的可视化', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n观察：")
    print("1. α相等且<1：样本集中在顶点（稀疏）")
    print("2. α相等且=1：均匀分布")
    print("3. α相等且>1：样本集中在中心")
    print("4. α不等：样本偏向较大α对应的类别")


def demonstrate_dirichlet_bayesian_update(true_probs: np.ndarray = np.array([0.2, 0.3, 0.5]),
                                         prior_alpha: np.ndarray = np.array([2, 2, 2]),
                                         n_observations: List[int] = [10, 50, 100, 500],
                                         show_plot: bool = True) -> None:
    """
    演示狄利克雷-多项共轭的贝叶斯更新
    
    这展示了如何使用狄利克雷分布作为多项分布的先验，
    并通过观测数据更新后验分布。
    
    Args:
        true_probs: 真实的类别概率
        prior_alpha: 先验狄利克雷分布参数
        n_observations: 不同的观测数量
        show_plot: 是否显示图形
    """
    print("\n贝叶斯更新：狄利克雷-多项共轭")
    print("=" * 60)
    print(f"真实概率: {true_probs}")
    print(f"先验: Dirichlet({prior_alpha})")
    print("-" * 60)
    
    K = len(true_probs)
    np.random.seed(42)
    
    # 生成所有数据
    max_n = max(n_observations)
    all_data = np.random.multinomial(1, true_probs, max_n)
    
    results = []
    
    for n in n_observations:
        # 获取n个观测
        data = all_data[:n]
        counts = np.sum(data, axis=0)
        
        # 更新后验
        prior = DirichletDistribution(prior_alpha)
        posterior = prior.update(counts)
        
        # 计算统计量
        posterior_mean = posterior.mean()
        posterior_var = posterior.variance()
        
        # KL散度（与真实分布的差异）
        kl_div = np.sum(true_probs * np.log(true_probs / (posterior_mean + 1e-10)))
        
        results.append({
            'n': n,
            'counts': counts,
            'posterior_alpha': posterior.alpha,
            'posterior_mean': posterior_mean,
            'posterior_var': posterior_var,
            'kl_divergence': kl_div
        })
        
        print(f"\n观测{n}个数据后:")
        print(f"  观测计数: {counts}")
        print(f"  后验: Dirichlet({posterior.alpha})")
        print(f"  后验均值: {posterior_mean}")
        print(f"  与真实分布的KL散度: {kl_div:.4f}")
    
    if show_plot and K == 3:
        # 只对3维情况进行可视化
        fig, axes = plt.subplots(1, len(n_observations), figsize=(5*len(n_observations), 5))
        
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        
        def to_barycentric_2d(points):
            return points @ vertices
        
        for idx, (n, res) in enumerate(zip(n_observations, results)):
            ax = axes[idx]
            
            # 绘制三角形
            triangle = plt.Polygon(vertices, fill=False, 
                                  edgecolor='black', linewidth=2)
            ax.add_patch(triangle)
            
            # 采样并绘制后验分布
            posterior = DirichletDistribution(res['posterior_alpha'])
            samples = posterior.sample(500)
            points_2d = to_barycentric_2d(samples)
            ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                      alpha=0.3, s=10, c='blue')
            
            # 标记真实值
            true_2d = to_barycentric_2d(true_probs.reshape(1, -1))[0]
            ax.plot(true_2d[0], true_2d[1], 'g*', markersize=20, 
                   label='真实')
            
            # 标记后验均值
            mean_2d = to_barycentric_2d(res['posterior_mean'].reshape(1, -1))[0]
            ax.plot(mean_2d[0], mean_2d[1], 'r^', markersize=15, 
                   label='后验均值')
            
            # 设置属性
            ax.set_title(f'n={n}\nKL={res["kl_divergence"]:.3f}')
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, 1.0)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend()
        
        plt.suptitle('狄利克雷后验随数据增加的演化', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n关键观察：")
    print("1. 后验分布随数据增加变得更集中")
    print("2. 后验均值逐渐接近真实概率")
    print("3. KL散度逐渐减小")
    print("4. 先验的影响随数据增加而减弱")