"""
1.6 信息论 (Information Theory)
=================================

信息论提供了量化信息的数学框架。

核心概念：
1. 熵 (Entropy)：随机变量的不确定性
2. 条件熵：给定一个变量后另一个变量的不确定性
3. 互信息：两个变量之间的信息共享量
4. KL散度：两个分布之间的差异

这些概念在特征选择、模型选择和优化中都很重要。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import entropy as scipy_entropy
from scipy.stats import uniform, norm, expon


def compute_entropy(probs: np.ndarray) -> float:
    """
    计算离散分布的香农熵
    
    H(X) = -Σ p(x) log p(x)
    
    熵的性质：
    - 非负性： H(X) ≥ 0
    - 均匀分布时熵最大
    - 确定事件熵为0
    
    Args:
        probs: 概率分布
        
    Returns:
        熵值（以自然对数为底）
    """
    # 避免log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def entropy_demo(
    distributions: List[str] = ['uniform', 'gaussian', 'exponential'],
    n_bins: int = 20,
    show_plot: bool = True
) -> None:
    """
    演示不同分布的熵
    
    熵反映了分布的“不确定性”或“信息量”：
    - 均匀分布：熵最大（最不确定）
    - 独热分布：熵最小（最确定）
    
    Args:
        distributions: 要演示的分布列表
        n_bins: 离散化的bin数量
        show_plot: 是否显示图形
    """
    print("\n不同分布的熵：")
    print("=" * 40)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
    for idx, dist_name in enumerate(distributions):
        # 生成样本
        if dist_name == 'uniform':
            samples = np.random.uniform(0, 1, 10000)
            theoretical_entropy = np.log(1)  # 连续均匀分布的微分熵
        elif dist_name == 'gaussian':
            samples = np.random.normal(0, 1, 10000)
            theoretical_entropy = 0.5 * np.log(2 * np.pi * np.e)  # 正态分布的微分熵
        elif dist_name == 'exponential':
            samples = np.random.exponential(1, 10000)
            theoretical_entropy = 1  # 指数分布的微分熵
        else:
            continue
        
        # 离散化
        hist, bin_edges = np.histogram(samples, bins=n_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        probs = hist * bin_width
        probs = probs / probs.sum()  # 归一化
        
        # 计算熵
        empirical_entropy = compute_entropy(probs)
        
        print(f"\n{dist_name.capitalize()} 分布:")
        print(f"  经验熵 (bins={n_bins}): {empirical_entropy:.4f}")
        print(f"  理论微分熵: {theoretical_entropy:.4f}")
        
        if show_plot and idx < len(axes):
            ax = axes[idx]
            
            # 绘制直方图
            ax.hist(samples, bins=50, density=True, alpha=0.7,
                   color='blue', edgecolor='black')
            
            # 绘制理论分布
            x_range = np.linspace(samples.min(), samples.max(), 100)
            if dist_name == 'uniform':
                pdf = uniform.pdf(x_range, 0, 1)
            elif dist_name == 'gaussian':
                pdf = norm.pdf(x_range, 0, 1)
            elif dist_name == 'exponential':
                pdf = expon.pdf(x_range, scale=1)
            
            ax.plot(x_range, pdf, 'r-', linewidth=2, label='理论分布')
            ax.set_xlabel('值')
            ax.set_ylabel('概率密度')
            ax.set_title(f'{dist_name.capitalize()} 分布\nH={empirical_entropy:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 演示熵随分布参数的变化
    if show_plot and len(axes) > len(distributions):
        # 二项分布的熵随p的关系
        ax = axes[3]
        p_values = np.linspace(0.001, 0.999, 100)
        entropies = [-p*np.log(p) - (1-p)*np.log(1-p) for p in p_values]
        
        ax.plot(p_values, entropies, 'b-', linewidth=2)
        ax.set_xlabel('成功概率 p')
        ax.set_ylabel('熵 H(p)')
        ax.set_title('二项分布的熵')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='r', linestyle='--', label='p=0.5 (最大熵)')
        ax.legend()
        
        # 多项分布的熵
        ax = axes[4]
        n_outcomes = [2, 3, 4, 5, 10, 20, 50, 100]
        max_entropies = [np.log(n) for n in n_outcomes]
        
        ax.plot(n_outcomes, max_entropies, 'g-o', linewidth=2)
        ax.set_xlabel('结果数量 n')
        ax.set_ylabel('最大熵 log(n)')
        ax.set_title('均匀分布的熵隌结果数的关系')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(5, len(axes)):
            axes[idx].axis('off')
    
    if show_plot:
        plt.suptitle('信息论：熵的概念和性质', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n熵的重要性质：")
    print("1. 熵是不确定性的度量")
    print("2. 均匀分布有最大熵")
    print("3. 熵可以用于特征选择和决策树构建")


def mutual_information_demo() -> None:
    """
    演示互信息的概念
    
    互信息 I(X;Y) 表示两个变量共享的信息量：
    I(X;Y) = H(X) + H(Y) - H(X,Y)
           = H(X) - H(X|Y)
           = H(Y) - H(Y|X)
    
    性质：
    - 非负性：I(X;Y) ≥ 0
    - 对称性：I(X;Y) = I(Y;X)
    - 独立变量的互信息为0
    """
    print("\n互信息演示：")
    print("=" * 40)
    
    # 创建两个相关变量
    n_samples = 1000
    
    # 情况1：完全独立
    X_indep = np.random.normal(0, 1, n_samples)
    Y_indep = np.random.normal(0, 1, n_samples)
    
    # 情况2：部分相关
    X_partial = np.random.normal(0, 1, n_samples)
    Y_partial = 0.7 * X_partial + 0.3 * np.random.normal(0, 1, n_samples)
    
    # 情况3：完全相关
    X_full = np.random.normal(0, 1, n_samples)
    Y_full = 2 * X_full + 1
    
    # 计算互信息（简化版，使用离散化）
    def estimate_mi(X, Y, n_bins=10):
        # 离散化
        hist_2d, x_edges, y_edges = np.histogram2d(X, Y, bins=n_bins)
        hist_2d = hist_2d / hist_2d.sum()
        
        # 边缘分布
        p_x = hist_2d.sum(axis=1)
        p_y = hist_2d.sum(axis=0)
        
        # 计算互信息
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    mi_indep = estimate_mi(X_indep, Y_indep)
    mi_partial = estimate_mi(X_partial, Y_partial)
    mi_full = estimate_mi(X_full, Y_full)
    
    print(f"独立变量的互信息: {mi_indep:.4f}")
    print(f"部分相关变量的互信息: {mi_partial:.4f}")
    print(f"完全相关变量的互信息: {mi_full:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 独立变量
    axes[0].scatter(X_indep, Y_indep, alpha=0.5, s=10)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'独立变量\nMI={mi_indep:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # 部分相关
    axes[1].scatter(X_partial, Y_partial, alpha=0.5, s=10)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title(f'部分相关\nMI={mi_partial:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    # 完全相关
    axes[2].scatter(X_full, Y_full, alpha=0.5, s=10)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title(f'完全相关\nMI={mi_full:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('互信息：量化变量间的依赖关系', fontsize=14)
    plt.tight_layout()
    plt.show()