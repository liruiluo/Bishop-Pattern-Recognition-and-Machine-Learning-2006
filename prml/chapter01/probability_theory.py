"""
1.2 概率论回顾 (Probability Theory)
====================================

概率论是机器学习的数学基础。几乎所有的机器学习算法
都可以从概率的角度来理解。这一节回顾最重要的概率概念。

核心概念：
1. 概率的基本规则（加法规则、乘法规则）
2. 贝叶斯定理 - 机器学习的核心
3. 概率密度函数
4. 期望和方差
5. 协方差和相关性

理解这些概念对于后续章节至关重要。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Tuple


def demonstrate_bayes_theorem(
    prior: float = 0.01,
    likelihood: float = 0.99,
    false_positive: float = 0.05
) -> None:
    """
    贝叶斯定理演示 - 以医疗诊断为例
    
    贝叶斯定理是机器学习的核心：
    P(θ|D) = P(D|θ) × P(θ) / P(D)
    
    其中：
    - P(θ|D) 是后验概率（给定数据后参数的概率）
    - P(D|θ) 是似然（给定参数时数据的概率）
    - P(θ) 是先验概率（参数的初始信念）
    - P(D) 是证据（数据的边际概率）
    
    医疗诊断例子：
    假设某种疾病在人群中的患病率是1%（先验）。
    有一种检测方法，对真正患病的人，99%会显示阳性（真阳性率）。
    对健康的人，5%会错误地显示阳性（假阳性率）。
    
    问题：如果某人检测结果为阳性，他真正患病的概率是多少？
    
    这个例子展示了一个反直觉的结果：即使检测很准确，
    由于疾病本身很罕见，阳性结果可能仍然是假阳性！
    
    Args:
        prior: 疾病的患病率 P(患病)
        likelihood: 真阳性率 P(阳性|患病)
        false_positive: 假阳性率 P(阳性|健康)
    """
    # 计算各种概率
    # P(健康) = 1 - P(患病)
    prior_healthy = 1 - prior
    
    # 使用全概率公式计算 P(阳性)
    # P(阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)
    evidence = likelihood * prior + false_positive * prior_healthy
    
    # 使用贝叶斯定理计算后验概率
    # P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
    posterior = (likelihood * prior) / evidence
    
    print("贝叶斯定理应用：医疗诊断")
    print("-" * 40)
    print(f"先验概率 P(患病) = {prior:.1%}")
    print(f"似然 P(阳性|患病) = {likelihood:.1%}")
    print(f"假阳性率 P(阳性|健康) = {false_positive:.1%}")
    print("-" * 40)
    print(f"证据 P(阳性) = {evidence:.3%}")
    print(f"后验概率 P(患病|阳性) = {posterior:.1%}")
    print("-" * 40)
    
    # 解释结果
    if posterior < 0.5:
        print(f"注意：即使检测呈阳性，实际患病概率只有{posterior:.1%}！")
        print("这是因为疾病本身很罕见（基率谬误）。")
        print("这说明了考虑先验概率的重要性。")
    
    # 可视化贝叶斯更新过程
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1：先验、似然和后验
    ax1 = axes[0]
    categories = ['先验\nP(患病)', '似然\nP(阳性|患病)', '后验\nP(患病|阳性)']
    values = [prior, likelihood, posterior]
    colors = ['blue', 'green', 'red']
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('概率')
    ax1.set_title('贝叶斯定理：从先验到后验')
    ax1.set_ylim([0, 1])
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom')
    
    # 子图2：人群分布
    ax2 = axes[1]
    population = 10000
    sick = int(population * prior)
    healthy = population - sick
    true_positive = int(sick * likelihood)
    false_positive_count = int(healthy * false_positive)
    
    # 绘制人群分布
    labels = ['真阳性', '假阴性', '假阳性', '真阴性']
    sizes = [true_positive, sick - true_positive, 
             false_positive_count, healthy - false_positive_count]
    colors_pie = ['red', 'lightcoral', 'orange', 'lightgreen']
    explode = (0.1, 0, 0.1, 0)  # 突出显示阳性结果
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.0f', shadow=True, startangle=90)
    ax2.set_title(f'10000人中的检测结果分布')
    
    # 子图3：多次检测的效果
    ax3 = axes[2]
    n_tests = 5
    probs = [prior]
    for i in range(n_tests):
        # 每次阳性结果后更新概率
        current_prior = probs[-1]
        evidence = likelihood * current_prior + false_positive * (1 - current_prior)
        posterior = (likelihood * current_prior) / evidence
        probs.append(posterior)
    
    ax3.plot(range(n_tests + 1), probs, 'b-o', linewidth=2, markersize=8)
    ax3.set_xlabel('阳性检测次数')
    ax3.set_ylabel('P(患病|连续阳性)')
    ax3.set_title('多次独立检测的贝叶斯更新')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    for i, prob in enumerate(probs):
        ax3.text(i, prob + 0.02, f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def visualize_distributions(
    distributions_config: Dict[str, Any],
    show_plot: bool = True
) -> None:
    """
    可视化常见的概率分布
    
    这个函数展示机器学习中最常用的概率分布：
    1. 高斯分布（正态分布）- 最重要的连续分布
    2. 二项分布 - 离散分布的代表
    3. 贝塔分布 - 共轭先验的例子
    
    理解这些分布的性质对于选择合适的模型至关重要。
    
    Args:
        distributions_config: 分布的配置参数
        show_plot: 是否显示图形
    """
    if not show_plot:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 高斯分布（正态分布）
    ax1 = axes[0, 0]
    gaussian_cfg = distributions_config.get('gaussian', {})
    mean = gaussian_cfg.get('mean', 0)
    std = gaussian_cfg.get('std', 1)
    n_samples = gaussian_cfg.get('n_samples', 1000)
    
    # 生成样本
    samples_gaussian = np.random.normal(mean, std, n_samples)
    
    # 绘制直方图和理论曲线
    ax1.hist(samples_gaussian, bins=50, density=True, alpha=0.7, 
             color='blue', edgecolor='black', label='样本分布')
    
    # 理论概率密度函数
    x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
    pdf_gaussian = stats.norm.pdf(x_range, mean, std)
    ax1.plot(x_range, pdf_gaussian, 'r-', linewidth=2, label='理论PDF')
    
    ax1.set_title(f'高斯分布 N({mean}, {std}²)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('概率密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax1.text(0.02, 0.98, f'均值: {np.mean(samples_gaussian):.3f}\n'
                         f'标准差: {np.std(samples_gaussian):.3f}',
             transform=ax1.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 二项分布
    ax2 = axes[0, 1]
    binomial_cfg = distributions_config.get('binomial', {})
    n = binomial_cfg.get('n', 10)
    p = binomial_cfg.get('p', 0.3)
    n_samples = binomial_cfg.get('n_samples', 1000)
    
    # 生成样本
    samples_binomial = np.random.binomial(n, p, n_samples)
    
    # 绘制频率分布
    unique, counts = np.unique(samples_binomial, return_counts=True)
    ax2.bar(unique, counts/n_samples, alpha=0.7, color='green', 
            edgecolor='black', label='样本频率')
    
    # 理论概率质量函数
    x_range_binom = np.arange(0, n+1)
    pmf_binomial = stats.binom.pmf(x_range_binom, n, p)
    ax2.plot(x_range_binom, pmf_binomial, 'ro-', linewidth=2, 
             markersize=8, label='理论PMF')
    
    ax2.set_title(f'二项分布 B({n}, {p})')
    ax2.set_xlabel('成功次数 k')
    ax2.set_ylabel('概率')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加期望值标记
    expected = n * p
    ax2.axvline(x=expected, color='red', linestyle='--', alpha=0.5)
    ax2.text(expected, ax2.get_ylim()[1]*0.9, f'E[X]={expected:.1f}',
             ha='center', bbox=dict(boxstyle='round', facecolor='white'))
    
    # 3. 多维高斯分布
    ax3 = axes[1, 0]
    
    # 生成2D高斯分布
    mean_2d = [0, 0]
    # 协方差矩阵
    cov_matrix = [[1, 0.5], 
                   [0.5, 1]]
    samples_2d = np.random.multivariate_normal(mean_2d, cov_matrix, 500)
    
    # 绘制散点图
    ax3.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.5, s=20)
    
    # 绘制等高线
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean_2d, cov_matrix)
    contours = ax3.contour(X, Y, rv.pdf(pos), colors='red', levels=5)
    ax3.clabel(contours, inline=True, fontsize=8)
    
    ax3.set_title('二维高斯分布（相关性=0.5）')
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. 贝塔分布（Beta分布）
    ax4 = axes[1, 1]
    
    # 不同参数的贝塔分布
    alpha_beta_pairs = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
    x_range_beta = np.linspace(0, 1, 100)
    
    for alpha, beta in alpha_beta_pairs:
        pdf_beta = stats.beta.pdf(x_range_beta, alpha, beta)
        ax4.plot(x_range_beta, pdf_beta, linewidth=2, 
                label=f'α={alpha}, β={beta}')
    
    ax4.set_title('贝塔分布 Beta(α, β)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('概率密度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    
    # 添加说明
    ax4.text(0.5, -0.15, '贝塔分布常用作二项分布参数p的先验分布',
             ha='center', transform=ax4.transAxes, fontsize=10,
             style='italic', color='gray')
    
    plt.suptitle('机器学习中的重要概率分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def compute_expectations_and_covariances() -> None:
    """
    计算和展示期望、方差、协方差的概念
    
    这些是概率论中的核心概念：
    
    期望 E[X]：随机变量的平均值
    E[X] = Σ x × P(x)  (离散)
    E[X] = ∫ x × p(x) dx  (连续)
    
    方差 Var[X]：度量随机变量的离散程度
    Var[X] = E[(X - E[X])²] = E[X²] - E[X]²
    
    协方差 Cov[X,Y]：度量两个随机变量的线性相关性
    Cov[X,Y] = E[(X - E[X])(Y - E[Y])]
    
    相关系数：标准化的协方差
    ρ(X,Y) = Cov[X,Y] / (σ_X × σ_Y)
    """
    print("\n期望、方差和协方差的计算：")
    print("=" * 50)
    
    # 生成两个相关的随机变量
    n_samples = 1000
    
    # X ~ N(0, 1)
    X = np.random.normal(0, 1, n_samples)
    
    # Y = 0.7X + 0.3Z, 其中 Z ~ N(0, 1)
    # 这样Y和X有正相关性
    Z = np.random.normal(0, 1, n_samples)
    Y = 0.7 * X + 0.3 * Z
    
    # 计算统计量
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    var_X = np.var(X)
    var_Y = np.var(Y)
    
    # 协方差和相关系数
    cov_XY = np.cov(X, Y)[0, 1]
    corr_XY = np.corrcoef(X, Y)[0, 1]
    
    print(f"X的统计量：")
    print(f"  期望 E[X] = {mean_X:.4f} (理论值: 0)")
    print(f"  方差 Var[X] = {var_X:.4f} (理论值: 1)")
    print()
    print(f"Y的统计量：")
    print(f"  期望 E[Y] = {mean_Y:.4f}")
    print(f"  方差 Var[Y] = {var_Y:.4f}")
    print()
    print(f"X和Y的关系：")
    print(f"  协方差 Cov[X,Y] = {cov_XY:.4f}")
    print(f"  相关系数 ρ(X,Y) = {corr_XY:.4f}")
    print()
    print("解释：")
    print(f"- 相关系数 {corr_XY:.2f} 表明X和Y有较强的正线性关系")
    print("- 这是因为Y的构造中包含了0.7倍的X")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 散点图
    ax1 = axes[0]
    ax1.scatter(X, Y, alpha=0.5, s=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'X与Y的散点图 (ρ={corr_XY:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # 添加回归线
    coeffs = np.polyfit(X, Y, 1)
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
    ax1.legend()
    
    # X的分布
    ax2 = axes[1]
    ax2.hist(X, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=mean_X, color='red', linestyle='--', linewidth=2, label=f'均值={mean_X:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('概率密度')
    ax2.set_title(f'X的分布 (σ²={var_X:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Y的分布
    ax3 = axes[2]
    ax3.hist(Y, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=mean_Y, color='red', linestyle='--', linewidth=2, label=f'均值={mean_Y:.3f}')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('概率密度')
    ax3.set_title(f'Y的分布 (σ²={var_Y:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('期望、方差和协方差的可视化', fontsize=14)
    plt.tight_layout()
    plt.show()