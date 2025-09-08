"""
Chapter 2: Probability Distributions (概率分布)
===============================================

本章深入探讨机器学习中的核心概率分布，为后续章节奠定数学基础。

主要内容：
1. 二元变量 (2.1)
   - 伯努利分布
   - 二项分布
   - 贝塔分布（共轭先验）

2. 多项式变量 (2.2)
   - 多项分布
   - 狄利克雷分布（共轭先验）

3. 高斯分布 (2.3)
   - 一维和多维高斯
   - 最大似然估计
   - 贝叶斯推断

4. 指数族 (2.4)
   - 统一框架
   - 充分统计量
   - 共轭先验

5. 非参数方法 (2.5)
   - 直方图
   - 核密度估计
   - K近邻

这些分布是理解机器学习算法的基础，
特别是在贝叶斯方法、混合模型、神经网络等领域。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .binary_variables import (
    BernoulliDistribution,
    BinomialDistribution,
    BetaDistribution,
    demonstrate_bernoulli_distribution,
    demonstrate_beta_distribution,
    demonstrate_bayesian_inference
)

from .multinomial_variables import (
    MultinomialDistribution,
    DirichletDistribution,
    visualize_dirichlet_on_simplex,
    demonstrate_dirichlet_bayesian_update
)

from .gaussian_distribution import (
    UnivariateGaussian,
    MultivariateGaussian,
    demonstrate_univariate_gaussian,
    demonstrate_multivariate_gaussian,
    demonstrate_mle_convergence
)


def run_chapter02(cfg: DictConfig) -> None:
    """
    运行第2章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第2章：概率分布 (Probability Distributions)")
    print("="*80)
    
    # 2.1 二元变量
    print("\n" + "-"*60)
    print("2.1 二元变量 (Binary Variables)")
    print("-"*60)
    
    # 伯努利分布演示
    demonstrate_bernoulli_distribution(
        mu_values=cfg.chapter.binary_variables.bernoulli.mu_values,
        n_samples=cfg.chapter.binary_variables.bernoulli.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # 贝塔分布演示
    demonstrate_beta_distribution(
        param_pairs=cfg.chapter.binary_variables.beta.param_pairs,
        n_points=cfg.chapter.binary_variables.beta.n_points,
        show_plot=cfg.visualization.show_plots
    )
    
    # 贝叶斯推断演示
    demonstrate_bayesian_inference(
        true_mu=cfg.chapter.binary_variables.bayesian_inference.true_mu,
        prior_a=cfg.chapter.binary_variables.bayesian_inference.prior_a,
        prior_b=cfg.chapter.binary_variables.bayesian_inference.prior_b,
        data_sizes=cfg.chapter.binary_variables.bayesian_inference.data_sizes,
        show_plot=cfg.visualization.show_plots
    )
    
    # 2.2 多项式变量
    print("\n" + "-"*60)
    print("2.2 多项式变量 (Multinomial Variables)")
    print("-"*60)
    
    # 狄利克雷分布可视化
    visualize_dirichlet_on_simplex(
        alpha_values=[np.array(a) for a in cfg.chapter.multinomial_variables.dirichlet.alpha_values],
        n_samples=cfg.chapter.multinomial_variables.dirichlet.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # 狄利克雷贝叶斯更新
    demonstrate_dirichlet_bayesian_update(
        true_probs=np.array(cfg.chapter.multinomial_variables.multinomial.probabilities),
        prior_alpha=np.array([2, 2, 2]),  # 使用均匀先验
        n_observations=[10, 50, 100, 500],
        show_plot=cfg.visualization.show_plots
    )
    
    # 2.3 高斯分布
    print("\n" + "-"*60)
    print("2.3 高斯分布 (The Gaussian Distribution)")
    print("-"*60)
    
    # 一维高斯分布
    demonstrate_univariate_gaussian(
        mean_values=cfg.chapter.gaussian_distribution.univariate.mean_values,
        variance_values=cfg.chapter.gaussian_distribution.univariate.variance_values,
        n_samples=cfg.chapter.gaussian_distribution.univariate.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # 多维高斯分布
    demonstrate_multivariate_gaussian(
        dimension=cfg.chapter.gaussian_distribution.multivariate.dimension,
        covariance_types=cfg.chapter.gaussian_distribution.multivariate.covariance_types,
        n_samples=cfg.chapter.gaussian_distribution.multivariate.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # MLE收敛性
    demonstrate_mle_convergence(
        true_mean=cfg.chapter.gaussian_distribution.bayesian.true_mean,
        true_variance=cfg.chapter.gaussian_distribution.bayesian.true_variance,
        sample_sizes=cfg.chapter.gaussian_distribution.mle.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第2章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 共轭先验使贝叶斯推断变得简单")
    print("2. 高斯分布在机器学习中无处不在")
    print("3. 理解这些分布对于深入学习至关重要")