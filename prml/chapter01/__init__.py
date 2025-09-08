"""
Chapter 1: Introduction (引言)
================================

这一章是整本书的引言，通过一个简单但富有启发性的例子——多项式曲线拟合，
介绍了机器学习中的核心概念。这些概念将贯穿整本书的学习。

本章的主要内容：
1. 多项式曲线拟合 (1.1节)
   - 展示了过拟合和欠拟合问题
   - 引入了正则化的概念
   - 说明了模型复杂度与数据量的关系

2. 概率论回顾 (1.2节)
   - 概率的基本定义和规则
   - 贝叶斯定理
   - 概率密度函数
   - 期望和协方差

3. 模型选择 (1.3节)
   - 训练集、验证集和测试集的概念
   - 交叉验证方法

4. 维度诅咒 (1.4节)
   - 高维空间的特殊性质
   - 为什么高维问题很困难

5. 决策论 (1.5节)
   - 最小化错分类率
   - 最小化期望损失
   - 拒绝选项

6. 信息论 (1.6节)
   - 信息熵
   - 相对熵和互信息

通过这一章的学习，你将建立起机器学习的基本概念框架，
为后续章节的深入学习打下坚实的基础。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings

# 导入各个小节的实现
from .polynomial_fitting import (
    PolynomialCurveFitting,
    generate_sinusoidal_data,
    plot_polynomial_fits,
    demonstrate_overfitting,
    demonstrate_regularization
)

from .probability_theory import (
    demonstrate_bayes_theorem,
    visualize_distributions,
    compute_expectations_and_covariances
)

from .model_selection import (
    cross_validation_demo,
    train_validation_test_split
)

from .curse_of_dimensionality import (
    demonstrate_curse_of_dimensionality,
    volume_of_hypersphere
)

from .decision_theory import (
    classification_boundaries_demo,
    loss_functions_comparison
)

from .information_theory import (
    entropy_demo,
    mutual_information_demo
)


def run_chapter01(cfg: DictConfig) -> None:
    """
    运行第1章的所有演示代码
    
    这个函数是第1章的主入口，它会按照书中的顺序
    运行各个小节的演示代码，展示机器学习的基本概念。
    
    Args:
        cfg: Hydra配置对象，包含所有运行参数
    """
    
    print("\n" + "="*80)
    print("第1章：引言 (Introduction)")
    print("="*80)
    
    # 1.1 多项式曲线拟合
    print("\n" + "-"*60)
    print("1.1 多项式曲线拟合 (Polynomial Curve Fitting)")
    print("-"*60)
    
    # 生成训练数据
    # 这里我们使用sin(2πx)函数加上高斯噪声来生成数据
    # 这是书中的经典例子，展示了真实函数与观测数据的关系
    X_train, y_train, X_test, y_test = generate_sinusoidal_data(
        n_train=cfg.chapter.polynomial_fitting.data_generation.n_train,
        n_test=cfg.chapter.polynomial_fitting.data_generation.n_test,
        noise_std=cfg.chapter.polynomial_fitting.data_generation.noise_std,
        seed=cfg.general.seed
    )
    
    print(f"生成了 {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本")
    print(f"数据维度: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # 演示不同阶数多项式的拟合效果
    # 这个演示展示了模型复杂度如何影响拟合效果
    print("\n展示不同阶数多项式的拟合效果...")
    demonstrate_overfitting(
        X_train, y_train, X_test, y_test,
        orders=cfg.chapter.polynomial_fitting.polynomial_orders,
        show_plot=cfg.visualization.show_plots
    )
    
    # 演示正则化的效果
    # 正则化是控制过拟合的重要技术
    if cfg.chapter.polynomial_fitting.regularization.enabled:
        print("\n展示正则化对高阶多项式的影响...")
        demonstrate_regularization(
            X_train, y_train, X_test, y_test,
            order=9,  # 使用9阶多项式
            lambda_values=cfg.chapter.polynomial_fitting.regularization.lambda_values,
            show_plot=cfg.visualization.show_plots
        )
    
    # 1.2 概率论回顾
    print("\n" + "-"*60)
    print("1.2 概率论回顾 (Probability Theory)")
    print("-"*60)
    
    # 贝叶斯定理演示
    # 这是机器学习中最重要的定理之一
    print("\n贝叶斯定理演示：")
    demonstrate_bayes_theorem(
        prior=cfg.chapter.probability_theory.bayes_example.prior_prob,
        likelihood=cfg.chapter.probability_theory.bayes_example.likelihood,
        false_positive=cfg.chapter.probability_theory.bayes_example.false_positive_rate
    )
    
    # 概率分布可视化
    print("\n可视化常见概率分布...")
    visualize_distributions(
        cfg.chapter.probability_theory.distributions,
        show_plot=cfg.visualization.show_plots
    )
    
    # 1.3 模型选择
    print("\n" + "-"*60)
    print("1.3 模型选择 (Model Selection)")
    print("-"*60)
    
    # 交叉验证演示
    # 这是选择模型复杂度的重要方法
    print("\n使用交叉验证选择最优多项式阶数...")
    best_order = cross_validation_demo(
        X_train, y_train,
        orders=cfg.chapter.model_selection.cross_validation.polynomial_orders,
        n_folds=cfg.chapter.model_selection.cross_validation.n_folds,
        show_plot=cfg.visualization.show_plots
    )
    print(f"交叉验证选择的最优多项式阶数: {best_order}")
    
    # 1.4 维度诅咒
    print("\n" + "-"*60)
    print("1.4 维度诅咒 (The Curse of Dimensionality)")
    print("-"*60)
    
    # 演示高维空间的特殊性质
    print("\n演示维度诅咒...")
    demonstrate_curse_of_dimensionality(
        dimensions=cfg.chapter.curse_of_dimensionality.dimensions,
        n_samples=cfg.chapter.curse_of_dimensionality.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # 1.5 决策论
    print("\n" + "-"*60)
    print("1.5 决策论 (Decision Theory)")
    print("-"*60)
    
    # 分类决策边界演示
    print("\n演示分类决策边界...")
    classification_boundaries_demo(
        n_classes=cfg.chapter.decision_theory.classification.n_classes,
        n_samples_per_class=cfg.chapter.decision_theory.classification.n_samples_per_class,
        show_plot=cfg.visualization.show_plots
    )
    
    # 损失函数比较
    print("\n比较不同的损失函数...")
    loss_functions_comparison(
        loss_types=cfg.chapter.decision_theory.loss_functions,
        show_plot=cfg.visualization.show_plots
    )
    
    # 1.6 信息论
    print("\n" + "-"*60)
    print("1.6 信息论 (Information Theory)")
    print("-"*60)
    
    # 熵的演示
    print("\n计算不同分布的熵...")
    entropy_demo(
        distributions=cfg.chapter.information_theory.entropy_demo.distributions,
        n_bins=cfg.chapter.information_theory.entropy_demo.n_bins,
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第1章演示完成！")
    print("="*80)