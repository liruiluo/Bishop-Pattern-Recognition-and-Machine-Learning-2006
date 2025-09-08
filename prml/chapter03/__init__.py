"""
Chapter 3: Linear Models for Regression (线性回归模型)
======================================================

本章介绍线性回归的系统理论，包括频率派和贝叶斯方法。

主要内容：
1. 线性基函数模型 (3.1)
   - 多项式基函数
   - 高斯基函数
   - Sigmoid基函数
   - 傅里叶基函数

2. 偏差-方差分解 (3.2)
   - 期望损失的分解
   - 模型复杂度的影响
   - 正则化的作用

3. 贝叶斯线性回归 (3.3)
   - 参数的后验分布
   - 预测分布
   - 顺序学习
   - 证据近似

核心思想：
线性模型虽然简单，但通过基函数变换可以建模复杂的非线性关系。
贝叶斯方法提供了参数不确定性的量化，自动实现正则化。

本章为后续的分类、神经网络、核方法等奠定基础。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .linear_basis_models import (
    BasisFunction,
    PolynomialBasis,
    GaussianBasis,
    SigmoidBasis,
    FourierBasis,
    LinearRegression,
    generate_synthetic_data,
    compare_basis_functions,
    demonstrate_regularization_effect
)

from .bias_variance import (
    bias_variance_experiment,
    bias_variance_vs_complexity,
    demonstrate_regularization_bias_variance
)

from .bayesian_linear_regression import (
    BayesianLinearRegression,
    demonstrate_bayesian_regression,
    sequential_bayesian_learning,
    compare_bayesian_vs_mle,
    demonstrate_evidence_approximation,
    demonstrate_model_comparison
)


def run_chapter03(cfg: DictConfig) -> None:
    """
    运行第3章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第3章：线性回归模型 (Linear Models for Regression)")
    print("="*80)
    
    # 生成示例数据
    print("\n生成示例数据...")
    X_train, y_train = generate_synthetic_data(
        n_samples=cfg.chapter.linear_basis_models.data_generation.n_train,
        noise_std=cfg.chapter.linear_basis_models.data_generation.noise_std,
        x_min=cfg.chapter.linear_basis_models.data_generation.x_min,
        x_max=cfg.chapter.linear_basis_models.data_generation.x_max,
        seed=42
    )
    
    X_test, y_test = generate_synthetic_data(
        n_samples=cfg.chapter.linear_basis_models.data_generation.n_test,
        noise_std=cfg.chapter.linear_basis_models.data_generation.noise_std,
        x_min=cfg.chapter.linear_basis_models.data_generation.x_min,
        x_max=cfg.chapter.linear_basis_models.data_generation.x_max,
        seed=123
    )
    
    # 3.1 线性基函数模型
    print("\n" + "-"*60)
    print("3.1 线性基函数模型 (Linear Basis Function Models)")
    print("-"*60)
    
    # 比较不同基函数
    compare_basis_functions(
        X_train, y_train, X_test, y_test,
        n_basis=cfg.chapter.linear_basis_models.polynomial.order,
        regularization=0.0,
        show_plot=cfg.visualization.show_plots
    )
    
    # 演示正则化效果
    if cfg.chapter.linear_basis_models.regularization.enabled:
        demonstrate_regularization_effect(
            X_train, y_train, X_test, y_test,
            basis_type='polynomial',
            n_basis=cfg.chapter.linear_basis_models.polynomial.order,
            lambda_values=cfg.chapter.linear_basis_models.regularization.lambda_values,
            show_plot=cfg.visualization.show_plots
        )
    
    # 3.2 偏差-方差分解
    print("\n" + "-"*60)
    print("3.2 偏差-方差分解 (The Bias-Variance Decomposition)")
    print("-"*60)
    
    # 偏差-方差实验
    bias_variance_experiment(
        n_datasets=cfg.chapter.bias_variance.n_datasets,
        n_samples=cfg.chapter.bias_variance.n_samples,
        polynomial_orders=cfg.chapter.bias_variance.polynomial_orders,
        noise_std=cfg.chapter.bias_variance.noise_std,
        show_plot=cfg.visualization.show_plots
    )
    
    # 偏差-方差随复杂度变化
    bias_variance_vs_complexity(
        n_datasets=cfg.chapter.bias_variance.n_datasets,
        n_samples=cfg.chapter.bias_variance.n_samples,
        max_order=9,
        noise_std=cfg.chapter.bias_variance.noise_std,
        show_plot=cfg.visualization.show_plots
    )
    
    # 正则化对偏差-方差的影响
    demonstrate_regularization_bias_variance(
        n_datasets=cfg.chapter.bias_variance.n_datasets,
        n_samples=cfg.chapter.bias_variance.n_samples,
        polynomial_order=9,
        noise_std=cfg.chapter.bias_variance.noise_std,
        show_plot=cfg.visualization.show_plots
    )
    
    # 3.3 贝叶斯线性回归
    print("\n" + "-"*60)
    print("3.3 贝叶斯线性回归 (Bayesian Linear Regression)")
    print("-"*60)
    
    # 基本贝叶斯回归演示
    demonstrate_bayesian_regression(
        n_train=cfg.chapter.linear_basis_models.data_generation.n_train,
        basis_type=cfg.chapter.bayesian_linear_regression.basis_type,
        n_basis=cfg.chapter.bayesian_linear_regression.n_basis,
        alpha=cfg.chapter.bayesian_linear_regression.prior.alpha,
        beta=cfg.chapter.bayesian_linear_regression.prior.beta,
        show_plot=cfg.visualization.show_plots
    )
    
    # 顺序贝叶斯学习
    if cfg.chapter.bayesian_linear_regression.sequential_learning.enabled:
        sequential_bayesian_learning(
            basis_type=cfg.chapter.bayesian_linear_regression.basis_type,
            n_basis=cfg.chapter.bayesian_linear_regression.n_basis,
            alpha=cfg.chapter.bayesian_linear_regression.prior.alpha,
            beta=cfg.chapter.bayesian_linear_regression.prior.beta,
            data_sequence=cfg.chapter.bayesian_linear_regression.sequential_learning.data_points,
            show_plot=cfg.visualization.show_plots
        )
    
    # 贝叶斯 vs MLE
    compare_bayesian_vs_mle(
        n_train_list=[5, 20, 100],
        n_basis=cfg.chapter.bayesian_linear_regression.n_basis,
        alpha=cfg.chapter.bayesian_linear_regression.prior.alpha,
        beta=cfg.chapter.bayesian_linear_regression.prior.beta,
        show_plot=cfg.visualization.show_plots
    )
    
    # 3.4 贝叶斯模型比较
    print("\n" + "-"*60)
    print("3.4 贝叶斯模型比较 (Bayesian Model Comparison)")
    print("-"*60)
    
    # 模型比较
    demonstrate_model_comparison(
        X_train, y_train,
        model_orders=cfg.chapter.model_comparison.model_orders,
        alpha=cfg.chapter.bayesian_linear_regression.prior.alpha,
        beta=cfg.chapter.bayesian_linear_regression.prior.beta,
        show_plot=cfg.visualization.show_plots
    )
    
    # 3.5 证据近似
    print("\n" + "-"*60)
    print("3.5 证据近似 (The Evidence Approximation)")
    print("-"*60)
    
    if cfg.chapter.evidence_approximation.show_convergence:
        # 证据近似（经验贝叶斯）
        result = demonstrate_evidence_approximation(
            X_train, y_train,
            n_basis=cfg.chapter.bayesian_linear_regression.n_basis,
            show_plot=cfg.visualization.show_plots
        )
        
        print(f"\n优化后的超参数：")
        print(f"  α = {result['alpha']:.4f} (先验精度)")
        print(f"  β = {result['beta']:.4f} (噪声精度)")
        print(f"  有效参数数 γ = {result['gamma']:.2f}")
    
    # 3.6 固定基函数的局限性
    print("\n" + "-"*60)
    print("3.6 固定基函数的局限性 (Limitations of Fixed Basis Functions)")
    print("-"*60)
    
    print("\n维度诅咒演示：")
    print("-" * 40)
    
    # 计算不同维度下需要的基函数数量
    dimensions = cfg.chapter.limitations.curse_of_dimensionality.dimensions
    n_basis_per_dim = cfg.chapter.limitations.curse_of_dimensionality.n_basis_per_dim
    
    print(f"每个维度使用{n_basis_per_dim}个基函数：")
    for d in dimensions:
        total_basis = n_basis_per_dim ** d
        print(f"  {d}维: {total_basis:,} 个基函数")
    
    print("\n局部 vs 全局基函数：")
    print("-" * 40)
    print("局部基函数（如高斯）：")
    print("  - 优点：局部支撑，改变一处不影响远处")
    print("  - 缺点：需要很多基函数覆盖输入空间")
    print("\n全局基函数（如多项式）：")
    print("  - 优点：少量基函数就能表示复杂函数")
    print("  - 缺点：改变一处影响全局，易振荡")
    
    print("\n" + "="*80)
    print("第3章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 线性模型通过基函数变换可以建模非线性关系")
    print("2. 偏差-方差分解揭示了模型复杂度的权衡")
    print("3. 贝叶斯方法提供不确定性量化和自动正则化")
    print("4. 边际似然实现自动模型选择")
    print("5. 证据近似自动确定超参数")
    print("6. 固定基函数的局限导致了核方法和神经网络的发展")