"""
Chapter 4: Linear Models for Classification (分类的线性模型)
============================================================

本章介绍分类任务的线性方法，从判别函数到概率模型。

主要内容：
1. 判别函数 (4.1)
   - 二分类
   - 多分类
   - 线性判别分析

2. 概率生成模型 (4.2)
   - 高斯判别分析（LDA/QDA）
   - 朴素贝叶斯
   - 指数族连接

3. 概率判别模型 (4.3)
   - 逻辑回归
   - Softmax回归
   - 迭代优化方法

核心思想：
线性模型虽然简单，但是许多复杂模型的基础。
通过理解线性分类器，可以更好地理解神经网络、SVM等。

判别模型 vs 生成模型：
- 生成模型：建模p(x|y)和p(y)，可以生成新样本
- 判别模型：直接建模p(y|x)，通常性能更好

本章为后续的神经网络、核方法等奠定基础。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .discriminant_functions import (
    LinearDiscriminant,
    MultiClassDiscriminant,
    generate_linear_separable_data,
    visualize_decision_boundary,
    demonstrate_two_class_discriminant,
    demonstrate_multi_class_discriminant,
    compare_multiclass_strategies
)

from .probabilistic_generative import (
    GaussianGenerativeClassifier,
    NaiveBayesClassifier,
    demonstrate_gaussian_generative,
    demonstrate_naive_bayes
)

from .probabilistic_discriminative import (
    LogisticRegression,
    SoftmaxRegression,
    demonstrate_logistic_regression,
    demonstrate_softmax_regression
)


def run_chapter04(cfg: DictConfig) -> None:
    """
    运行第4章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第4章：分类的线性模型 (Linear Models for Classification)")
    print("="*80)
    
    # 4.1 判别函数
    print("\n" + "-"*60)
    print("4.1 判别函数 (Discriminant Functions)")
    print("-"*60)
    
    # 二分类判别函数
    demonstrate_two_class_discriminant(
        n_samples=cfg.chapter.data_generation.linearly_separable.n_samples,
        show_plot=cfg.visualization.show_plots
    )
    
    # 多分类判别函数
    demonstrate_multi_class_discriminant(
        n_classes=cfg.chapter.discriminant_functions.multi_class.n_classes,
        n_samples_per_class=cfg.chapter.data_generation.multi_classification.n_samples_per_class,
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较多分类策略
    compare_multiclass_strategies(
        n_classes=4,
        n_samples_per_class=30,
        show_plot=cfg.visualization.show_plots
    )
    
    # 4.2 概率生成模型
    print("\n" + "-"*60)
    print("4.2 概率生成模型 (Probabilistic Generative Models)")
    print("-"*60)
    
    # 高斯生成模型
    demonstrate_gaussian_generative(
        n_samples=300,
        show_plot=cfg.visualization.show_plots
    )
    
    # 朴素贝叶斯
    demonstrate_naive_bayes(
        n_samples=500,
        show_plot=cfg.visualization.show_plots
    )
    
    # 4.3 概率判别模型
    print("\n" + "-"*60)
    print("4.3 概率判别模型 (Probabilistic Discriminative Models)")
    print("-"*60)
    
    # 逻辑回归
    demonstrate_logistic_regression(
        n_samples=200,
        show_plot=cfg.visualization.show_plots
    )
    
    # Softmax回归
    demonstrate_softmax_regression(
        n_classes=cfg.chapter.discriminant_functions.multi_class.n_classes,
        n_samples=300,
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较生成模型和判别模型
    print("\n" + "-"*60)
    print("生成模型 vs 判别模型")
    print("-"*60)
    
    # 生成数据
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=2,
                              n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, class_sep=1.5,
                              random_state=42)
    
    # 分割数据
    n_train = 140
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 训练生成模型（高斯）
    gen_model = GaussianGenerativeClassifier(shared_covariance=False)
    gen_model.fit(X_train, y_train)
    gen_acc = gen_model.score(X_test, y_test)
    
    # 训练判别模型（逻辑回归）
    disc_model = LogisticRegression(method='irls')
    disc_model.fit(X_train, y_train)
    disc_acc = disc_model.score(X_test, y_test)
    
    print("\n测试集性能：")
    print(f"生成模型（高斯）: {gen_acc:.2%}")
    print(f"判别模型（逻辑回归）: {disc_acc:.2%}")
    
    # 小样本情况
    print("\n小样本性能（n=20）：")
    X_small = X_train[:20]
    y_small = y_train[:20]
    
    gen_model_small = GaussianGenerativeClassifier(shared_covariance=True)
    gen_model_small.fit(X_small, y_small)
    gen_acc_small = gen_model_small.score(X_test, y_test)
    
    disc_model_small = LogisticRegression(method='irls')
    disc_model_small.fit(X_small, y_small)
    disc_acc_small = disc_model_small.score(X_test, y_test)
    
    print(f"生成模型: {gen_acc_small:.2%}")
    print(f"判别模型: {disc_acc_small:.2%}")
    
    if cfg.visualization.show_plots:
        # 可视化比较
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = [(gen_model, "生成模型（高斯）"),
                 (disc_model, "判别模型（逻辑回归）")]
        
        for idx, (model, title) in enumerate(models):
            ax = axes[idx]
            
            # 创建网格
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            
            # 预测概率
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_grid)
                if proba.shape[1] > 1:
                    proba = proba[:, 1]
                else:
                    proba = proba[:, 0]
            else:
                proba = model.predict(X_grid)
            
            proba = proba.reshape(xx.shape)
            
            # 绘制
            contour = ax.contourf(xx, yy, proba, levels=20,
                                 cmap='RdBu_r', alpha=0.8)
            ax.contour(xx, yy, proba, levels=[0.5],
                      colors='black', linewidths=2)
            
            # 训练数据
            ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                      c='blue', s=30, alpha=0.5, edgecolor='black')
            ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                      c='red', s=30, alpha=0.5, edgecolor='black')
            
            # 测试数据
            ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                      c='blue', s=100, marker='^', edgecolor='black')
            ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                      c='red', s=100, marker='^', edgecolor='black')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(f'{title}\n测试准确率: {model.score(X_test, y_test):.2%}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('生成模型 vs 判别模型', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*80)
    print("第4章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 判别函数直接划分输入空间")
    print("2. 生成模型建模类条件密度p(x|y)")
    print("3. 判别模型直接建模后验p(y|x)")
    print("4. 逻辑回归是线性模型但输出概率")
    print("5. 优化方法：梯度下降、牛顿法、IRLS")
    print("6. 判别模型通常性能更好，生成模型更灵活")