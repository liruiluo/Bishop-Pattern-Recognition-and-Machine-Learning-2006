"""
1.5 决策论 (Decision Theory)
=============================

决策论告诉我们如何基于概率做出最优决策。

核心概念：
1. 损失函数：量化错误决策的代价
2. 风险：损失的期望值
3. 贝叶斯决策：最小化期望风险

这些概念在分类和回归问题中都很重要。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.datasets import make_classification, make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def classification_boundaries_demo(
    n_classes: int = 3,
    n_samples_per_class: int = 100,
    show_plot: bool = True
) -> None:
    """
    演示分类决策边界
    
    展示不同决策规则下的分类边界：
    1. 最大后验概率 (MAP)
    2. 最小错分类率
    3. 拒绝选项
    
    Args:
        n_classes: 类别数
        n_samples_per_class: 每个类的样本数
        show_plot: 是否显示图形
    """
    print("\n分类决策边界演示：")
    print("=" * 40)
    
    # 生成分类数据
    np.random.seed(42)
    X, y = make_blobs(n_samples=n_classes * n_samples_per_class,
                      n_features=2,
                      centers=n_classes,
                      cluster_std=0.5,
                      random_state=42)
    
    # 训练线性判别分析模型
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # 计算后验概率
    proba = lda.predict_proba(X)
    predictions = lda.predict(X)
    
    # 计算分类精度
    accuracy = np.mean(predictions == y)
    print(f"分类精度: {accuracy:.2%}")
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y, predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    print("\n混淆矩阵:")
    print(confusion_matrix)
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1：原始数据和决策边界
        ax1 = axes[0]
        
        # 创建网格用于绘制决策边界
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black',
                   cmap=plt.cm.coolwarm, s=50)
        ax1.set_xlabel('特征 1')
        ax1.set_ylabel('特征 2')
        ax1.set_title('分类决策边界')
        
        # 子图2：后验概率分布
        ax2 = axes[1]
        
        # 只显示第一个类的后验概率
        Z_proba = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        Z_proba = Z_proba.reshape(xx.shape)
        
        contour = ax2.contourf(xx, yy, Z_proba, levels=20, cmap=plt.cm.viridis)
        plt.colorbar(contour, ax=ax2)
        ax2.scatter(X[y==0, 0], X[y==0, 1], c='red', edgecolors='black',
                   s=50, label='类 0')
        ax2.scatter(X[y!=0, 0], X[y!=0, 1], c='blue', edgecolors='black',
                   s=20, alpha=0.3, label='其他类')
        ax2.set_xlabel('特征 1')
        ax2.set_ylabel('特征 2')
        ax2.set_title('类0的后验概率 P(C=0|x)')
        ax2.legend()
        
        # 子图3：拒绝选项
        ax3 = axes[2]
        
        # 计算最大后验概率
        max_proba = np.max(proba, axis=1)
        
        # 设置拒绝阈值
        reject_threshold = 0.7
        rejected = max_proba < reject_threshold
        
        ax3.scatter(X[~rejected, 0], X[~rejected, 1], 
                   c=predictions[~rejected], cmap=plt.cm.coolwarm,
                   edgecolors='black', s=50, label='分类')
        ax3.scatter(X[rejected, 0], X[rejected, 1],
                   c='gray', marker='x', s=100, label='拒绝')
        ax3.set_xlabel('特征 1')
        ax3.set_ylabel('特征 2')
        ax3.set_title(f'拒绝选项 (阈值={reject_threshold})')
        ax3.legend()
        
        print(f"\n拒绝率: {np.mean(rejected):.2%}")
        
        plt.suptitle('决策论在分类中的应用', fontsize=14)
        plt.tight_layout()
        plt.show()


def loss_functions_comparison(
    loss_types: List[str] = ['squared', 'absolute', 'huber'],
    show_plot: bool = True
) -> None:
    """
    比较不同损失函数
    
    展示回归问题中常用的损失函数：
    1. 平方损失：对大误差敏感
    2. 绝对值损失：对异常值鲁棒
    3. Huber损失：结合两者优点
    
    Args:
        loss_types: 要比较的损失函数类型
        show_plot: 是否显示图形
    """
    print("\n损失函数比较：")
    print("=" * 40)
    
    # 定义损失函数
    def squared_loss(error):
        return error ** 2
    
    def absolute_loss(error):
        return np.abs(error)
    
    def huber_loss(error, delta=1.0):
        return np.where(np.abs(error) <= delta,
                       0.5 * error ** 2,
                       delta * (np.abs(error) - 0.5 * delta))
    
    # 误差范围
    errors = np.linspace(-3, 3, 100)
    
    print("损失函数特点：")
    print("-" * 40)
    print("平方损失: L(e) = e²")
    print("  - 优点: 可微，便于优化")
    print("  - 缺点: 对异常值敏感")
    print()
    print("绝对值损失: L(e) = |e|")
    print("  - 优点: 对异常值鲁棒")
    print("  - 缺点: 在e=0处不可微")
    print()
    print("Huber损失: L(e) = {0.5e² if |e|≤δ, δ(|e|-0.5δ) if |e|>δ}")
    print("  - 优点: 结合两者优点")
    print("  - 缺点: 需要调节参数δ")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 子图1：损失函数对比
        ax1 = axes[0]
        
        if 'squared' in loss_types:
            ax1.plot(errors, squared_loss(errors), 'b-', 
                    linewidth=2, label='平方损失')
        if 'absolute' in loss_types:
            ax1.plot(errors, absolute_loss(errors), 'r-',
                    linewidth=2, label='绝对值损失')
        if 'huber' in loss_types:
            ax1.plot(errors, huber_loss(errors), 'g-',
                    linewidth=2, label='Huber损失')
        
        ax1.set_xlabel('误差 e')
        ax1.set_ylabel('损失 L(e)')
        ax1.set_title('不同损失函数的比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：导数对比
        ax2 = axes[1]
        
        # 计算导数（数值近似）
        de = errors[1] - errors[0]
        
        if 'squared' in loss_types:
            grad_squared = 2 * errors
            ax2.plot(errors, grad_squared, 'b-',
                    linewidth=2, label='平方损失梯度')
        if 'absolute' in loss_types:
            grad_absolute = np.sign(errors)
            ax2.plot(errors, grad_absolute, 'r-',
                    linewidth=2, label='绝对值损失梯度')
        if 'huber' in loss_types:
            grad_huber = np.where(np.abs(errors) <= 1.0,
                                 errors,
                                 np.sign(errors))
            ax2.plot(errors, grad_huber, 'g-',
                    linewidth=2, label='Huber损失梯度')
        
        ax2.set_xlabel('误差 e')
        ax2.set_ylabel('梯度 dL/de')
        ax2.set_title('损失函数的梯度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('决策论：损失函数的选择', fontsize=14)
        plt.tight_layout()
        plt.show()