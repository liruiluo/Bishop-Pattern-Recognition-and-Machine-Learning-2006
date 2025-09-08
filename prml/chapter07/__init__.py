"""
Chapter 7: Sparse Kernel Machines (稀疏核机器)
==============================================

本章介绍稀疏核机器，主要包括支持向量机(SVM)和相关向量机(RVM)。

主要内容：
1. 最大间隔分类器 (7.1)
   - 线性可分SVM
   - 软间隔SVM
   - 核SVM
   - SMO算法

2. 相关向量机 (7.2)
   - RVM回归
   - RVM分类
   - 自动相关性确定

核心概念：
稀疏性是这些方法的关键特征。
只有少数训练样本（支持向量/相关向量）对最终模型有贡献。

SVM vs RVM：
- SVM：基于最大间隔原理，凸优化，需要调参
- RVM：贝叶斯方法，自动确定超参数，提供概率输出

这些方法在实践中非常成功，特别是在高维数据上。
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .support_vector_machines import (
    SVM,
    SimplifiedSMO,
    visualize_svm_decision_boundary,
    demonstrate_svm_classification,
    compare_c_parameter,
    demonstrate_smo_algorithm
)

from .relevance_vector_machines import (
    RVMRegressor,
    RVMClassifier,
    demonstrate_rvm_regression,
    demonstrate_rvm_classification,
    compare_svm_rvm
)


def run_chapter07(cfg: DictConfig) -> None:
    """
    运行第7章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第7章：稀疏核机器 (Sparse Kernel Machines)")
    print("="*80)
    
    # 7.1 支持向量机
    print("\n" + "-"*60)
    print("7.1 支持向量机 (Support Vector Machines)")
    print("-"*60)
    
    # SVM分类演示
    demonstrate_svm_classification(
        show_plot=cfg.visualization.show_plots
    )
    
    # C参数影响
    compare_c_parameter(
        show_plot=cfg.visualization.show_plots
    )
    
    # SMO算法演示
    demonstrate_smo_algorithm(
        show_plot=cfg.visualization.show_plots
    )
    
    # 多类分类SVM
    demonstrate_multiclass_svm(
        show_plot=cfg.visualization.show_plots
    )
    
    # 7.2 相关向量机
    print("\n" + "-"*60)
    print("7.2 相关向量机 (Relevance Vector Machines)")
    print("-"*60)
    
    # RVM回归
    demonstrate_rvm_regression(
        show_plot=cfg.visualization.show_plots
    )
    
    # RVM分类
    demonstrate_rvm_classification(
        show_plot=cfg.visualization.show_plots
    )
    
    # SVM vs RVM比较
    compare_svm_rvm(
        show_plot=cfg.visualization.show_plots
    )
    
    # 稀疏性分析
    print("\n" + "-"*60)
    print("稀疏性分析")
    print("-"*60)
    
    analyze_sparsity(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第7章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. SVM通过最大化间隔获得良好泛化")
    print("2. 核技巧使SVM能处理非线性问题")
    print("3. 稀疏性使预测高效")
    print("4. RVM提供概率输出和自动参数选择")
    print("5. RVM通常比SVM更稀疏")
    print("6. 两种方法都是实践中的重要工具")


def demonstrate_multiclass_svm(show_plot: bool = True) -> None:
    """
    演示多类分类SVM
    
    展示一对一(OVO)和一对多(OVR)策略。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n多类分类SVM演示")
    print("=" * 60)
    
    # 生成三类数据
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=300, n_features=2, 
                              n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, n_classes=3,
                              random_state=42)
    
    # 将标签转换为-1, 0, 1
    y = y - 1
    
    print(f"数据集：")
    print(f"  样本数: {len(X)}")
    print(f"  类别数: {len(np.unique(y))}")
    
    # One-vs-Rest策略
    print("\n一对多(One-vs-Rest)策略：")
    ovr_classifiers = []
    
    for class_label in np.unique(y):
        # 创建二分类问题
        y_binary = np.where(y == class_label, 1, -1)
        
        # 训练SVM
        svm = SVM(C=1.0, kernel='rbf', gamma=0.5)
        svm.fit(X, y_binary)
        ovr_classifiers.append(svm)
        
        print(f"  类{class_label} vs 其他: {len(svm.support_vectors_)}个支持向量")
    
    # 预测函数
    def predict_ovr(X_test):
        # 获取每个分类器的决策函数值
        scores = np.array([clf.decision_function(X_test) 
                          for clf in ovr_classifiers]).T
        # 选择得分最高的类
        return np.argmax(scores, axis=1) - 1
    
    # 计算准确率
    y_pred_ovr = predict_ovr(X)
    acc_ovr = np.mean(y_pred_ovr == y)
    print(f"  OVR训练准确率: {acc_ovr:.3f}")
    
    # One-vs-One策略
    print("\n一对一(One-vs-One)策略：")
    ovo_classifiers = {}
    classes = np.unique(y)
    
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if i < j:
                # 选择两个类的数据
                mask = np.logical_or(y == class_i, y == class_j)
                X_pair = X[mask]
                y_pair = y[mask]
                y_pair = np.where(y_pair == class_i, -1, 1)
                
                # 训练SVM
                svm = SVM(C=1.0, kernel='rbf', gamma=0.5)
                svm.fit(X_pair, y_pair)
                ovo_classifiers[(class_i, class_j)] = svm
                
                print(f"  类{class_i} vs 类{class_j}: "
                      f"{len(svm.support_vectors_)}个支持向量")
    
    # 预测函数（投票）
    def predict_ovo(X_test):
        n_test = len(X_test)
        votes = np.zeros((n_test, len(classes)))
        
        for (class_i, class_j), clf in ovo_classifiers.items():
            predictions = clf.predict(X_test)
            # -1表示class_i获胜，1表示class_j获胜
            votes[predictions == -1, classes == class_i] += 1
            votes[predictions == 1, classes == class_j] += 1
        
        return classes[np.argmax(votes, axis=1)]
    
    # 计算准确率
    y_pred_ovo = predict_ovo(X)
    acc_ovo = np.mean(y_pred_ovo == y)
    print(f"  OVO训练准确率: {acc_ovo:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # OVR决策边界
        ax1 = axes[0]
        Z_ovr = predict_ovr(np.c_[xx.ravel(), yy.ravel()])
        Z_ovr = Z_ovr.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z_ovr, alpha=0.3, cmap='viridis')
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                   s=50, edgecolors='black', linewidth=1)
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.set_title(f'One-vs-Rest (准确率={acc_ovr:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # OVO决策边界
        ax2 = axes[1]
        Z_ovo = predict_ovo(np.c_[xx.ravel(), yy.ravel()])
        Z_ovo = Z_ovo.reshape(xx.shape)
        
        ax2.contourf(xx, yy, Z_ovo, alpha=0.3, cmap='viridis')
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                   s=50, edgecolors='black', linewidth=1)
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.set_title(f'One-vs-One (准确率={acc_ovo:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('多类分类SVM策略比较')
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. OVR需要k个分类器（k是类别数）")
    print("2. OVO需要k(k-1)/2个分类器")
    print("3. OVO通常更准确但计算量大")
    print("4. OVR的决策区域可能有歧义区域")


def analyze_sparsity(show_plot: bool = True) -> None:
    """
    分析稀疏性
    
    比较不同方法和参数下的稀疏性。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\n稀疏性分析")
    print("=" * 60)
    
    # 生成不同规模的数据集
    sample_sizes = [50, 100, 200, 500]
    
    svm_sparsity = []
    rvm_sparsity = []
    
    for n_samples in sample_sizes:
        # 生成数据
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
        y = np.where(y == 0, -1, 1)  # SVM需要-1和1
        
        # SVM
        svm = SVM(C=1.0, kernel='rbf', gamma=0.5)
        svm.fit(X, y)
        svm_sparse = len(svm.support_vectors_) / n_samples
        svm_sparsity.append(svm_sparse)
        
        # RVM（需要0和1标签）
        y_rvm = np.where(y == -1, 0, 1)
        rvm = RVMClassifier(kernel='rbf', gamma=0.5)
        rvm.fit(X, y_rvm)
        rvm_sparse = len(rvm.relevance_vectors_) / n_samples
        rvm_sparsity.append(rvm_sparse)
        
        print(f"\nn_samples = {n_samples}:")
        print(f"  SVM: {len(svm.support_vectors_)}个支持向量 ({svm_sparse:.1%})")
        print(f"  RVM: {len(rvm.relevance_vectors_)}个相关向量 ({rvm_sparse:.1%})")
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        plt.plot(sample_sizes, svm_sparsity, 'b-o', 
                label='SVM', linewidth=2, markersize=8)
        plt.plot(sample_sizes, rvm_sparsity, 'r-s', 
                label='RVM', linewidth=2, markersize=8)
        
        plt.xlabel('训练样本数')
        plt.ylabel('稀疏性（向量数/样本数）')
        plt.title('稀疏性比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # 添加百分比标签
        for i, (size, svm_sp, rvm_sp) in enumerate(zip(sample_sizes, 
                                                        svm_sparsity, 
                                                        rvm_sparsity)):
            plt.text(size, svm_sp + 0.02, f'{svm_sp:.0%}', 
                    ha='center', color='blue')
            plt.text(size, rvm_sp - 0.02, f'{rvm_sp:.0%}', 
                    ha='center', color='red')
        
        plt.tight_layout()
        plt.show()
    
    # C参数对稀疏性的影响
    print("\nC参数对SVM稀疏性的影响：")
    print("-" * 40)
    
    # 固定数据集
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    sv_counts = []
    
    for C in C_values:
        svm = SVM(C=C, kernel='rbf', gamma=0.5)
        svm.fit(X, y)
        sv_counts.append(len(svm.support_vectors_))
        print(f"C = {C:6.2f}: {len(svm.support_vectors_)}个支持向量")
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        plt.semilogx(C_values, sv_counts, 'g-o', linewidth=2, markersize=8)
        plt.xlabel('C参数 (log scale)')
        plt.ylabel('支持向量数')
        plt.title('C参数对SVM稀疏性的影响')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for C, count in zip(C_values, sv_counts):
            plt.text(C, count + 1, str(count), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    print("\n结论：")
    print("1. RVM通常比SVM更稀疏")
    print("2. 稀疏性随数据规模变化")
    print("3. C参数影响SVM的稀疏性")
    print("4. C越大，支持向量越少（硬间隔）")
    print("5. 稀疏性影响预测速度")