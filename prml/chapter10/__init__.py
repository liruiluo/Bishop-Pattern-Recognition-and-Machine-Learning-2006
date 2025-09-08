"""
Chapter 10: Approximate Inference (近似推理)
============================================

本章介绍处理复杂概率模型的近似推理方法。

主要内容：
1. 变分推理 (10.1-10.2)
   - 均场近似
   - 变分贝叶斯
   - 证据下界(ELBO)

2. 变分混合模型 (10.3)
   - 变分GMM
   - 自动相关性判定(ARD)
   
3. 期望传播 (10.7)
   - EP算法
   - 矩匹配
   - 鲁棒回归

核心思想：
当精确推理不可行时，使用优化方法找到最佳近似分布。

变分推理 vs 期望传播：
- VI: 最小化KL(q||p)，全局优化，保证收敛
- EP: 最小化KL(p||q)，局部优化，更精确但可能不收敛

应用场景：
- 大规模贝叶斯模型
- 深度学习中的贝叶斯方法
- 主题模型
- 变分自编码器
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .variational_inference import (
    VariationalGMM,
    VariationalLinearRegression,
    demonstrate_variational_gmm,
    demonstrate_variational_linear
)

from .expectation_propagation import (
    EPBinaryClassifier,
    EPRegressor,
    demonstrate_ep_classification,
    demonstrate_ep_regression
)


def run_chapter10(cfg: DictConfig) -> None:
    """
    运行第10章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第10章：近似推理 (Approximate Inference)")
    print("="*80)
    
    # 10.1-10.2 变分推理
    print("\n" + "-"*60)
    print("10.1-10.2 变分推理")
    print("-"*60)
    
    # 变分GMM
    demonstrate_variational_gmm(
        show_plot=cfg.visualization.show_plots
    )
    
    # 变分线性回归（ARD）
    demonstrate_variational_linear(
        show_plot=cfg.visualization.show_plots
    )
    
    # 均场近似演示
    demonstrate_mean_field(
        show_plot=cfg.visualization.show_plots
    )
    
    # 10.7 期望传播
    print("\n" + "-"*60)
    print("10.7 期望传播")
    print("-"*60)
    
    # EP分类
    demonstrate_ep_classification(
        show_plot=cfg.visualization.show_plots
    )
    
    # EP鲁棒回归
    demonstrate_ep_regression(
        show_plot=cfg.visualization.show_plots
    )
    
    # VI vs EP 比较
    compare_vi_ep(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第10章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 变分推理将推理转化为优化问题")
    print("2. ELBO提供了模型证据的下界")
    print("3. ARD自动进行特征选择")
    print("4. EP通过局部近似获得全局近似")
    print("5. 不同噪声模型提供鲁棒性")
    print("6. 近似方法在大规模问题中必不可少")


def demonstrate_mean_field(show_plot: bool = True) -> None:
    """
    演示均场近似
    """
    print("\n均场近似演示")
    print("=" * 60)
    
    # 创建一个简单的耦合系统
    np.random.seed(42)
    
    # 2D Ising模型作为例子
    size = 20
    beta = 0.5  # 逆温度
    
    # 初始化自旋
    spins = np.random.choice([-1, 1], size=(size, size))
    
    print(f"2D Ising模型：{size}x{size}格子")
    
    # 均场迭代
    max_iter = 50
    magnetizations = []
    
    for iteration in range(max_iter):
        # 计算每个自旋的有效场
        mean_field = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                # 计算邻居的平均磁化
                neighbors = []
                if i > 0:
                    neighbors.append(spins[i-1, j])
                if i < size-1:
                    neighbors.append(spins[i+1, j])
                if j > 0:
                    neighbors.append(spins[i, j-1])
                if j < size-1:
                    neighbors.append(spins[i, j+1])
                
                if neighbors:
                    mean_field[i, j] = beta * np.mean(neighbors)
        
        # 更新自旋（均场近似）
        spins = np.tanh(mean_field)
        
        # 记录磁化强度
        magnetizations.append(np.mean(np.abs(spins)))
    
    print(f"最终平均磁化强度: {magnetizations[-1]:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 初始配置
        ax1 = axes[0]
        im1 = ax1.imshow(np.random.choice([-1, 1], size=(size, size)),
                        cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('初始配置')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # 最终配置
        ax2 = axes[1]
        im2 = ax2.imshow(spins, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title('均场近似结果')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        # 磁化强度演化
        ax3 = axes[2]
        ax3.plot(magnetizations, 'b-', linewidth=2)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('平均磁化强度')
        ax3.set_title('收敛过程')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('均场近似：2D Ising模型', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 均场近似假设变量独立")
    print("2. 迭代更新直到收敛")
    print("3. 忽略了涨落和关联")
    print("4. 在高温（小β）下更准确")


def compare_vi_ep(show_plot: bool = True) -> None:
    """
    比较变分推理和期望传播
    """
    print("\n变分推理 vs 期望传播")
    print("=" * 60)
    
    # 生成一个多峰分布的数据
    np.random.seed(42)
    
    # 混合高斯数据
    n_samples = 300
    
    # 分量1
    X1 = np.random.multivariate_normal([2, 2], [[0.5, 0.3], [0.3, 0.5]], n_samples//3)
    # 分量2
    X2 = np.random.multivariate_normal([-2, -2], [[0.8, -0.2], [-0.2, 0.6]], n_samples//3)
    # 分量3
    X3 = np.random.multivariate_normal([0, 3], [[0.4, 0], [0, 0.8]], n_samples//3)
    
    X = np.vstack([X1, X2, X3])
    
    # 添加标签（用于分类比较）
    y = np.array([0]*(n_samples//3) + [1]*(n_samples//3) + [0]*(n_samples//3))
    
    print(f"数据集：{n_samples}个样本，3个高斯分量")
    
    # 1. 变分GMM
    vgmm = VariationalGMM(n_components=5, alpha_0=0.1)
    vgmm.fit(X)
    
    # 2. EP分类（使用扩展特征）
    X_expanded = np.column_stack([X, X[:, 0]**2, X[:, 1]**2, X[:, 0]*X[:, 1]])
    ep_clf = EPBinaryClassifier(damping=0.5)
    ep_clf.fit(X_expanded, y)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始数据
        ax1 = axes[0, 0]
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, alpha=0.6)
        ax1.set_title('原始数据')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.grid(True, alpha=0.3)
        
        # VI聚类结果
        ax2 = axes[0, 1]
        vi_labels = vgmm.predict(X)
        ax2.scatter(X[:, 0], X[:, 1], c=vi_labels, cmap='viridis', s=20, alpha=0.6)
        ax2.set_title(f'VI聚类 (K={vgmm.n_components})')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.grid(True, alpha=0.3)
        
        # EP分类结果
        ax3 = axes[0, 2]
        ep_pred = ep_clf.predict(X_expanded)
        ax3.scatter(X[:, 0], X[:, 1], c=ep_pred, cmap='coolwarm', s=20, alpha=0.6)
        ax3.set_title(f'EP分类 (精度={np.mean(ep_pred==y):.3f})')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        ax3.grid(True, alpha=0.3)
        
        # VI责任度
        ax4 = axes[1, 0]
        # 显示最大责任度的熵
        max_resp = np.max(vgmm.r_, axis=1)
        scatter4 = ax4.scatter(X[:, 0], X[:, 1], c=max_resp, cmap='YlOrRd',
                              s=20, alpha=0.6, vmin=0, vmax=1)
        ax4.set_title('VI：最大责任度')
        ax4.set_xlabel('特征1')
        ax4.set_ylabel('特征2')
        plt.colorbar(scatter4, ax=ax4)
        ax4.grid(True, alpha=0.3)
        
        # EP预测概率
        ax5 = axes[1, 1]
        ep_proba = ep_clf.predict_proba(X_expanded)
        scatter5 = ax5.scatter(X[:, 0], X[:, 1], c=ep_proba, cmap='RdBu_r',
                              s=20, alpha=0.6, vmin=0, vmax=1)
        ax5.set_title('EP：预测概率')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        plt.colorbar(scatter5, ax=ax5)
        ax5.grid(True, alpha=0.3)
        
        # 收敛比较
        ax6 = axes[1, 2]
        # VI的ELBO
        ax6.plot(vgmm.elbo_, 'b-', linewidth=2, label='VI (ELBO)')
        ax6_twin = ax6.twinx()
        # EP的边缘似然
        ax6_twin.plot(ep_clf.marginal_likelihood_, 'r-', linewidth=2, label='EP (边缘似然)')
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('VI目标', color='b')
        ax6_twin.set_ylabel('EP目标', color='r')
        ax6.set_title('收敛比较')
        ax6.grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.suptitle('变分推理 vs 期望传播', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n方法比较：")
    print("┌─────────────┬──────────────────┬──────────────────┐")
    print("│ 特性        │ 变分推理(VI)     │ 期望传播(EP)     │")
    print("├─────────────┼──────────────────┼──────────────────┤")
    print("│ KL散度      │ KL(q||p)         │ KL(p||q)         │")
    print("│ 近似质量    │ 欠拟合倾向       │ 过拟合倾向       │")
    print("│ 收敛性      │ 保证收敛         │ 可能不收敛       │")
    print("│ 计算复杂度  │ 较低             │ 较高             │")
    print("│ 适用场景    │ 大规模问题       │ 高精度要求       │")
    print("└─────────────┴──────────────────┴──────────────────┘")
    
    print("\n观察：")
    print("1. VI倾向于找到单峰近似")
    print("2. EP能更好地捕获多峰性")
    print("3. VI的ELBO单调增加")
    print("4. EP可能震荡但通常更精确")
    print("5. 选择取决于问题和精度要求")