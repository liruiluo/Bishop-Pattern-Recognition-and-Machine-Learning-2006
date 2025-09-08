"""
Chapter 9: Mixture Models and EM (混合模型与EM算法)
===================================================

本章介绍混合模型和期望最大化（EM）算法。

主要内容：
1. K均值聚类 (9.1)
   - 标准K均值
   - K-means++初始化
   - 模糊K均值

2. 混合高斯模型 (9.2)
   - 概率模型
   - 不同协方差类型
   - 模型选择

3. EM算法 (9.3)
   - E步：计算期望
   - M步：最大化
   - 收敛性质

4. 一般EM算法 (9.4)
   - 缺失数据
   - 隐变量模型
   - 变体和扩展

核心概念：
EM算法是处理含有隐变量模型的通用框架。
它保证似然函数单调不减，但可能收敛到局部最优。

K均值vs GMM：
- K均值：硬分配，球形簇
- GMM：软分配，任意椭圆簇
- K均值是GMM的特殊情况

应用：
- 聚类分析
- 密度估计
- 缺失数据插补
- 无监督学习
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .k_means import (
    KMeans,
    FuzzyKMeans,
    elbow_method,
    silhouette_analysis,
    demonstrate_kmeans,
    demonstrate_fuzzy_kmeans
)

from .gaussian_mixture import (
    GaussianMixtureModel,
    demonstrate_gmm,
    model_selection_gmm
)


def run_chapter09(cfg: DictConfig) -> None:
    """
    运行第9章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第9章：混合模型与EM算法 (Mixture Models and EM)")
    print("="*80)
    
    # 9.1 K均值聚类
    print("\n" + "-"*60)
    print("9.1 K均值聚类")
    print("-"*60)
    
    # 标准K均值
    demonstrate_kmeans(
        show_plot=cfg.visualization.show_plots
    )
    
    # 模糊K均值
    demonstrate_fuzzy_kmeans(
        show_plot=cfg.visualization.show_plots
    )
    
    # K值选择
    demonstrate_k_selection(
        show_plot=cfg.visualization.show_plots
    )
    
    # 9.2-9.3 混合高斯模型与EM算法
    print("\n" + "-"*60)
    print("9.2-9.3 混合高斯模型与EM算法")
    print("-"*60)
    
    # GMM演示
    demonstrate_gmm(
        show_plot=cfg.visualization.show_plots
    )
    
    # GMM vs K-means比较
    compare_gmm_kmeans(
        show_plot=cfg.visualization.show_plots
    )
    
    # 协方差类型比较
    compare_covariance_types(
        show_plot=cfg.visualization.show_plots
    )
    
    # 9.4 EM算法的一般形式
    print("\n" + "-"*60)
    print("9.4 EM算法的一般形式")
    print("-"*60)
    
    # EM算法用于缺失数据
    demonstrate_em_missing_data(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第9章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. K均值是简单高效的聚类算法")
    print("2. GMM提供概率框架和软分配")
    print("3. EM算法保证似然单调不减")
    print("4. 初始化对结果影响很大")
    print("5. 信息准则可用于模型选择")
    print("6. EM是处理隐变量的通用框架")


def demonstrate_k_selection(show_plot: bool = True) -> None:
    """
    演示K值选择方法
    """
    print("\nK值选择方法演示")
    print("=" * 60)
    
    # 生成数据（真实K=4）
    np.random.seed(42)
    
    clusters = []
    for center in [(2, 2), (-2, 2), (-2, -2), (2, -2)]:
        cluster = np.random.randn(75, 2) * 0.5 + center
        clusters.append(cluster)
    X = np.vstack(clusters)
    
    print(f"数据集：{len(X)}个样本，真实K=4")
    
    # 尝试不同的K值
    k_range = range(2, 10)
    
    # 计算各种指标
    inertias = []
    silhouettes = []
    
    for k in k_range:
        # K均值
        kmeans = KMeans(n_clusters=k, n_init=5)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_analysis(X, kmeans.labels_))
    
    # GMM的信息准则
    results = model_selection_gmm(X, k_range, show_plot=False)
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 肘部法则
        ax1 = axes[0, 0]
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=4, color='r', linestyle='--', label='真实K=4')
        ax1.set_xlabel('K')
        ax1.set_ylabel('失真度量')
        ax1.set_title('肘部法则 (K-means)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 轮廓系数
        ax2 = axes[0, 1]
        ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
        best_k_silhouette = k_range[np.argmax(silhouettes)]
        ax2.axvline(x=best_k_silhouette, color='b', linestyle='--',
                   label=f'最优K={best_k_silhouette}')
        ax2.axvline(x=4, color='r', linestyle='--', label='真实K=4')
        ax2.set_xlabel('K')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓分析 (K-means)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # AIC
        ax3 = axes[1, 0]
        ax3.plot(k_range, results['aic'], 'mo-', linewidth=2, markersize=8)
        best_k_aic = k_range[np.argmin(results['aic'])]
        ax3.axvline(x=best_k_aic, color='b', linestyle='--',
                   label=f'最优K={best_k_aic}')
        ax3.axvline(x=4, color='r', linestyle='--', label='真实K=4')
        ax3.set_xlabel('K')
        ax3.set_ylabel('AIC')
        ax3.set_title('AIC (GMM)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # BIC
        ax4 = axes[1, 1]
        ax4.plot(k_range, results['bic'], 'co-', linewidth=2, markersize=8)
        best_k_bic = k_range[np.argmin(results['bic'])]
        ax4.axvline(x=best_k_bic, color='b', linestyle='--',
                   label=f'最优K={best_k_bic}')
        ax4.axvline(x=4, color='r', linestyle='--', label='真实K=4')
        ax4.set_xlabel('K')
        ax4.set_ylabel('BIC')
        ax4.set_title('BIC (GMM)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('K值选择方法比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n各方法选择的K值：")
    print(f"  轮廓系数: K={k_range[np.argmax(silhouettes)]}")
    print(f"  AIC: K={k_range[np.argmin(results['aic'])]}")
    print(f"  BIC: K={k_range[np.argmin(results['bic'])]}")
    print(f"  真实值: K=4")
    
    print("\n观察：")
    print("1. 不同方法可能给出不同的K值")
    print("2. BIC倾向于选择较小的K（更保守）")
    print("3. 轮廓系数考虑簇的紧密度和分离度")
    print("4. 需要结合领域知识选择K")


def compare_gmm_kmeans(show_plot: bool = True) -> None:
    """
    比较GMM和K-means
    """
    print("\nGMM vs K-means比较")
    print("=" * 60)
    
    # 生成不同形状和大小的簇
    np.random.seed(42)
    
    # 簇1：小而紧密
    cov1 = np.array([[0.2, 0], [0, 0.2]])
    cluster1 = np.random.multivariate_normal([2, 2], cov1, 100)
    
    # 簇2：大而分散
    cov2 = np.array([[1.0, 0], [0, 1.0]])
    cluster2 = np.random.multivariate_normal([-2, 2], cov2, 100)
    
    # 簇3：椭圆形
    cov3 = np.array([[0.8, 0.6], [0.6, 0.8]])
    cluster3 = np.random.multivariate_normal([0, -2], cov3, 100)
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("数据集：3个不同形状和大小的簇")
    
    # K-means
    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_
    
    # GMM
    gmm = GaussianMixtureModel(n_components=3, covariance_type='full')
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_proba = gmm.predict_proba(X)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始数据
        ax1 = axes[0, 0]
        ax1.scatter(X[:100, 0], X[:100, 1], c='red', s=20, alpha=0.6, label='簇1(小)')
        ax1.scatter(X[100:200, 0], X[100:200, 1], c='green', s=20, alpha=0.6, label='簇2(大)')
        ax1.scatter(X[200:, 0], X[200:, 1], c='blue', s=20, alpha=0.6, label='簇3(椭圆)')
        ax1.set_title('原始数据')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # K-means结果
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels,
                              cmap='viridis', s=20, alpha=0.6)
        ax2.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='*', s=300,
                   edgecolors='black', linewidths=2)
        ax2.set_title('K-means（硬分配）')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        plt.colorbar(scatter2, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # GMM结果
        ax3 = axes[0, 2]
        scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=gmm_labels,
                              cmap='viridis', s=20, alpha=0.6)
        ax3.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='*', s=300,
                   edgecolors='black', linewidths=2)
        ax3.set_title('GMM（硬分配）')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        plt.colorbar(scatter3, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # K-means假设的协方差
        ax4 = axes[1, 0]
        ax4.scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.3)
        
        # 绘制K-means隐含的球形协方差
        from matplotlib.patches import Circle
        for k in range(3):
            cluster_points = X[kmeans_labels == k]
            if len(cluster_points) > 0:
                radius = np.mean(np.linalg.norm(
                    cluster_points - kmeans.cluster_centers_[k], axis=1
                ))
                circle = Circle(kmeans.cluster_centers_[k], radius,
                              alpha=0.3, edgecolor='red', 
                              facecolor='none', linewidth=2)
                ax4.add_patch(circle)
        
        ax4.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='*', s=200)
        ax4.set_title('K-means（球形协方差）')
        ax4.set_xlabel('特征1')
        ax4.set_ylabel('特征2')
        ax4.set_xlim(X[:, 0].min()-1, X[:, 0].max()+1)
        ax4.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1)
        ax4.grid(True, alpha=0.3)
        
        # GMM协方差
        ax5 = axes[1, 1]
        ax5.scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.3)
        
        # 绘制GMM的协方差椭圆
        from matplotlib.patches import Ellipse
        for k in range(3):
            cov = gmm.covariances_[k]
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0],
                                         eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)
            
            ellipse = Ellipse(gmm.means_[k], width, height,
                            angle=angle, alpha=0.3,
                            edgecolor='red', facecolor='none',
                            linewidth=2)
            ax5.add_patch(ellipse)
        
        ax5.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='*', s=200)
        ax5.set_title('GMM（完整协方差）')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        ax5.set_xlim(X[:, 0].min()-1, X[:, 0].max()+1)
        ax5.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1)
        ax5.grid(True, alpha=0.3)
        
        # GMM软分配
        ax6 = axes[1, 2]
        # 显示最大责任度
        max_proba = np.max(gmm_proba, axis=1)
        scatter6 = ax6.scatter(X[:, 0], X[:, 1], c=max_proba,
                              cmap='RdYlBu_r', s=20, alpha=0.6,
                              vmin=0.33, vmax=1.0)
        ax6.set_title('GMM软分配（最大责任度）')
        ax6.set_xlabel('特征1')
        ax6.set_ylabel('特征2')
        plt.colorbar(scatter6, ax=ax6)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('K-means vs GMM', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n主要区别：")
    print("1. 协方差：K-means假设球形，GMM可以任意形状")
    print("2. 分配：K-means硬分配，GMM软分配")
    print("3. 概率：K-means无概率解释，GMM是概率模型")
    print("4. 簇大小：K-means倾向等大小，GMM可以不同大小")


def compare_covariance_types(show_plot: bool = True) -> None:
    """
    比较GMM的不同协方差类型
    """
    print("\nGMM协方差类型比较")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    # 生成椭圆形簇
    mean1 = [2, 2]
    cov1 = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    mean2 = [-2, -2]
    cov2 = np.array([[0.5, -0.3], [-0.3, 0.8]])
    
    X1 = np.random.multivariate_normal(mean1, cov1, 200)
    X2 = np.random.multivariate_normal(mean2, cov2, 200)
    X = np.vstack([X1, X2])
    
    print("数据集：2个椭圆形簇")
    
    # 不同协方差类型
    cov_types = ['spherical', 'diag', 'tied', 'full']
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, cov_type in enumerate(cov_types):
            ax = axes[idx]
            
            # 拟合GMM
            gmm = GaussianMixtureModel(n_components=2, 
                                      covariance_type=cov_type)
            gmm.fit(X)
            
            # 绘制数据和预测
            labels = gmm.predict(X)
            ax.scatter(X[:, 0], X[:, 1], c=labels, 
                      cmap='coolwarm', s=20, alpha=0.6)
            
            # 绘制协方差
            from matplotlib.patches import Ellipse
            for k in range(2):
                if cov_type == 'tied':
                    cov = gmm.covariances_
                else:
                    cov = gmm.covariances_[k]
                
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0],
                                             eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(eigenvalues)
                
                ellipse = Ellipse(gmm.means_[k], width, height,
                                angle=angle, alpha=0.3,
                                edgecolor='black', facecolor='none',
                                linewidth=2)
                ax.add_patch(ellipse)
            
            ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                      c='red', marker='*', s=200,
                      edgecolors='black', linewidths=2)
            
            # 计算AIC
            aic = gmm.aic(X)
            
            ax.set_title(f'{cov_type}\nAIC={aic:.1f}')
            ax.set_xlabel('特征1')
            ax.set_ylabel('特征2')
            ax.set_xlim(X[:, 0].min()-1, X[:, 0].max()+1)
            ax.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('GMM协方差类型比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n协方差类型说明：")
    print("1. spherical：σ²I，每个分量一个参数")
    print("2. diag：对角矩阵，每个分量d个参数")
    print("3. tied：共享协方差，所有分量共享d(d+1)/2个参数")
    print("4. full：完整协方差，每个分量d(d+1)/2个参数")
    print("\n权衡：")
    print("- 参数越多，表达能力越强")
    print("- 参数越多，越容易过拟合")
    print("- 需要根据数据量和问题选择")


def demonstrate_em_missing_data(show_plot: bool = True) -> None:
    """
    演示EM算法处理缺失数据
    """
    print("\nEM算法处理缺失数据")
    print("=" * 60)
    
    # 生成完整数据
    np.random.seed(42)
    n_samples = 200
    
    # 二维高斯分布
    mean = [2, 3]
    cov = [[1, 0.5], [0.5, 1]]
    X_complete = np.random.multivariate_normal(mean, cov, n_samples)
    
    # 随机缺失一些数据
    missing_ratio = 0.3
    X_missing = X_complete.copy()
    missing_mask = np.random.rand(n_samples, 2) < missing_ratio
    X_missing[missing_mask] = np.nan
    
    n_missing = np.sum(missing_mask)
    print(f"数据集：{n_samples}个样本，{n_missing}个缺失值({n_missing/(n_samples*2):.1%})")
    
    # EM算法填补缺失值
    def em_imputation(X, max_iter=100, tol=1e-4):
        """
        使用EM算法填补缺失值
        
        假设数据来自单个高斯分布
        """
        X_imputed = X.copy()
        n_samples, n_features = X.shape
        
        # 初始化：用均值填补
        for j in range(n_features):
            col = X[:, j]
            mean_j = np.nanmean(col)
            X_imputed[np.isnan(col), j] = mean_j
        
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # M步：估计参数
            mu = np.mean(X_imputed, axis=0)
            Sigma = np.cov(X_imputed.T)
            
            # E步：更新缺失值的期望
            for i in range(n_samples):
                missing_idx = np.isnan(X[i])
                if np.any(missing_idx):
                    observed_idx = ~missing_idx
                    
                    # 条件期望
                    # E[X_m | X_o] = μ_m + Σ_{mo} Σ_{oo}^{-1} (X_o - μ_o)
                    mu_m = mu[missing_idx]
                    mu_o = mu[observed_idx]
                    Sigma_mo = Sigma[np.ix_(missing_idx, observed_idx)]
                    Sigma_oo = Sigma[np.ix_(observed_idx, observed_idx)]
                    
                    try:
                        X_imputed[i, missing_idx] = mu_m + \
                            Sigma_mo @ np.linalg.solve(Sigma_oo, 
                                                       X_imputed[i, observed_idx] - mu_o)
                    except:
                        X_imputed[i, missing_idx] = mu_m
            
            # 计算对数似然（仅观测数据）
            ll = 0
            for i in range(n_samples):
                observed_idx = ~np.isnan(X[i])
                if np.any(observed_idx):
                    x_o = X[i, observed_idx]
                    mu_o = mu[observed_idx]
                    Sigma_oo = Sigma[np.ix_(observed_idx, observed_idx)]
                    
                    try:
                        from scipy.stats import multivariate_normal
                        ll += multivariate_normal.logpdf(x_o, mu_o, Sigma_oo)
                    except:
                        pass
            
            # 检查收敛
            if abs(ll - prev_ll) < tol:
                print(f"EM收敛于{iteration}次迭代")
                break
            prev_ll = ll
        
        return X_imputed, mu, Sigma
    
    # 执行EM插补
    X_em_imputed, mu_em, Sigma_em = em_imputation(X_missing)
    
    # 简单插补（均值）
    X_mean_imputed = X_missing.copy()
    for j in range(2):
        col = X_missing[:, j]
        mean_j = np.nanmean(col)
        X_mean_imputed[np.isnan(col), j] = mean_j
    
    # 计算插补误差
    em_error = np.mean((X_em_imputed[missing_mask] - X_complete[missing_mask])**2)
    mean_error = np.mean((X_mean_imputed[missing_mask] - X_complete[missing_mask])**2)
    
    print(f"\n插补误差（MSE）：")
    print(f"  EM算法: {em_error:.4f}")
    print(f"  均值插补: {mean_error:.4f}")
    print(f"  改进: {(mean_error - em_error)/mean_error:.1%}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 完整数据
        ax1 = axes[0, 0]
        ax1.scatter(X_complete[:, 0], X_complete[:, 1],
                   c='blue', s=20, alpha=0.6)
        ax1.set_title('完整数据')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.grid(True, alpha=0.3)
        
        # 缺失数据
        ax2 = axes[0, 1]
        # 绘制非缺失点
        complete_points = ~np.any(missing_mask, axis=1)
        partial_missing = np.any(missing_mask, axis=1) & ~np.all(missing_mask, axis=1)
        
        ax2.scatter(X_complete[complete_points, 0],
                   X_complete[complete_points, 1],
                   c='blue', s=20, alpha=0.6, label='完整')
        ax2.scatter(X_complete[partial_missing, 0],
                   X_complete[partial_missing, 1],
                   c='orange', s=20, alpha=0.6, label='部分缺失')
        ax2.set_title('缺失模式')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # EM插补结果
        ax3 = axes[0, 2]
        ax3.scatter(X_em_imputed[:, 0], X_em_imputed[:, 1],
                   c='green', s=20, alpha=0.6)
        ax3.scatter(X_em_imputed[missing_mask.any(axis=1), 0],
                   X_em_imputed[missing_mask.any(axis=1), 1],
                   c='red', s=30, alpha=0.8, marker='^',
                   label='插补值')
        ax3.set_title(f'EM插补 (MSE={em_error:.4f})')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 插补误差分布
        ax4 = axes[1, 0]
        em_errors = X_em_imputed[missing_mask] - X_complete[missing_mask]
        mean_errors = X_mean_imputed[missing_mask] - X_complete[missing_mask]
        
        ax4.hist(em_errors, bins=30, alpha=0.5, color='green', 
                label='EM', density=True)
        ax4.hist(mean_errors, bins=30, alpha=0.5, color='red',
                label='均值', density=True)
        ax4.set_xlabel('插补误差')
        ax4.set_ylabel('密度')
        ax4.set_title('插补误差分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 协方差比较
        ax5 = axes[1, 1]
        from matplotlib.patches import Ellipse
        
        # 真实协方差
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0],
                                     eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        ellipse_true = Ellipse(mean, width, height,
                              angle=angle, alpha=0.3,
                              edgecolor='blue', facecolor='none',
                              linewidth=2, label='真实')
        ax5.add_patch(ellipse_true)
        
        # EM估计的协方差
        eigenvalues, eigenvectors = np.linalg.eig(Sigma_em)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0],
                                     eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        ellipse_em = Ellipse(mu_em, width, height,
                            angle=angle, alpha=0.3,
                            edgecolor='green', facecolor='none',
                            linewidth=2, linestyle='--', label='EM估计')
        ax5.add_patch(ellipse_em)
        
        ax5.scatter(X_complete[:, 0], X_complete[:, 1],
                   c='gray', s=10, alpha=0.3)
        ax5.set_xlim(X_complete[:, 0].min()-1, X_complete[:, 0].max()+1)
        ax5.set_ylim(X_complete[:, 1].min()-1, X_complete[:, 1].max()+1)
        ax5.set_title('协方差估计')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 迭代过程（简化展示）
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.7, 'EM算法步骤：', fontsize=14, 
                ha='center', weight='bold')
        ax6.text(0.5, 0.5, '1. E步：估计缺失值的期望', fontsize=12, ha='center')
        ax6.text(0.5, 0.4, '2. M步：更新参数（μ, Σ）', fontsize=12, ha='center')
        ax6.text(0.5, 0.3, '3. 重复直到收敛', fontsize=12, ha='center')
        ax6.text(0.5, 0.1, f'结果：MSE降低{(mean_error-em_error)/mean_error:.1%}',
                fontsize=12, ha='center', color='green', weight='bold')
        ax6.axis('off')
        
        plt.suptitle('EM算法处理缺失数据', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EM算法利用数据的相关性")
    print("2. 比简单插补方法更准确")
    print("3. 迭代改进参数估计")
    print("4. 适用于MAR（随机缺失）数据")