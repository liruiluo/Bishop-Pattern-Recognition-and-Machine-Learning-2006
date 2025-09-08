"""
Chapter 12: Continuous Latent Variables (连续潜变量)
====================================================

本章介绍连续潜变量模型，用于降维和特征提取。

主要内容：
1. 主成分分析 (12.1)
   - 线性PCA
   - 概率PCA
   - 核PCA

2. 因子分析 (12.2)
   - 最大似然FA
   - EM算法

3. 独立成分分析 (12.4)
   - FastICA
   - 盲源分离

核心概念：
潜变量模型假设高维观测数据由低维潜在因素生成。

数学框架：
x = f(z) + ε
其中z是潜变量，f是映射函数，ε是噪声。

线性模型：
- PCA: x = Wz + μ, z ~ N(0,I), ε ~ N(0,σ²I)
- FA: x = Λz + μ + ε, z ~ N(0,I), ε ~ N(0,Ψ)
- ICA: x = As, s独立非高斯

主要区别：
- PCA：正交投影，最大方差
- FA：斜交投影，解释协方差
- ICA：非正交，统计独立

应用：
- 数据压缩
- 特征提取
- 去噪
- 可视化
- 信号分离
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .pca import (
    PCA,
    ProbabilisticPCA,
    KernelPCA,
    demonstrate_pca,
    demonstrate_probabilistic_pca,
    demonstrate_kernel_pca
)

from .ica import (
    FastICA,
    generate_sources,
    demonstrate_ica,
    demonstrate_ica_vs_pca
)


def run_chapter12(cfg: DictConfig) -> None:
    """
    运行第12章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第12章：连续潜变量 (Continuous Latent Variables)")
    print("="*80)
    
    # 12.1 主成分分析
    print("\n" + "-"*60)
    print("12.1 主成分分析 (PCA)")
    print("-"*60)
    
    # 标准PCA
    demonstrate_pca(
        show_plot=cfg.visualization.show_plots
    )
    
    # 概率PCA
    demonstrate_probabilistic_pca(
        show_plot=cfg.visualization.show_plots
    )
    
    # 核PCA
    demonstrate_kernel_pca(
        show_plot=cfg.visualization.show_plots
    )
    
    # 12.2 因子分析
    print("\n" + "-"*60)
    print("12.2 因子分析 (FA)")
    print("-"*60)
    
    demonstrate_factor_analysis(
        show_plot=cfg.visualization.show_plots
    )
    
    # 12.4 独立成分分析
    print("\n" + "-"*60)
    print("12.4 独立成分分析 (ICA)")
    print("-"*60)
    
    # ICA盲源分离
    demonstrate_ica(
        show_plot=cfg.visualization.show_plots
    )
    
    # ICA vs PCA
    demonstrate_ica_vs_pca(
        show_plot=cfg.visualization.show_plots
    )
    
    # 降维方法比较
    compare_dimensionality_reduction(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第12章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. PCA找最大方差方向")
    print("2. 概率PCA提供生成模型")
    print("3. 核PCA处理非线性关系")
    print("4. FA解释协方差结构")
    print("5. ICA寻找独立成分")
    print("6. 不同方法适用不同场景")


class FactorAnalysis:
    """
    因子分析
    
    假设数据由少数潜在因子加上独立噪声生成。
    """
    
    def __init__(self, n_factors: int = 2,
                 max_iter: int = 1000,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        """
        初始化因子分析
        
        Args:
            n_factors: 因子数量
            max_iter: EM迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 模型参数
        self.mean_ = None
        self.components_ = None  # 载荷矩阵Λ
        self.noise_variance_ = None  # 噪声方差Ψ（对角）
        
        # 训练信息
        self.n_iter_ = 0
        self.ll_curve_ = []
        
    def fit(self, X: np.ndarray) -> 'FactorAnalysis':
        """
        使用EM算法拟合FA模型
        
        Args:
            X: 数据矩阵
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 初始化参数
        # 使用PCA初始化
        pca = PCA(n_components=self.n_factors)
        pca.fit(X)
        self.components_ = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # 初始化噪声方差
        var_exp = np.sum(pca.explained_variance_[:self.n_factors])
        var_total = np.var(X_centered, axis=0).sum()
        self.noise_variance_ = np.ones(n_features) * (var_total - var_exp) / n_features
        
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            # E步：计算因子的后验
            # E[z|x] 和 E[zz^T|x]
            
            # 计算精度矩阵
            Psi_inv = np.diag(1.0 / self.noise_variance_)
            
            # 后验协方差
            # Σ_z = (I + Λ^T Ψ^{-1} Λ)^{-1}
            Sigma_z = np.linalg.inv(
                np.eye(self.n_factors) + self.components_.T @ Psi_inv @ self.components_
            )
            
            # 后验均值
            # E[z|x] = Σ_z Λ^T Ψ^{-1} x
            Ez = X_centered @ Psi_inv @ self.components_ @ Sigma_z
            
            # E[zz^T|x] = Σ_z + E[z|x]E[z|x]^T
            Ezz = Sigma_z + Ez.T @ Ez / n_samples
            
            # M步：更新参数
            # 更新Λ
            self.components_ = (X_centered.T @ Ez) @ np.linalg.inv(Ezz)
            
            # 更新Ψ
            for i in range(n_features):
                x_i = X_centered[:, i]
                lambda_i = self.components_[i]
                self.noise_variance_[i] = np.mean(x_i ** 2) - \
                                         2 * np.mean(x_i * (Ez @ lambda_i)) + \
                                         lambda_i @ Ezz @ lambda_i
                self.noise_variance_[i] = max(self.noise_variance_[i], 1e-6)
            
            # 计算对数似然
            ll = self._log_likelihood(X_centered)
            self.ll_curve_.append(ll)
            
            # 检查收敛
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        
        self.n_iter_ = iteration + 1
        return self
    
    def _log_likelihood(self, X_centered: np.ndarray) -> float:
        """
        计算对数似然
        
        Args:
            X_centered: 中心化数据
            
        Returns:
            对数似然
        """
        n_samples, n_features = X_centered.shape
        
        # 协方差矩阵 C = ΛΛ^T + Ψ
        C = self.components_ @ self.components_.T + np.diag(self.noise_variance_)
        
        # 多元高斯的对数似然
        sign, logdet = np.linalg.slogdet(C)
        inv_C = np.linalg.inv(C)
        
        ll = -0.5 * n_samples * (n_features * np.log(2 * np.pi) + logdet)
        for x in X_centered:
            ll -= 0.5 * x @ inv_C @ x
        
        return ll
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        提取因子得分
        
        Args:
            X: 数据
            
        Returns:
            因子得分
        """
        X_centered = X - self.mean_
        
        # 因子得分：E[z|x]
        Psi_inv = np.diag(1.0 / self.noise_variance_)
        Sigma_z = np.linalg.inv(
            np.eye(self.n_factors) + self.components_.T @ Psi_inv @ self.components_
        )
        scores = X_centered @ Psi_inv @ self.components_ @ Sigma_z
        
        return scores
    
    def get_covariance(self) -> np.ndarray:
        """
        获取模型协方差矩阵
        
        Returns:
            协方差矩阵
        """
        return self.components_ @ self.components_.T + np.diag(self.noise_variance_)


def demonstrate_factor_analysis(show_plot: bool = True) -> None:
    """
    演示因子分析
    """
    print("\n因子分析(FA)演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 500
    n_features = 6
    n_factors = 2
    
    # 潜在因子
    Z = np.random.randn(n_samples, n_factors)
    
    # 载荷矩阵
    Lambda_true = np.random.randn(n_features, n_factors)
    
    # 特异性噪声
    noise_std = np.array([0.5, 0.3, 0.4, 0.6, 0.2, 0.7])
    noise = np.random.randn(n_samples, n_features) * noise_std
    
    # 生成观测数据
    X = Z @ Lambda_true.T + noise
    
    print(f"数据集：{n_samples}个样本，{n_features}个特征")
    print(f"真实因子数：{n_factors}")
    
    # 拟合FA模型
    fa = FactorAnalysis(n_factors=n_factors)
    fa.fit(X)
    
    print(f"\nEM收敛：{fa.n_iter_}次迭代")
    
    # 提取因子得分
    scores = fa.transform(X)
    
    # 比较with PCA
    pca = PCA(n_components=n_factors)
    X_pca = pca.fit_transform(X)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 载荷矩阵
        ax1 = axes[0, 0]
        im1 = ax1.imshow(fa.components_, cmap='coolwarm', aspect='auto')
        ax1.set_xlabel('因子')
        ax1.set_ylabel('变量')
        ax1.set_title('FA载荷矩阵')
        plt.colorbar(im1, ax=ax1)
        
        # 噪声方差
        ax2 = axes[0, 1]
        ax2.bar(range(n_features), fa.noise_variance_, alpha=0.7, color='orange')
        ax2.set_xlabel('变量')
        ax2.set_ylabel('噪声方差')
        ax2.set_title('特异性方差')
        ax2.grid(True, alpha=0.3)
        
        # EM收敛
        ax3 = axes[0, 2]
        ax3.plot(fa.ll_curve_, 'b-', linewidth=2)
        ax3.set_xlabel('迭代')
        ax3.set_ylabel('对数似然')
        ax3.set_title('EM收敛')
        ax3.grid(True, alpha=0.3)
        
        # 因子得分
        ax4 = axes[1, 0]
        scatter4 = ax4.scatter(scores[:, 0], scores[:, 1], 
                              c=np.arange(n_samples), cmap='viridis',
                              s=20, alpha=0.6)
        ax4.set_xlabel('因子1')
        ax4.set_ylabel('因子2')
        ax4.set_title('FA因子得分')
        plt.colorbar(scatter4, ax=ax4)
        ax4.grid(True, alpha=0.3)
        
        # PCA得分（对比）
        ax5 = axes[1, 1]
        scatter5 = ax5.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=np.arange(n_samples), cmap='viridis',
                              s=20, alpha=0.6)
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
        ax5.set_title('PCA得分（对比）')
        plt.colorbar(scatter5, ax=ax5)
        ax5.grid(True, alpha=0.3)
        
        # 协方差重构
        ax6 = axes[1, 2]
        C_model = fa.get_covariance()
        C_empirical = np.cov(X.T)
        
        # 显示相关系数矩阵
        corr_model = C_model / np.sqrt(np.diag(C_model)[:, None] * np.diag(C_model)[None, :])
        corr_empirical = C_empirical / np.sqrt(np.diag(C_empirical)[:, None] * np.diag(C_empirical)[None, :])
        
        diff = corr_model - corr_empirical
        im6 = ax6.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax6.set_title('相关矩阵差异\n(模型 - 经验)')
        plt.colorbar(im6, ax=ax6)
        
        plt.suptitle('因子分析(FA)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. FA模型每个变量的特异性噪声")
    print("2. 载荷矩阵不正交")
    print("3. 解释协方差结构")
    print("4. 适合寻找潜在因素")


def compare_dimensionality_reduction(show_plot: bool = True) -> None:
    """
    比较不同降维方法
    """
    print("\n降维方法比较")
    print("=" * 60)
    
    # 生成瑞士卷数据
    from sklearn.datasets import make_swiss_roll
    n_samples = 1000
    X, color = make_swiss_roll(n_samples, noise=0.1, random_state=42)
    
    print(f"数据集：瑞士卷，{n_samples}个样本")
    
    # 应用不同方法
    methods = {}
    
    # PCA
    pca = PCA(n_components=2)
    methods['PCA'] = pca.fit_transform(X)
    
    # 核PCA
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)
    methods['Kernel PCA'] = kpca.fit_transform(X)
    
    # ICA
    ica = FastICA(n_components=2, random_state=42)
    methods['ICA'] = ica.fit_transform(X)
    
    # FA
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X)
    methods['FA'] = fa.transform(X)
    
    if show_plot:
        fig = plt.figure(figsize=(16, 12))
        
        # 原始数据
        ax = fig.add_subplot(3, 3, 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=10)
        ax.set_title('原始数据（3D）')
        
        # 各方法结果
        for i, (name, X_transformed) in enumerate(methods.items(), 2):
            ax = fig.add_subplot(3, 3, i)
            scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                               c=color, cmap='viridis', s=10, alpha=0.6)
            ax.set_title(name)
            ax.set_xlabel('成分1')
            ax.set_ylabel('成分2')
            plt.colorbar(scatter, ax=ax)
            ax.grid(True, alpha=0.3)
        
        # 方法特性表格
        ax = fig.add_subplot(3, 3, 6)
        ax.axis('off')
        
        table_data = [
            ['方法', '线性', '正交', '概率', '非高斯'],
            ['PCA', '✓', '✓', '✗', '✗'],
            ['PPCA', '✓', '✓', '✓', '✗'],
            ['Kernel PCA', '✗', '✓', '✗', '✗'],
            ['FA', '✓', '✗', '✓', '✗'],
            ['ICA', '✓', '✗', '✗', '✓']
        ]
        
        table = ax.table(cellText=table_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 第一行作为标题
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('方法特性比较')
        
        # 重构误差比较
        ax = fig.add_subplot(3, 3, 7)
        
        errors = []
        labels = []
        
        # PCA重构误差
        X_pca_recon = pca.inverse_transform(methods['PCA'])
        errors.append(np.mean((X - X_pca_recon) ** 2))
        labels.append('PCA')
        
        # ICA重构误差
        X_ica_recon = ica.inverse_transform(methods['ICA'])
        errors.append(np.mean((X - X_ica_recon) ** 2))
        labels.append('ICA')
        
        ax.bar(labels, errors, alpha=0.7, color=['blue', 'purple'])
        ax.set_ylabel('MSE')
        ax.set_title('重构误差')
        ax.grid(True, alpha=0.3)
        
        # 计算时间（模拟）
        ax = fig.add_subplot(3, 3, 8)
        times = [0.01, 0.05, 0.03, 0.02]  # 模拟时间
        ax.bar(['PCA', 'KPCA', 'ICA', 'FA'], times, 
               alpha=0.7, color=['blue', 'green', 'purple', 'orange'])
        ax.set_ylabel('时间(秒)')
        ax.set_title('计算时间（模拟）')
        ax.grid(True, alpha=0.3)
        
        # 应用场景
        ax = fig.add_subplot(3, 3, 9)
        ax.axis('off')
        
        text = """
        应用场景选择：
        
        • PCA：数据压缩、去噪
        • 核PCA：非线性降维
        • ICA：信号分离
        • FA：因素分析
        • PPCA：缺失数据
        
        选择依据：
        - 数据特性
        - 计算资源
        - 解释性需求
        """
        
        ax.text(0.1, 0.5, text, fontsize=10, 
               verticalalignment='center')
        ax.set_title('应用指南')
        
        plt.suptitle('降维方法综合比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n方法选择建议：")
    print("1. 线性关系 → PCA")
    print("2. 非线性流形 → 核PCA或流形学习")
    print("3. 独立源 → ICA")
    print("4. 潜在因素 → FA")
    print("5. 缺失数据 → 概率PCA")


from typing import Optional