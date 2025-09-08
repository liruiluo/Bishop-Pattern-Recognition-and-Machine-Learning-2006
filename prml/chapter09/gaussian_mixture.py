"""
9.2-9.3 混合高斯模型与EM算法 (Gaussian Mixture Models and EM Algorithm)
======================================================================

混合高斯模型(GMM)是多个高斯分布的加权组合：
p(x) = Σₖ πₖ N(x|μₖ, Σₖ)

其中：
- πₖ是第k个分量的混合系数（权重）
- μₖ是第k个分量的均值
- Σₖ是第k个分量的协方差矩阵

EM算法（期望最大化）：
用于含有隐变量模型的最大似然估计。

E步（期望步）：
计算隐变量的后验分布（责任度）：
γₙₖ = P(z_k=1|xₙ) = πₖ N(xₙ|μₖ, Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ, Σⱼ)

M步（最大化步）：
更新参数：
- Nₖ = Σₙ γₙₖ （有效点数）
- μₖ = (1/Nₖ) Σₙ γₙₖ xₙ
- Σₖ = (1/Nₖ) Σₙ γₙₖ (xₙ - μₖ)(xₙ - μₖ)ᵀ
- πₖ = Nₖ/N

EM算法的性质：
1. 保证似然函数单调不减
2. 收敛到局部最优
3. 对初始化敏感
4. 可能收敛很慢

GMM vs K-means：
- K-means是GMM的特殊情况（协方差固定为σ²I，σ→0）
- GMM提供软分配和概率模型
- GMM可以处理不同形状和大小的簇
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class GaussianMixtureModel:
    """
    高斯混合模型
    
    使用EM算法进行参数估计。
    """
    
    def __init__(self, n_components: int = 3,
                 covariance_type: str = 'full',
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 reg_covar: float = 1e-6,
                 init_method: str = 'kmeans',
                 random_state: Optional[int] = None):
        """
        初始化GMM
        
        Args:
            n_components: 混合分量数K
            covariance_type: 协方差类型
                            'full' - 每个分量有完整协方差矩阵
                            'diag' - 对角协方差
                            'spherical' - 球形协方差（σ²I）
                            'tied' - 所有分量共享协方差
            max_iter: 最大迭代次数
            tol: 收敛容差
            reg_covar: 协方差正则化项
            init_method: 初始化方法 ('random', 'kmeans')
            random_state: 随机种子
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_method = init_method
        self.random_state = random_state
        
        # 模型参数
        self.weights_ = None  # 混合系数 π
        self.means_ = None    # 均值 μ
        self.covariances_ = None  # 协方差 Σ
        
        # 训练信息
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = []
        self.responsibilities_ = None  # 责任度矩阵
        
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        初始化模型参数
        
        Args:
            X: 数据，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化混合系数（均匀）
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 初始化均值
        if self.init_method == 'kmeans':
            # 使用K-means初始化
            from .k_means import KMeans
            kmeans = KMeans(n_clusters=self.n_components, 
                          n_init=1, random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
            
            # 基于K-means结果初始化协方差
            labels = kmeans.labels_
            self.covariances_ = []
            
            for k in range(self.n_components):
                mask = labels == k
                if np.sum(mask) > 1:
                    X_k = X[mask]
                    cov_k = np.cov(X_k.T) + self.reg_covar * np.eye(n_features)
                else:
                    cov_k = np.eye(n_features)
                
                if self.covariance_type == 'full':
                    self.covariances_.append(cov_k)
                elif self.covariance_type == 'diag':
                    self.covariances_.append(np.diag(np.diag(cov_k)))
                elif self.covariance_type == 'spherical':
                    variance = np.mean(np.diag(cov_k))
                    self.covariances_.append(variance * np.eye(n_features))
            
            self.covariances_ = np.array(self.covariances_)
            
        else:  # random
            # 随机选择数据点作为初始均值
            indices = np.random.choice(n_samples, self.n_components, 
                                     replace=False)
            self.means_ = X[indices].copy()
            
            # 初始化为数据的协方差
            data_cov = np.cov(X.T) + self.reg_covar * np.eye(n_features)
            
            if self.covariance_type == 'full':
                self.covariances_ = np.array([data_cov] * self.n_components)
            elif self.covariance_type == 'diag':
                diag_cov = np.diag(np.diag(data_cov))
                self.covariances_ = np.array([diag_cov] * self.n_components)
            elif self.covariance_type == 'spherical':
                variance = np.mean(np.diag(data_cov))
                spherical_cov = variance * np.eye(n_features)
                self.covariances_ = np.array([spherical_cov] * self.n_components)
            elif self.covariance_type == 'tied':
                self.covariances_ = data_cov
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算责任度（后验概率）
        
        γₙₖ = πₖ N(xₙ|μₖ, Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ, Σⱼ)
        
        Args:
            X: 数据，shape (n_samples, n_features)
            
        Returns:
            责任度矩阵，shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # 计算每个分量的似然
        for k in range(self.n_components):
            if self.covariance_type == 'tied':
                cov = self.covariances_
            else:
                cov = self.covariances_[k]
            
            # 计算高斯概率密度
            try:
                rv = multivariate_normal(mean=self.means_[k], cov=cov)
                responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
            except:
                # 处理数值问题
                responsibilities[:, k] = 1e-10
        
        # 归一化得到后验概率
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1  # 避免除零
        responsibilities = responsibilities / responsibilities_sum
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M步：更新参数
        
        Args:
            X: 数据，shape (n_samples, n_features)
            responsibilities: 责任度，shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        
        # 计算有效点数
        N_k = responsibilities.sum(axis=0)  # shape (n_components,)
        
        # 更新混合系数
        self.weights_ = N_k / n_samples
        
        # 更新均值
        self.means_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # 更新协方差
        if self.covariance_type == 'full':
            self.covariances_ = []
            for k in range(self.n_components):
                diff = X - self.means_[k]  # shape (n_samples, n_features)
                weighted_diff = responsibilities[:, k:k+1] * diff
                cov = (weighted_diff.T @ diff) / N_k[k]
                # 添加正则化
                cov += self.reg_covar * np.eye(n_features)
                self.covariances_.append(cov)
            self.covariances_ = np.array(self.covariances_)
            
        elif self.covariance_type == 'diag':
            self.covariances_ = []
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff_sq = responsibilities[:, k:k+1] * (diff ** 2)
                var = weighted_diff_sq.sum(axis=0) / N_k[k]
                cov = np.diag(var + self.reg_covar)
                self.covariances_.append(cov)
            self.covariances_ = np.array(self.covariances_)
            
        elif self.covariance_type == 'spherical':
            self.covariances_ = []
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff_sq = responsibilities[:, k] * np.sum(diff ** 2, axis=1)
                var = weighted_diff_sq.sum() / (N_k[k] * n_features)
                cov = (var + self.reg_covar) * np.eye(n_features)
                self.covariances_.append(cov)
            self.covariances_ = np.array(self.covariances_)
            
        elif self.covariance_type == 'tied':
            # 所有分量共享协方差
            cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                cov += weighted_diff.T @ diff
            cov = cov / n_samples + self.reg_covar * np.eye(n_features)
            self.covariances_ = cov
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        计算对数似然
        
        log p(X) = Σₙ log Σₖ πₖ N(xₙ|μₖ, Σₖ)
        
        Args:
            X: 数据
            
        Returns:
            对数似然
        """
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for n in range(n_samples):
            likelihood_n = 0.0
            for k in range(self.n_components):
                if self.covariance_type == 'tied':
                    cov = self.covariances_
                else:
                    cov = self.covariances_[k]
                
                try:
                    rv = multivariate_normal(mean=self.means_[k], cov=cov)
                    likelihood_n += self.weights_[k] * rv.pdf(X[n])
                except:
                    likelihood_n += 1e-10
            
            log_likelihood += np.log(likelihood_n + 1e-10)
        
        return log_likelihood
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        使用EM算法拟合GMM
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            
        Returns:
            self
        """
        # 初始化参数
        self._initialize_parameters(X)
        
        self.log_likelihood_ = []
        prev_log_likelihood = -np.inf
        
        # EM迭代
        for iteration in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X)
            self.responsibilities_ = responsibilities
            
            # M步
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_.append(log_likelihood)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break
            
            prev_log_likelihood = log_likelihood
        
        self.n_iter_ = iteration + 1
        
        if self.converged_:
            print(f"EM算法在{self.n_iter_}次迭代后收敛")
        else:
            print(f"EM算法达到最大迭代次数{self.max_iter}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测数据的簇标签（硬分配）
        
        Args:
            X: 数据，shape (n_samples, n_features)
            
        Returns:
            簇标签，shape (n_samples,)
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测数据的簇概率（软分配）
        
        Args:
            X: 数据
            
        Returns:
            责任度矩阵，shape (n_samples, n_components)
        """
        return self._e_step(X)
    
    def sample(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        从GMM中采样
        
        Args:
            n_samples: 采样数量
            
        Returns:
            samples: 采样数据，shape (n_samples, n_features)
            labels: 分量标签，shape (n_samples,)
        """
        n_features = self.means_.shape[1]
        samples = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)
        
        # 根据混合系数采样分量
        component_samples = np.random.multinomial(n_samples, self.weights_)
        
        start_idx = 0
        for k, n_k in enumerate(component_samples):
            if n_k > 0:
                if self.covariance_type == 'tied':
                    cov = self.covariances_
                else:
                    cov = self.covariances_[k]
                
                # 从第k个高斯分量采样
                samples[start_idx:start_idx+n_k] = np.random.multivariate_normal(
                    self.means_[k], cov, size=n_k
                )
                labels[start_idx:start_idx+n_k] = k
                start_idx += n_k
        
        # 随机打乱顺序
        indices = np.random.permutation(n_samples)
        return samples[indices], labels[indices]
    
    def bic(self, X: np.ndarray) -> float:
        """
        计算贝叶斯信息准则(BIC)
        
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        
        Args:
            X: 数据
            
        Returns:
            BIC值
        """
        n_samples, n_features = X.shape
        
        # 计算参数数量
        n_mean_params = self.n_components * n_features
        n_weight_params = self.n_components - 1  # 因为和为1的约束
        
        if self.covariance_type == 'full':
            n_cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            n_cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            n_cov_params = self.n_components
        elif self.covariance_type == 'tied':
            n_cov_params = n_features * (n_features + 1) / 2
        
        n_params = n_mean_params + n_weight_params + n_cov_params
        
        log_likelihood = self._compute_log_likelihood(X)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic
    
    def aic(self, X: np.ndarray) -> float:
        """
        计算赤池信息准则(AIC)
        
        AIC = -2 * log_likelihood + 2 * n_params
        
        Args:
            X: 数据
            
        Returns:
            AIC值
        """
        n_samples, n_features = X.shape
        
        # 计算参数数量（同BIC）
        n_mean_params = self.n_components * n_features
        n_weight_params = self.n_components - 1
        
        if self.covariance_type == 'full':
            n_cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            n_cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            n_cov_params = self.n_components
        elif self.covariance_type == 'tied':
            n_cov_params = n_features * (n_features + 1) / 2
        
        n_params = n_mean_params + n_weight_params + n_cov_params
        
        log_likelihood = self._compute_log_likelihood(X)
        aic = -2 * log_likelihood + 2 * n_params
        
        return aic


def demonstrate_gmm(show_plot: bool = True) -> None:
    """
    演示高斯混合模型
    """
    print("\n高斯混合模型演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    # 真实的GMM参数
    true_means = np.array([[2, 2], [-2, 2], [0, -2]])
    true_covs = [
        np.array([[0.5, 0.2], [0.2, 0.5]]),
        np.array([[0.8, -0.3], [-0.3, 0.6]]),
        np.array([[0.3, 0], [0, 0.8]])
    ]
    true_weights = np.array([0.3, 0.5, 0.2])
    
    # 生成数据
    n_samples = 500
    samples = []
    true_labels = []
    
    for _ in range(n_samples):
        # 选择分量
        k = np.random.choice(3, p=true_weights)
        # 从该分量采样
        sample = np.random.multivariate_normal(true_means[k], true_covs[k])
        samples.append(sample)
        true_labels.append(k)
    
    X = np.array(samples)
    true_labels = np.array(true_labels)
    
    print(f"数据集：{n_samples}个样本，3个真实分量")
    
    # 拟合GMM
    gmm = GaussianMixtureModel(n_components=3, covariance_type='full',
                               init_method='kmeans')
    gmm.fit(X)
    
    print(f"\nGMM拟合结果：")
    print(f"  迭代次数: {gmm.n_iter_}")
    print(f"  收敛: {gmm.converged_}")
    print(f"  最终对数似然: {gmm.log_likelihood_[-1]:.2f}")
    
    # 显示参数
    print(f"\n估计的混合系数：")
    for k in range(3):
        print(f"  分量{k+1}: {gmm.weights_[k]:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 真实数据分布
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=true_labels,
                              cmap='viridis', s=20, alpha=0.6)
        ax1.set_title('真实分量')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        plt.colorbar(scatter1, ax=ax1)
        
        # GMM预测结果
        ax2 = axes[0, 1]
        predictions = gmm.predict(X)
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=predictions,
                              cmap='viridis', s=20, alpha=0.6)
        ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='*', s=300, 
                   edgecolors='black', linewidths=2)
        ax2.set_title('GMM预测')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        plt.colorbar(scatter2, ax=ax2)
        
        # 责任度（软分配）
        ax3 = axes[0, 2]
        responsibilities = gmm.predict_proba(X)
        # 显示最大责任度
        max_resp = np.max(responsibilities, axis=1)
        scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=max_resp,
                              cmap='RdYlBu_r', s=20, alpha=0.6,
                              vmin=0.33, vmax=1.0)
        ax3.set_title('最大责任度')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        plt.colorbar(scatter3, ax=ax3)
        
        # 对数似然曲线
        ax4 = axes[1, 0]
        ax4.plot(gmm.log_likelihood_, 'b-', linewidth=2)
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('对数似然')
        ax4.set_title('EM算法收敛')
        ax4.grid(True, alpha=0.3)
        
        # 协方差椭圆
        ax5 = axes[1, 1]
        ax5.scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.3)
        
        # 绘制协方差椭圆
        from matplotlib.patches import Ellipse
        for k in range(gmm.n_components):
            # 计算椭圆参数
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
            
            # 绘制2倍标准差椭圆
            ellipse2 = Ellipse(gmm.means_[k], 2*width, 2*height,
                             angle=angle, alpha=0.2,
                             edgecolor='blue', facecolor='none',
                             linewidth=1, linestyle='--')
            ax5.add_patch(ellipse2)
        
        ax5.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='*', s=200)
        ax5.set_title('协方差椭圆')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        ax5.set_xlim(X[:, 0].min()-1, X[:, 0].max()+1)
        ax5.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1)
        
        # 采样
        ax6 = axes[1, 2]
        samples, sample_labels = gmm.sample(n_samples=300)
        scatter6 = ax6.scatter(samples[:, 0], samples[:, 1], 
                              c=sample_labels, cmap='viridis',
                              s=20, alpha=0.6)
        ax6.set_title('从GMM采样')
        ax6.set_xlabel('特征1')
        ax6.set_ylabel('特征2')
        plt.colorbar(scatter6, ax=ax6)
        
        plt.suptitle('高斯混合模型分析', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EM算法保证似然单调增加")
    print("2. GMM可以拟合不同形状的簇")
    print("3. 提供概率分配（责任度）")
    print("4. 可以生成新样本")


def model_selection_gmm(X: np.ndarray, 
                       k_range: range,
                       show_plot: bool = True) -> Dict[str, List[float]]:
    """
    使用信息准则选择GMM的分量数
    
    Args:
        X: 数据
        k_range: K值范围
        show_plot: 是否显示图形
        
    Returns:
        各信息准则的值
    """
    print("\nGMM模型选择")
    print("=" * 60)
    
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    
    for k in k_range:
        print(f"\n拟合K={k}的GMM...")
        gmm = GaussianMixtureModel(n_components=k, 
                                  covariance_type='full',
                                  init_method='kmeans')
        gmm.fit(X)
        
        aic = gmm.aic(X)
        bic = gmm.bic(X)
        ll = gmm.log_likelihood_[-1] if gmm.log_likelihood_ else -np.inf
        
        aic_scores.append(aic)
        bic_scores.append(bic)
        log_likelihoods.append(ll)
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  对数似然: {ll:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # AIC
        ax1 = axes[0]
        ax1.plot(k_range, aic_scores, 'bo-', linewidth=2, markersize=8)
        best_k_aic = k_range[np.argmin(aic_scores)]
        ax1.axvline(x=best_k_aic, color='r', linestyle='--',
                   label=f'最优K={best_k_aic}')
        ax1.set_xlabel('分量数 K')
        ax1.set_ylabel('AIC')
        ax1.set_title('赤池信息准则')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # BIC
        ax2 = axes[1]
        ax2.plot(k_range, bic_scores, 'go-', linewidth=2, markersize=8)
        best_k_bic = k_range[np.argmin(bic_scores)]
        ax2.axvline(x=best_k_bic, color='r', linestyle='--',
                   label=f'最优K={best_k_bic}')
        ax2.set_xlabel('分量数 K')
        ax2.set_ylabel('BIC')
        ax2.set_title('贝叶斯信息准则')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 对数似然
        ax3 = axes[2]
        ax3.plot(k_range, log_likelihoods, 'ro-', linewidth=2, markersize=8)
        ax3.set_xlabel('分量数 K')
        ax3.set_ylabel('对数似然')
        ax3.set_title('对数似然（总是增加）')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('GMM模型选择', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print(f"\n最优K值：")
    print(f"  根据AIC: {k_range[np.argmin(aic_scores)]}")
    print(f"  根据BIC: {k_range[np.argmin(bic_scores)]}")
    
    return {
        'aic': aic_scores,
        'bic': bic_scores,
        'log_likelihood': log_likelihoods
    }