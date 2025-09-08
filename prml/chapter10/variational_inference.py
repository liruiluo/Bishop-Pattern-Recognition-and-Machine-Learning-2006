"""
10.1-10.2 变分推理 (Variational Inference)
==========================================

变分推理是一种将推理问题转化为优化问题的方法。
核心思想是用一个简单的分布q(Z)来近似复杂的后验分布p(Z|X)。

变分推理的核心公式：
log p(X) = L(q) + KL(q||p)

其中：
- L(q) = ∫ q(Z) log[p(X,Z)/q(Z)] dZ 是证据下界(ELBO)
- KL(q||p) = ∫ q(Z) log[q(Z)/p(Z|X)] dZ 是KL散度

因为KL散度非负，所以L(q) ≤ log p(X)，故L(q)称为证据下界。

均场近似(Mean Field Approximation)：
假设q(Z) = ∏ᵢ qᵢ(Zᵢ)，即各变量独立。

最优分布满足：
log qⱼ*(Zⱼ) = E_{q_{-j}}[log p(X, Z)] + const

其中q_{-j}表示除qⱼ外的所有因子。

变分贝叶斯方法的优势：
1. 提供后验的解析近似
2. 自动进行模型选择（通过ARD）
3. 避免过拟合
4. 计算效率高于MCMC

应用：
- 贝叶斯神经网络
- 主题模型(LDA)
- 变分自编码器(VAE)
- 贝叶斯深度学习
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln, loggamma
from scipy.stats import multivariate_normal, wishart, gamma
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class VariationalGMM:
    """
    变分贝叶斯高斯混合模型
    
    使用变分推理而非EM算法，能自动确定分量数。
    """
    
    def __init__(self, n_components: int = 10,
                 alpha_0: float = 1.0,
                 beta_0: float = 1.0,
                 nu_0: Optional[float] = None,
                 W_0: Optional[np.ndarray] = None,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        """
        初始化变分GMM
        
        Args:
            n_components: 最大分量数K
            alpha_0: Dirichlet分布的浓度参数
            beta_0: 高斯-Wishart分布的精度参数
            nu_0: Wishart分布的自由度
            W_0: Wishart分布的尺度矩阵
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_components = n_components
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.W_0 = W_0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 变分参数
        self.alpha_ = None  # Dirichlet参数
        self.beta_ = None   # 精度参数
        self.nu_ = None     # Wishart自由度
        self.W_ = None      # Wishart尺度矩阵
        self.m_ = None      # 均值参数
        
        # 责任度和统计量
        self.r_ = None      # 责任度矩阵
        self.N_ = None      # 有效点数
        self.x_bar_ = None  # 加权均值
        self.S_ = None      # 加权协方差
        
        # 训练信息
        self.elbo_ = []     # 证据下界历史
        self.converged_ = False
        self.n_iter_ = 0
        
    def _initialize(self, X: np.ndarray) -> None:
        """
        初始化变分参数
        
        Args:
            X: 数据，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 设置先验超参数
        if self.nu_0 is None:
            self.nu_0 = n_features
        if self.W_0 is None:
            self.W_0 = np.eye(n_features)
        
        # 初始化变分参数为先验
        self.alpha_ = np.ones(self.n_components) * self.alpha_0
        self.beta_ = np.ones(self.n_components) * self.beta_0
        self.nu_ = np.ones(self.n_components) * self.nu_0
        self.W_ = np.array([self.W_0.copy() for _ in range(self.n_components)])
        
        # 使用K-means++初始化均值
        from ..chapter09.k_means import KMeans
        kmeans = KMeans(n_clusters=self.n_components, init='k-means++',
                       n_init=1, random_state=self.random_state)
        kmeans.fit(X)
        self.m_ = kmeans.cluster_centers_.copy()
        
        # 初始化责任度（均匀）
        self.r_ = np.ones((n_samples, self.n_components)) / self.n_components
        
    def _e_step(self, X: np.ndarray) -> None:
        """
        E步：更新责任度
        
        计算q(Z_nk) = r_nk，即数据点n属于分量k的概率。
        
        Args:
            X: 数据，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 计算期望统计量
        # E[ln π_k] - Dirichlet分布
        E_ln_pi = digamma(self.alpha_) - digamma(np.sum(self.alpha_))
        
        # E[ln |Λ_k|] - Wishart分布
        E_ln_Lambda = np.zeros(self.n_components)
        for k in range(self.n_components):
            E_ln_Lambda[k] = np.sum(digamma(
                0.5 * (self.nu_[k] + 1 - np.arange(1, n_features + 1))
            ))
            E_ln_Lambda[k] += n_features * np.log(2)
            E_ln_Lambda[k] += np.log(np.linalg.det(self.W_[k]))
        
        # 计算每个数据点的责任度
        ln_rho = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # E[(x_n - μ_k)^T Λ_k (x_n - μ_k)]
            diff = X - self.m_[k]  # shape: (n_samples, n_features)
            
            # 期望精度矩阵
            E_Lambda_k = self.nu_[k] * self.W_[k]
            
            # 马氏距离的期望
            E_mahalanobis = np.sum(diff @ E_Lambda_k * diff, axis=1)
            E_mahalanobis += n_features / self.beta_[k]
            
            ln_rho[:, k] = E_ln_pi[k] + 0.5 * E_ln_Lambda[k] - \
                          0.5 * n_features * np.log(2 * np.pi) - \
                          0.5 * E_mahalanobis
        
        # 归一化得到责任度
        ln_rho_max = np.max(ln_rho, axis=1, keepdims=True)
        rho = np.exp(ln_rho - ln_rho_max)
        self.r_ = rho / np.sum(rho, axis=1, keepdims=True)
        
    def _m_step(self, X: np.ndarray) -> None:
        """
        M步：更新变分参数
        
        使用坐标上升更新每个变分分布的参数。
        
        Args:
            X: 数据，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 计算统计量
        self.N_ = np.sum(self.r_, axis=0)  # shape: (n_components,)
        self.x_bar_ = (self.r_.T @ X) / (self.N_[:, np.newaxis] + 1e-10)
        
        self.S_ = []
        for k in range(self.n_components):
            diff = X - self.x_bar_[k]
            S_k = (self.r_[:, k:k+1] * diff).T @ diff / (self.N_[k] + 1e-10)
            self.S_.append(S_k)
        self.S_ = np.array(self.S_)
        
        # 更新变分参数
        # 更新α (Dirichlet)
        self.alpha_ = self.alpha_0 + self.N_
        
        # 更新β (精度)
        self.beta_ = self.beta_0 + self.N_
        
        # 更新m (均值)
        for k in range(self.n_components):
            self.m_[k] = (self.beta_0 * self.m_[k] + self.N_[k] * self.x_bar_[k]) / \
                        self.beta_[k]
        
        # 更新W (Wishart尺度矩阵)
        W_0_inv = np.linalg.inv(self.W_0)
        for k in range(self.n_components):
            W_k_inv = W_0_inv + self.N_[k] * self.S_[k]
            
            # 添加均值校正项
            diff_m = self.x_bar_[k] - self.m_[k]
            W_k_inv += self.beta_0 * self.N_[k] / self.beta_[k] * \
                      np.outer(diff_m, diff_m)
            
            self.W_[k] = np.linalg.inv(W_k_inv + 1e-6 * np.eye(n_features))
        
        # 更新ν (Wishart自由度)
        self.nu_ = self.nu_0 + self.N_
        
    def _compute_elbo(self, X: np.ndarray) -> float:
        """
        计算证据下界(ELBO)
        
        ELBO = E_q[log p(X, Z, π, μ, Λ)] - E_q[log q(Z, π, μ, Λ)]
        
        Args:
            X: 数据
            
        Returns:
            ELBO值
        """
        n_samples, n_features = X.shape
        
        # E[log p(X|Z, μ, Λ)]
        E_log_p_X = 0
        for k in range(self.n_components):
            if self.N_[k] > 0:
                E_log_p_X += 0.5 * self.N_[k] * (
                    np.sum(digamma(0.5 * (self.nu_[k] + 1 - np.arange(1, n_features+1)))) +
                    n_features * np.log(2) + 
                    np.log(np.linalg.det(self.W_[k])) -
                    n_features * np.log(2 * np.pi) -
                    n_features / self.beta_[k] -
                    self.nu_[k] * np.trace(self.S_[k] @ self.W_[k])
                )
        
        # E[log p(Z|π)]
        E_ln_pi = digamma(self.alpha_) - digamma(np.sum(self.alpha_))
        E_log_p_Z = np.sum(self.r_ @ E_ln_pi)
        
        # E[log p(π)]
        ln_C_alpha0 = gammaln(self.n_components * self.alpha_0) - \
                     self.n_components * gammaln(self.alpha_0)
        E_log_p_pi = ln_C_alpha0 + (self.alpha_0 - 1) * np.sum(E_ln_pi)
        
        # E[log q(Z)]
        E_log_q_Z = np.sum(self.r_ * np.log(self.r_ + 1e-10))
        
        # E[log q(π)]
        ln_C_alpha = gammaln(np.sum(self.alpha_)) - \
                    np.sum(gammaln(self.alpha_))
        E_log_q_pi = ln_C_alpha + np.sum((self.alpha_ - 1) * E_ln_pi)
        
        # 简化：忽略μ和Λ的先验和变分项（它们通常相互抵消）
        
        elbo = E_log_p_X + E_log_p_Z + E_log_p_pi - E_log_q_Z - E_log_q_pi
        
        return elbo
    
    def fit(self, X: np.ndarray) -> 'VariationalGMM':
        """
        拟合变分GMM
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            
        Returns:
            self
        """
        # 初始化
        self._initialize(X)
        
        prev_elbo = -np.inf
        
        # 变分推理迭代
        for iteration in range(self.max_iter):
            # E步
            self._e_step(X)
            
            # M步
            self._m_step(X)
            
            # 计算ELBO
            elbo = self._compute_elbo(X)
            self.elbo_.append(elbo)
            
            # 检查收敛
            if abs(elbo - prev_elbo) < self.tol:
                self.converged_ = True
                break
            
            prev_elbo = elbo
        
        self.n_iter_ = iteration + 1
        
        # 裁剪无效分量（N_k很小的分量）
        self._prune_components()
        
        if self.converged_:
            print(f"变分推理在{self.n_iter_}次迭代后收敛")
        else:
            print(f"变分推理达到最大迭代次数{self.max_iter}")
        
        return self
    
    def _prune_components(self, threshold: float = 1e-3) -> None:
        """
        裁剪无效分量
        
        Args:
            threshold: 有效点数阈值
        """
        # 找出有效分量
        effective_components = self.N_ > threshold
        n_effective = np.sum(effective_components)
        
        if n_effective < self.n_components:
            print(f"裁剪分量：{self.n_components} -> {n_effective}")
            
            # 只保留有效分量
            self.alpha_ = self.alpha_[effective_components]
            self.beta_ = self.beta_[effective_components]
            self.nu_ = self.nu_[effective_components]
            self.W_ = self.W_[effective_components]
            self.m_ = self.m_[effective_components]
            self.N_ = self.N_[effective_components]
            self.r_ = self.r_[:, effective_components]
            
            # 重新归一化责任度
            self.r_ = self.r_ / np.sum(self.r_, axis=1, keepdims=True)
            
            self.n_components = n_effective
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测簇标签
        
        Args:
            X: 数据
            
        Returns:
            簇标签
        """
        self._e_step(X)
        return np.argmax(self.r_, axis=1)
    
    def get_parameters(self) -> Dict:
        """
        获取估计的模型参数
        
        Returns:
            参数字典
        """
        # 混合权重的期望
        E_pi = self.alpha_ / np.sum(self.alpha_)
        
        # 均值的期望
        E_mu = self.m_
        
        # 协方差的期望（近似）
        E_cov = []
        for k in range(self.n_components):
            E_cov.append(np.linalg.inv(self.nu_[k] * self.W_[k]))
        
        return {
            'weights': E_pi,
            'means': E_mu,
            'covariances': np.array(E_cov)
        }


class VariationalLinearRegression:
    """
    变分贝叶斯线性回归
    
    自动确定相关特征（ARD - 自动相关性判定）。
    """
    
    def __init__(self, alpha_0: float = 1e-2,
                 beta_0: float = 1e-2,
                 max_iter: int = 100,
                 tol: float = 1e-4):
        """
        初始化变分线性回归
        
        Args:
            alpha_0: 权重精度的先验参数
            beta_0: 噪声精度的先验参数
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_iter = max_iter
        self.tol = tol
        
        # 变分参数
        self.alpha_ = None  # 权重精度
        self.beta_ = None   # 噪声精度
        self.m_ = None      # 权重均值
        self.S_ = None      # 权重协方差
        
        # 训练信息
        self.elbo_ = []
        self.converged_ = False
        self.n_iter_ = 0
        
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """添加偏置项"""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VariationalLinearRegression':
        """
        拟合变分贝叶斯线性回归
        
        Args:
            X: 特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples,)
            
        Returns:
            self
        """
        # 添加偏置
        Phi = self._add_bias(X)
        n_samples, n_features = Phi.shape
        
        # 初始化变分参数
        self.alpha_ = np.ones(n_features) * self.alpha_0
        self.beta_ = self.beta_0
        
        prev_elbo = -np.inf
        
        for iteration in range(self.max_iter):
            # 更新后验（闭形式解）
            A = np.diag(self.alpha_)
            self.S_ = np.linalg.inv(A + self.beta_ * Phi.T @ Phi)
            self.m_ = self.beta_ * self.S_ @ Phi.T @ y
            
            # 更新超参数
            # 更新α（ARD）
            gamma = 1 - self.alpha_ * np.diag(self.S_)
            self.alpha_ = gamma / (self.m_ ** 2 + 1e-10)
            
            # 更新β
            y_pred = Phi @ self.m_
            residual = y - y_pred
            self.beta_ = (n_samples - np.sum(gamma)) / (np.sum(residual ** 2) + 1e-10)
            
            # 计算ELBO
            elbo = self._compute_elbo(Phi, y)
            self.elbo_.append(elbo)
            
            # 检查收敛
            if abs(elbo - prev_elbo) < self.tol:
                self.converged_ = True
                break
            
            prev_elbo = elbo
        
        self.n_iter_ = iteration + 1
        
        # 裁剪不相关特征
        self._prune_features()
        
        return self
    
    def _compute_elbo(self, Phi: np.ndarray, y: np.ndarray) -> float:
        """
        计算证据下界
        
        Args:
            Phi: 设计矩阵
            y: 目标值
            
        Returns:
            ELBO
        """
        n_samples = Phi.shape[0]
        
        # 数据拟合项
        y_pred = Phi @ self.m_
        data_fit = -0.5 * self.beta_ * np.sum((y - y_pred) ** 2)
        
        # 正则化项
        reg = -0.5 * np.sum(self.alpha_ * self.m_ ** 2)
        
        # 对数行列式项
        log_det = 0.5 * np.log(np.linalg.det(self.S_) + 1e-10)
        
        elbo = data_fit + reg + log_det
        
        return elbo
    
    def _prune_features(self, threshold: float = 1e3) -> None:
        """
        裁剪不相关特征（α很大的特征）
        
        Args:
            threshold: α阈值
        """
        relevant_features = self.alpha_ < threshold
        n_relevant = np.sum(relevant_features)
        
        if n_relevant < len(self.alpha_):
            print(f"ARD裁剪特征：{len(self.alpha_)} -> {n_relevant}")
            print(f"  不相关特征索引: {np.where(~relevant_features)[0]}")
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征
            return_std: 是否返回预测标准差
            
        Returns:
            预测均值（和标准差）
        """
        Phi = self._add_bias(X)
        
        # 预测均值
        y_mean = Phi @ self.m_
        
        if return_std:
            # 预测方差
            y_var = 1 / self.beta_ + np.sum((Phi @ self.S_) * Phi, axis=1)
            y_std = np.sqrt(y_var)
            return y_mean, y_std
        
        return y_mean


def demonstrate_variational_gmm(show_plot: bool = True) -> None:
    """
    演示变分贝叶斯GMM
    """
    print("\n变分贝叶斯GMM演示")
    print("=" * 60)
    
    # 生成数据（3个真实簇）
    np.random.seed(42)
    
    means = [[-2, 2], [2, 2], [0, -2]]
    covs = [0.5 * np.eye(2), 0.3 * np.eye(2), 0.4 * np.eye(2)]
    weights = [0.4, 0.4, 0.2]
    
    n_samples = 500
    X = []
    true_labels = []
    
    for _ in range(n_samples):
        k = np.random.choice(3, p=weights)
        x = np.random.multivariate_normal(means[k], covs[k])
        X.append(x)
        true_labels.append(k)
    
    X = np.array(X)
    true_labels = np.array(true_labels)
    
    print(f"数据集：{n_samples}个样本，3个真实簇")
    
    # 拟合变分GMM（设置较大的初始K）
    vgmm = VariationalGMM(n_components=10, alpha_0=0.01)
    vgmm.fit(X)
    
    print(f"\n最终分量数: {vgmm.n_components}")
    
    # 获取参数
    params = vgmm.get_parameters()
    print(f"混合权重: {params['weights']}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 真实数据
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=true_labels,
                              cmap='viridis', s=20, alpha=0.6)
        ax1.set_title('真实簇')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        plt.colorbar(scatter1, ax=ax1)
        
        # 变分GMM预测
        ax2 = axes[0, 1]
        predictions = vgmm.predict(X)
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=predictions,
                              cmap='viridis', s=20, alpha=0.6)
        ax2.scatter(params['means'][:, 0], params['means'][:, 1],
                   c='red', marker='*', s=300,
                   edgecolors='black', linewidths=2)
        ax2.set_title(f'变分GMM (K={vgmm.n_components})')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        plt.colorbar(scatter2, ax=ax2)
        
        # ELBO曲线
        ax3 = axes[0, 2]
        ax3.plot(vgmm.elbo_, 'b-', linewidth=2)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('ELBO')
        ax3.set_title('证据下界')
        ax3.grid(True, alpha=0.3)
        
        # 分量权重
        ax4 = axes[1, 0]
        ax4.bar(range(len(params['weights'])), params['weights'])
        ax4.set_xlabel('分量索引')
        ax4.set_ylabel('权重')
        ax4.set_title('混合权重')
        ax4.grid(True, alpha=0.3)
        
        # 责任度热图
        ax5 = axes[1, 1]
        # 显示前100个样本的责任度
        im = ax5.imshow(vgmm.r_[:100].T, aspect='auto', cmap='YlOrRd')
        ax5.set_xlabel('样本索引')
        ax5.set_ylabel('分量索引')
        ax5.set_title('责任度矩阵（前100个样本）')
        plt.colorbar(im, ax=ax5)
        
        # 有效点数
        ax6 = axes[1, 2]
        ax6.bar(range(len(vgmm.N_)), vgmm.N_)
        ax6.set_xlabel('分量索引')
        ax6.set_ylabel('有效点数')
        ax6.set_title('N_k（有效样本数）')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('变分贝叶斯GMM分析', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 变分GMM自动确定了有效分量数")
    print("2. 不需要模型选择（自动ARD）")
    print("3. ELBO单调增加保证收敛")
    print("4. 避免了过拟合")


def demonstrate_variational_linear(show_plot: bool = True) -> None:
    """
    演示变分贝叶斯线性回归（ARD）
    """
    print("\n变分贝叶斯线性回归演示")
    print("=" * 60)
    
    # 生成数据（部分特征不相关）
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_relevant = 3  # 只有3个相关特征
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 真实权重（只有前3个非零）
    w_true = np.zeros(n_features)
    w_true[:n_relevant] = np.array([3.0, -2.0, 1.5])
    
    # 生成目标值
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    print(f"数据集：{n_samples}个样本，{n_features}个特征")
    print(f"相关特征数：{n_relevant}")
    
    # 拟合变分线性回归
    vb_reg = VariationalLinearRegression()
    vb_reg.fit(X, y)
    
    print(f"\nARD后权重精度α:")
    for i in range(n_features):
        relevance = "相关" if vb_reg.alpha_[i+1] < 100 else "不相关"
        print(f"  特征{i}: α={vb_reg.alpha_[i+1]:.2e} ({relevance})")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 真实权重 vs 估计权重
        ax1 = axes[0, 0]
        ax1.bar(range(n_features), w_true, alpha=0.5, label='真实')
        ax1.bar(range(n_features), vb_reg.m_[1:], alpha=0.5, label='估计')
        ax1.set_xlabel('特征索引')
        ax1.set_ylabel('权重')
        ax1.set_title('权重比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 权重精度（对数尺度）
        ax2 = axes[0, 1]
        ax2.semilogy(range(n_features), vb_reg.alpha_[1:], 'bo-')
        ax2.axhline(y=100, color='r', linestyle='--', label='阈值')
        ax2.set_xlabel('特征索引')
        ax2.set_ylabel('权重精度α（对数尺度）')
        ax2.set_title('ARD：权重精度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ELBO曲线
        ax3 = axes[0, 2]
        ax3.plot(vb_reg.elbo_, 'b-', linewidth=2)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('ELBO')
        ax3.set_title('证据下界')
        ax3.grid(True, alpha=0.3)
        
        # 预测 vs 真实
        ax4 = axes[1, 0]
        y_pred = vb_reg.predict(X)
        ax4.scatter(y, y_pred, alpha=0.6)
        ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax4.set_xlabel('真实值')
        ax4.set_ylabel('预测值')
        ax4.set_title('预测精度')
        ax4.grid(True, alpha=0.3)
        
        # 预测不确定性
        ax5 = axes[1, 1]
        # 测试数据
        X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
        X_test_full = np.random.randn(100, n_features)
        X_test_full[:, 0] = X_test.ravel()
        
        y_mean, y_std = vb_reg.predict(X_test_full, return_std=True)
        
        ax5.plot(X_test, y_mean, 'b-', label='预测均值')
        ax5.fill_between(X_test.ravel(),
                         y_mean - 2*y_std,
                         y_mean + 2*y_std,
                         alpha=0.3, label='95%置信区间')
        ax5.set_xlabel('特征0')
        ax5.set_ylabel('预测值')
        ax5.set_title('预测不确定性')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 权重后验协方差
        ax6 = axes[1, 2]
        # 显示权重协方差矩阵的对角线（方差）
        weight_var = np.diag(vb_reg.S_[1:, 1:])  # 排除偏置
        ax6.bar(range(n_features), np.sqrt(weight_var))
        ax6.set_xlabel('特征索引')
        ax6.set_ylabel('权重标准差')
        ax6.set_title('权重不确定性')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('变分贝叶斯线性回归（ARD）', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. ARD自动识别了相关特征")
    print("2. 不相关特征的α值很大（权重趋于0）")
    print("3. 提供了预测不确定性估计")
    print("4. 避免了过拟合")