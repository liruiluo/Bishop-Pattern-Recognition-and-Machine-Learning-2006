"""
10.7 期望传播 (Expectation Propagation, EP)
============================================

期望传播是一种确定性近似推理算法，通过迭代优化局部近似来获得全局近似。

核心思想：
将复杂的后验分布分解为多个因子的乘积：
p(θ|D) ∝ p(θ) ∏ᵢ p(xᵢ|θ)

然后用简单分布（如高斯）近似每个因子：
q(θ) = ∏ᵢ f̃ᵢ(θ)

EP算法步骤：
1. 初始化所有近似因子f̃ᵢ
2. 对每个因子i：
   a. 移除因子i：q⁻ⁱ(θ) = q(θ)/f̃ᵢ(θ)
   b. 计算新的后验：p̂(θ) ∝ q⁻ⁱ(θ)fᵢ(θ)
   c. 矩匹配：找到q̂(θ)使得矩与p̂(θ)匹配
   d. 更新因子：f̃ᵢ(θ) = q̂(θ)/q⁻ⁱ(θ)
3. 重复直到收敛

EP vs 变分推理：
- EP：局部KL散度最小化，KL(p||q)
- VI：全局KL散度最小化，KL(q||p)
- EP通常给出更好的近似，但可能不收敛

应用：
- 高斯过程分类
- 近似消息传递
- 稀疏高斯过程
- 深度学习中的近似推理
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.special import expit  # sigmoid函数
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class EPBinaryClassifier:
    """
    期望传播二分类器
    
    使用EP算法进行贝叶斯逻辑回归。
    """
    
    def __init__(self, max_iter: int = 50,
                 damping: float = 0.5,
                 tol: float = 1e-3,
                 alpha: float = 1.0):
        """
        初始化EP分类器
        
        Args:
            max_iter: 最大迭代次数
            damping: 阻尼因子（防止震荡）
            tol: 收敛容差
            alpha: 先验精度
        """
        self.max_iter = max_iter
        self.damping = damping
        self.tol = tol
        self.alpha = alpha
        
        # EP近似参数
        self.site_means_ = None     # 站点均值
        self.site_precisions_ = None  # 站点精度
        self.cavity_means_ = None   # 空腔均值
        self.cavity_vars_ = None    # 空腔方差
        
        # 后验参数
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        
        # 训练信息
        self.converged_ = False
        self.n_iter_ = 0
        self.marginal_likelihood_ = []
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return expit(x)
    
    def _probit(self, x: np.ndarray) -> np.ndarray:
        """Probit函数（高斯CDF）"""
        return norm.cdf(x)
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """添加偏置项"""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def _initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        初始化EP参数
        
        Args:
            X: 特征（已添加偏置）
            y: 标签 {-1, +1}
        """
        n_samples, n_features = X.shape
        
        # 初始化站点参数（高斯近似）
        self.site_means_ = np.zeros((n_samples, n_features))
        self.site_precisions_ = np.zeros((n_samples, n_features, n_features))
        
        # 初始化为先验
        for i in range(n_samples):
            self.site_precisions_[i] = np.eye(n_features) * 1e-4
        
        # 先验参数
        self.prior_mean_ = np.zeros(n_features)
        self.prior_precision_ = np.eye(n_features) * self.alpha
        
    def _compute_cavity(self, X: np.ndarray, site_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算空腔分布（移除一个站点）
        
        Args:
            X: 特征矩阵
            site_idx: 要移除的站点索引
            
        Returns:
            cavity_mean: 空腔均值
            cavity_cov: 空腔协方差
        """
        n_samples, n_features = X.shape
        
        # 计算总精度（先验 + 所有站点）
        total_precision = self.prior_precision_.copy()
        total_mean_times_precision = self.prior_precision_ @ self.prior_mean_
        
        for i in range(n_samples):
            if i != site_idx:
                total_precision += self.site_precisions_[i]
                total_mean_times_precision += self.site_precisions_[i] @ self.site_means_[i]
        
        # 空腔协方差和均值
        cavity_cov = np.linalg.inv(total_precision + 1e-6 * np.eye(n_features))
        cavity_mean = cavity_cov @ total_mean_times_precision
        
        return cavity_mean, cavity_cov
    
    def _moment_matching(self, cavity_mean: np.ndarray,
                        cavity_cov: np.ndarray,
                        x: np.ndarray,
                        y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        矩匹配：计算倾斜分布的高斯近似
        
        对于分类，倾斜分布是：
        p̂(w) ∝ N(w|cavity_mean, cavity_cov) * σ(y * w^T x)
        
        Args:
            cavity_mean: 空腔均值
            cavity_cov: 空腔协方差
            x: 数据点
            y: 标签
            
        Returns:
            new_mean: 新的均值
            new_cov: 新的协方差
        """
        # 计算边缘统计量
        cavity_var_x = x.T @ cavity_cov @ x
        cavity_mean_x = x.T @ cavity_mean
        
        # 使用Probit近似（更稳定）
        z = y * cavity_mean_x / np.sqrt(1 + cavity_var_x)
        
        # 计算矩
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        # 防止数值问题
        if Phi_z < 1e-10:
            Phi_z = 1e-10
        
        # 导数
        d_log_Z = y * phi_z / (Phi_z * np.sqrt(1 + cavity_var_x))
        d2_log_Z = -d_log_Z * (d_log_Z + z / (1 + cavity_var_x))
        
        # 更新均值和协方差
        delta_precision = -d2_log_Z * np.outer(x, x)
        delta_mean_times_precision = d_log_Z * x
        
        # 新的精度和均值
        new_precision = np.linalg.inv(cavity_cov) + delta_precision
        new_cov = np.linalg.inv(new_precision + 1e-6 * np.eye(len(cavity_mean)))
        new_mean = new_cov @ (np.linalg.inv(cavity_cov) @ cavity_mean + delta_mean_times_precision)
        
        return new_mean, new_cov
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EPBinaryClassifier':
        """
        使用EP算法拟合分类器
        
        Args:
            X: 特征，shape (n_samples, n_features)
            y: 标签，shape (n_samples,)，取值{0, 1}
            
        Returns:
            self
        """
        # 转换标签为{-1, +1}
        y = 2 * y - 1
        
        # 添加偏置
        X = self._add_bias(X)
        n_samples, n_features = X.shape
        
        # 初始化
        self._initialize(X, y)
        
        for iteration in range(self.max_iter):
            old_site_means = self.site_means_.copy()
            old_site_precisions = self.site_precisions_.copy()
            
            # 对每个数据点进行EP更新
            for i in range(n_samples):
                # 计算空腔分布
                cavity_mean, cavity_cov = self._compute_cavity(X, i)
                
                # 矩匹配
                new_mean, new_cov = self._moment_matching(
                    cavity_mean, cavity_cov, X[i], y[i]
                )
                
                # 更新站点参数（带阻尼）
                new_precision = np.linalg.inv(new_cov + 1e-6 * np.eye(n_features))
                new_site_precision = new_precision - np.linalg.inv(cavity_cov + 1e-6 * np.eye(n_features))
                new_site_mean = new_cov @ (new_precision @ new_mean - np.linalg.inv(cavity_cov) @ cavity_mean)
                
                # 应用阻尼
                self.site_precisions_[i] = (1 - self.damping) * self.site_precisions_[i] + \
                                          self.damping * new_site_precision
                self.site_means_[i] = (1 - self.damping) * self.site_means_[i] + \
                                     self.damping * new_site_mean
            
            # 计算后验
            self._compute_posterior(X)
            
            # 计算边缘似然
            marginal_ll = self._compute_marginal_likelihood(X, y)
            self.marginal_likelihood_.append(marginal_ll)
            
            # 检查收敛
            mean_diff = np.mean(np.abs(self.site_means_ - old_site_means))
            prec_diff = np.mean(np.abs(self.site_precisions_ - old_site_precisions))
            
            if mean_diff < self.tol and prec_diff < self.tol:
                self.converged_ = True
                break
        
        self.n_iter_ = iteration + 1
        
        if self.converged_:
            print(f"EP在{self.n_iter_}次迭代后收敛")
        else:
            print(f"EP达到最大迭代次数{self.max_iter}")
        
        return self
    
    def _compute_posterior(self, X: np.ndarray) -> None:
        """
        计算后验分布
        
        Args:
            X: 特征矩阵
        """
        n_samples, n_features = X.shape
        
        # 总精度 = 先验精度 + 所有站点精度
        total_precision = self.prior_precision_.copy()
        total_mean_times_precision = self.prior_precision_ @ self.prior_mean_
        
        for i in range(n_samples):
            total_precision += self.site_precisions_[i]
            total_mean_times_precision += self.site_precisions_[i] @ self.site_means_[i]
        
        # 后验协方差和均值
        self.posterior_cov_ = np.linalg.inv(total_precision + 1e-6 * np.eye(n_features))
        self.posterior_mean_ = self.posterior_cov_ @ total_mean_times_precision
    
    def _compute_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算（近似）边缘似然
        
        Args:
            X: 特征
            y: 标签
            
        Returns:
            对数边缘似然
        """
        # 简化计算：使用拉普拉斯近似
        predictions = X @ self.posterior_mean_
        log_likelihood = np.sum(np.log(self._sigmoid(y * predictions) + 1e-10))
        
        # 加上先验项
        log_prior = -0.5 * self.posterior_mean_.T @ self.prior_precision_ @ self.posterior_mean_
        
        return log_likelihood + log_prior
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（带不确定性）
        
        Args:
            X: 特征
            
        Returns:
            预测概率
        """
        X = self._add_bias(X)
        n_samples = X.shape[0]
        
        proba = np.zeros(n_samples)
        
        for i in range(n_samples):
            # 预测分布的均值和方差
            mean = X[i] @ self.posterior_mean_
            var = X[i] @ self.posterior_cov_ @ X[i]
            
            # 使用Probit近似积分
            proba[i] = self._probit(mean / np.sqrt(1 + var))
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征
            
        Returns:
            预测标签
        """
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)


class EPRegressor:
    """
    期望传播回归器
    
    处理非高斯噪声的回归问题。
    """
    
    def __init__(self, noise_model: str = 'laplace',
                 max_iter: int = 50,
                 damping: float = 0.8,
                 tol: float = 1e-3):
        """
        初始化EP回归器
        
        Args:
            noise_model: 噪声模型 ('laplace', 'student_t')
            max_iter: 最大迭代次数
            damping: 阻尼因子
            tol: 收敛容差
        """
        self.noise_model = noise_model
        self.max_iter = max_iter
        self.damping = damping
        self.tol = tol
        
        # EP参数
        self.site_means_ = None
        self.site_vars_ = None
        
        # 后验参数
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        
        # 训练信息
        self.converged_ = False
        self.n_iter_ = 0
        
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """添加偏置项"""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
    
    def _laplace_moments(self, mu: float, sigma2: float, y: float, scale: float = 1.0) -> Tuple[float, float]:
        """
        计算拉普拉斯分布的高斯近似矩
        
        p(y|f) = (1/2b) exp(-|y-f|/b)
        
        Args:
            mu: 空腔均值
            sigma2: 空腔方差
            y: 观测值
            scale: 拉普拉斯尺度参数
            
        Returns:
            新均值和方差
        """
        # 使用变分近似
        # 对于拉普拉斯噪声，最优近似是收缩估计
        kappa = 1.0 / (sigma2 + scale**2)
        
        new_var = 1.0 / (1.0 / sigma2 + kappa)
        new_mean = new_var * (mu / sigma2 + kappa * y)
        
        return new_mean, new_var
    
    def _student_t_moments(self, mu: float, sigma2: float, y: float, df: float = 3.0) -> Tuple[float, float]:
        """
        计算Student-t分布的高斯近似矩
        
        Args:
            mu: 空腔均值
            sigma2: 空腔方差
            y: 观测值
            df: 自由度
            
        Returns:
            新均值和方差
        """
        # 使用迭代重加权
        residual = y - mu
        weight = (df + 1) / (df + residual**2 / sigma2)
        
        new_var = sigma2 / weight
        new_mean = mu + weight * residual
        
        return new_mean, new_var
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EPRegressor':
        """
        拟合EP回归器
        
        Args:
            X: 特征
            y: 目标值
            
        Returns:
            self
        """
        X = self._add_bias(X)
        n_samples, n_features = X.shape
        
        # 初始化站点参数
        self.site_means_ = np.zeros(n_samples)
        self.site_vars_ = np.ones(n_samples) * 100  # 大的初始方差
        
        # 先验
        prior_mean = np.zeros(n_features)
        prior_cov = np.eye(n_features) * 100
        
        for iteration in range(self.max_iter):
            old_site_means = self.site_means_.copy()
            old_site_vars = self.site_vars_.copy()
            
            # 对每个数据点进行EP更新
            for i in range(n_samples):
                # 计算空腔分布（标量情况）
                # 使用投影：将多维后验投影到一维
                proj_mean = X[i] @ prior_mean
                proj_var = X[i] @ prior_cov @ X[i]
                
                # 移除站点i的贡献
                cavity_var = 1.0 / (1.0 / proj_var - 1.0 / self.site_vars_[i])
                cavity_mean = cavity_var * (proj_mean / proj_var - self.site_means_[i] / self.site_vars_[i])
                
                # 确保数值稳定
                cavity_var = max(cavity_var, 1e-6)
                
                # 矩匹配
                if self.noise_model == 'laplace':
                    new_mean, new_var = self._laplace_moments(cavity_mean, cavity_var, y[i])
                elif self.noise_model == 'student_t':
                    new_mean, new_var = self._student_t_moments(cavity_mean, cavity_var, y[i])
                else:
                    # 默认高斯
                    new_var = cavity_var / 2
                    new_mean = (cavity_mean + y[i]) / 2
                
                # 更新站点参数（带阻尼）
                new_site_var = 1.0 / (1.0 / new_var - 1.0 / cavity_var)
                new_site_mean = new_site_var * (new_mean / new_var - cavity_mean / cavity_var)
                
                # 确保方差为正
                new_site_var = max(new_site_var, 1e-6)
                
                self.site_vars_[i] = (1 - self.damping) * self.site_vars_[i] + \
                                    self.damping * new_site_var
                self.site_means_[i] = (1 - self.damping) * self.site_means_[i] + \
                                     self.damping * new_site_mean
            
            # 更新后验
            # 这里使用简化版本：假设站点独立
            precision = np.linalg.inv(prior_cov)
            for i in range(n_samples):
                precision += (1.0 / self.site_vars_[i]) * np.outer(X[i], X[i])
            
            self.posterior_cov_ = np.linalg.inv(precision + 1e-6 * np.eye(n_features))
            
            weighted_sum = np.linalg.inv(prior_cov) @ prior_mean
            for i in range(n_samples):
                weighted_sum += (self.site_means_[i] / self.site_vars_[i]) * X[i]
            
            self.posterior_mean_ = self.posterior_cov_ @ weighted_sum
            
            # 检查收敛
            mean_diff = np.mean(np.abs(self.site_means_ - old_site_means))
            var_diff = np.mean(np.abs(self.site_vars_ - old_site_vars))
            
            if mean_diff < self.tol and var_diff < self.tol:
                self.converged_ = True
                break
        
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征
            return_std: 是否返回标准差
            
        Returns:
            预测值（和标准差）
        """
        X = self._add_bias(X)
        
        y_mean = X @ self.posterior_mean_
        
        if return_std:
            y_var = np.sum((X @ self.posterior_cov_) * X, axis=1)
            y_std = np.sqrt(y_var)
            return y_mean, y_std
        
        return y_mean


def demonstrate_ep_classification(show_plot: bool = True) -> None:
    """
    演示EP分类
    """
    print("\n期望传播分类演示")
    print("=" * 60)
    
    # 生成非线性可分数据
    np.random.seed(42)
    n_samples = 200
    
    # 类别1：内圆
    r1 = np.random.uniform(0, 2, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.zeros(n_samples // 2)
    
    # 类别2：外环
    r2 = np.random.uniform(3, 5, n_samples // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.ones(n_samples // 2)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # 添加非线性特征
    X_expanded = np.column_stack([X, X[:, 0]**2, X[:, 1]**2, X[:, 0]*X[:, 1]])
    
    print(f"数据集：{n_samples}个样本，环形分布")
    
    # 训练EP分类器
    ep_clf = EPBinaryClassifier(damping=0.5)
    ep_clf.fit(X_expanded, y)
    
    # 预测
    predictions = ep_clf.predict(X_expanded)
    accuracy = np.mean(predictions == y)
    print(f"训练精度: {accuracy:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始数据
        ax1 = axes[0, 0]
        ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=20, alpha=0.6, label='类0')
        ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=20, alpha=0.6, label='类1')
        ax1.set_title('原始数据')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # EP预测
        ax2 = axes[0, 1]
        ax2.scatter(X[predictions == 0, 0], X[predictions == 0, 1], 
                   c='blue', s=20, alpha=0.6, label='预测0')
        ax2.scatter(X[predictions == 1, 0], X[predictions == 1, 1],
                   c='red', s=20, alpha=0.6, label='预测1')
        ax2.set_title(f'EP预测 (精度={accuracy:.3f})')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 预测概率
        ax3 = axes[0, 2]
        proba = ep_clf.predict_proba(X_expanded)
        scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=proba, cmap='RdBu_r',
                              s=20, alpha=0.6, vmin=0, vmax=1)
        ax3.set_title('预测概率')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        plt.colorbar(scatter3, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # 边缘似然
        ax4 = axes[1, 0]
        ax4.plot(ep_clf.marginal_likelihood_, 'b-', linewidth=2)
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('边缘似然')
        ax4.set_title('收敛曲线')
        ax4.grid(True, alpha=0.3)
        
        # 决策边界
        ax5 = axes[1, 1]
        xx, yy = np.meshgrid(np.linspace(-6, 6, 100),
                            np.linspace(-6, 6, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        X_grid_expanded = np.column_stack([X_grid, X_grid[:, 0]**2,
                                          X_grid[:, 1]**2,
                                          X_grid[:, 0]*X_grid[:, 1]])
        Z = ep_clf.predict_proba(X_grid_expanded).reshape(xx.shape)
        
        ax5.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.3)
        ax5.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        ax5.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=20, alpha=0.6)
        ax5.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=20, alpha=0.6)
        ax5.set_title('决策边界')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        ax5.grid(True, alpha=0.3)
        
        # 不确定性
        ax6 = axes[1, 2]
        # 计算预测不确定性（熵）
        entropy = -proba * np.log(proba + 1e-10) - (1-proba) * np.log(1-proba + 1e-10)
        scatter6 = ax6.scatter(X[:, 0], X[:, 1], c=entropy, cmap='YlOrRd',
                              s=20, alpha=0.6)
        ax6.set_title('预测不确定性（熵）')
        ax6.set_xlabel('特征1')
        ax6.set_ylabel('特征2')
        plt.colorbar(scatter6, ax=ax6)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('期望传播分类', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EP提供了概率预测")
    print("2. 决策边界捕获了非线性模式")
    print("3. 不确定性在决策边界附近最高")


def demonstrate_ep_regression(show_plot: bool = True) -> None:
    """
    演示EP回归（鲁棒回归）
    """
    print("\n期望传播回归演示（鲁棒性）")
    print("=" * 60)
    
    # 生成带异常值的数据
    np.random.seed(42)
    n_samples = 100
    n_outliers = 10
    
    # 正常数据
    X = np.random.uniform(-3, 3, n_samples - n_outliers)
    y = 2 * X + 1 + 0.1 * np.random.randn(n_samples - n_outliers)
    
    # 添加异常值
    X_outliers = np.random.uniform(-3, 3, n_outliers)
    y_outliers = np.random.uniform(-5, 10, n_outliers)
    
    X_all = np.concatenate([X, X_outliers])
    y_all = np.concatenate([y, y_outliers])
    
    # 打乱顺序
    indices = np.random.permutation(n_samples)
    X_all = X_all[indices].reshape(-1, 1)
    y_all = y_all[indices]
    
    print(f"数据集：{n_samples}个样本，{n_outliers}个异常值")
    
    # 训练不同的回归器
    # 1. EP with Laplace噪声（鲁棒）
    ep_laplace = EPRegressor(noise_model='laplace')
    ep_laplace.fit(X_all, y_all)
    
    # 2. EP with Student-t噪声（鲁棒）
    ep_student = EPRegressor(noise_model='student_t')
    ep_student.fit(X_all, y_all)
    
    # 3. 普通最小二乘（对比）
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_all, y_all)
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 排序用于绘图
        sort_idx = np.argsort(X_all.ravel())
        X_sorted = X_all[sort_idx]
        
        # Laplace噪声模型
        ax1 = axes[0, 0]
        y_pred_laplace = ep_laplace.predict(X_sorted)
        ax1.scatter(X_all, y_all, c='gray', s=20, alpha=0.6, label='数据')
        ax1.plot(X_sorted, y_pred_laplace, 'b-', linewidth=2, label='EP (Laplace)')
        ax1.plot(X_sorted, ols.predict(X_sorted), 'r--', linewidth=2, label='OLS')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('EP回归 - Laplace噪声')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Student-t噪声模型
        ax2 = axes[0, 1]
        y_pred_student = ep_student.predict(X_sorted)
        ax2.scatter(X_all, y_all, c='gray', s=20, alpha=0.6, label='数据')
        ax2.plot(X_sorted, y_pred_student, 'g-', linewidth=2, label='EP (Student-t)')
        ax2.plot(X_sorted, ols.predict(X_sorted), 'r--', linewidth=2, label='OLS')
        ax2.set_xlabel('X')
        ax2.set_ylabel('y')
        ax2.set_title('EP回归 - Student-t噪声')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 站点方差（Laplace）
        ax3 = axes[1, 0]
        ax3.scatter(X_all, ep_laplace.site_vars_, c='blue', s=20, alpha=0.6)
        ax3.set_xlabel('X')
        ax3.set_ylabel('站点方差')
        ax3.set_title('站点方差 - Laplace（低方差=异常值）')
        ax3.grid(True, alpha=0.3)
        
        # 预测不确定性
        ax4 = axes[1, 1]
        X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
        y_mean_l, y_std_l = ep_laplace.predict(X_test, return_std=True)
        y_mean_s, y_std_s = ep_student.predict(X_test, return_std=True)
        
        ax4.fill_between(X_test.ravel(), y_mean_l - 2*y_std_l, y_mean_l + 2*y_std_l,
                        alpha=0.3, color='blue', label='EP Laplace')
        ax4.fill_between(X_test.ravel(), y_mean_s - 2*y_std_s, y_mean_s + 2*y_std_s,
                        alpha=0.3, color='green', label='EP Student-t')
        ax4.scatter(X_all, y_all, c='gray', s=20, alpha=0.6)
        ax4.set_xlabel('X')
        ax4.set_ylabel('y')
        ax4.set_title('预测不确定性（95%置信区间）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('期望传播鲁棒回归', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EP with 重尾分布对异常值鲁棒")
    print("2. OLS受异常值影响大")
    print("3. 站点方差自动识别异常值")
    print("4. 提供了预测不确定性估计")