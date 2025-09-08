"""
12.1 主成分分析 (Principal Component Analysis, PCA)
====================================================

PCA是最基本的线性降维方法，通过找到数据方差最大的方向来提取主成分。

数学原理：
给定数据矩阵X (n×d)，PCA寻找投影矩阵W (d×k)，使得：
Y = XW
其中Y是降维后的数据。

目标函数（最大方差）：
max_W Tr(W^T S W)
s.t. W^T W = I

其中S是数据协方差矩阵。

解：
W的列是S的前k个特征向量（对应最大特征值）。

等价形式：
1. 最大方差：找到投影后方差最大的方向
2. 最小重构误差：min ||X - X_reconstructed||²
3. 最大似然（概率PCA）

概率PCA：
假设生成模型：
x = Wz + μ + ε
其中z ~ N(0, I)，ε ~ N(0, σ²I)

EM算法可用于缺失数据和在线学习。

核PCA：
使用核技巧在特征空间中进行PCA：
K_ij = φ(x_i)^T φ(x_j) = κ(x_i, x_j)

应用：
- 数据压缩
- 特征提取
- 去噪
- 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, svd
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class PCA:
    """
    主成分分析
    
    通过正交变换将数据投影到方差最大的方向。
    """
    
    def __init__(self, n_components: Optional[int] = None,
                 whiten: bool = False,
                 svd_solver: str = 'auto'):
        """
        初始化PCA
        
        Args:
            n_components: 主成分数量（None表示保留所有）
            whiten: 是否白化（使成分具有单位方差）
            svd_solver: SVD求解器 ('auto', 'full', 'randomized')
        """
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        
        # 拟合的参数
        self.mean_ = None
        self.components_ = None  # 主成分（特征向量）
        self.explained_variance_ = None  # 解释方差
        self.explained_variance_ratio_ = None  # 解释方差比例
        self.singular_values_ = None  # 奇异值
        self.n_samples_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型
        
        Args:
            X: 数据矩阵，shape (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 决定成分数量
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = min(self.n_components, n_samples, n_features)
        
        # SVD分解
        if self.svd_solver == 'randomized' and n_components < 0.8 * min(n_samples, n_features):
            # 使用随机SVD（对大数据更快）
            from sklearn.utils.extmath import randomized_svd
            U, S, Vt = randomized_svd(X_centered, n_components)
        else:
            # 完整SVD
            U, S, Vt = svd(X_centered, full_matrices=False)
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components]
        
        # 主成分（右奇异向量）
        self.components_ = Vt
        
        # 奇异值和解释方差
        self.singular_values_ = S
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.sum((X_centered ** 2) / (n_samples - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        # 如果需要白化，存储标准差
        if self.whiten:
            self.whiten_std_ = np.sqrt(self.explained_variance_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到主成分空间
        
        Args:
            X: 数据矩阵
            
        Returns:
            投影后的数据
        """
        # 中心化
        X_centered = X - self.mean_
        
        # 投影
        X_transformed = X_centered @ self.components_.T
        
        # 白化
        if self.whiten:
            X_transformed /= self.whiten_std_
        
        return X_transformed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        从主成分空间重构原始数据
        
        Args:
            X: 降维数据
            
        Returns:
            重构的数据
        """
        # 反白化
        if self.whiten:
            X = X * self.whiten_std_
        
        # 反投影
        X_reconstructed = X @ self.components_ + self.mean_
        
        return X_reconstructed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并转换
        
        Args:
            X: 数据矩阵
            
        Returns:
            转换后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def get_covariance(self) -> np.ndarray:
        """
        获取成分的协方差矩阵
        
        Returns:
            协方差矩阵
        """
        # 从主成分重构协方差
        cov = self.components_.T @ np.diag(self.explained_variance_) @ self.components_
        return cov
    
    def score(self, X: np.ndarray) -> float:
        """
        计算平均对数似然（用于模型选择）
        
        Args:
            X: 数据
            
        Returns:
            平均对数似然
        """
        # 这里使用重构误差的负值作为分数
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        return -mse


class ProbabilisticPCA:
    """
    概率主成分分析
    
    PCA的概率解释，使用EM算法估计参数。
    """
    
    def __init__(self, n_components: int = 2,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        """
        初始化概率PCA
        
        Args:
            n_components: 潜在空间维度
            max_iter: EM迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 模型参数
        self.mean_ = None
        self.W_ = None  # 载荷矩阵
        self.sigma2_ = None  # 噪声方差
        
        # 训练信息
        self.n_iter_ = 0
        self.ll_curve_ = []  # 对数似然曲线
        
    def fit(self, X: np.ndarray) -> 'ProbabilisticPCA':
        """
        使用EM算法拟合模型
        
        Args:
            X: 数据矩阵
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 初始化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 初始化W（使用PCA结果）
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        self.W_ = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # 初始化噪声方差
        self.sigma2_ = np.mean(pca.explained_variance_[self.n_components:]) \
                      if len(pca.explained_variance_) > self.n_components else 0.1
        
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            # E步：计算潜在变量的后验
            # E[z|x] 和 E[zz^T|x]
            M = self.W_.T @ self.W_ + self.sigma2_ * np.eye(self.n_components)
            M_inv = np.linalg.inv(M)
            
            # 对每个数据点
            Ez = X_centered @ self.W_ @ M_inv  # shape: (n_samples, n_components)
            Ezz = self.sigma2_ * M_inv + Ez[:, :, np.newaxis] * Ez[:, np.newaxis, :]
            
            # M步：更新参数
            # 更新W
            sum_xz = X_centered.T @ Ez  # shape: (n_features, n_components)
            sum_zz = np.sum(Ezz, axis=0)  # shape: (n_components, n_components)
            self.W_ = sum_xz @ np.linalg.inv(sum_zz)
            
            # 更新sigma2
            recon_error = 0
            for i in range(n_samples):
                x = X_centered[i]
                z = Ez[i]
                recon_error += np.sum(x ** 2) - 2 * x @ self.W_ @ z + \
                              np.trace(self.W_.T @ self.W_ @ Ezz[i])
            self.sigma2_ = recon_error / (n_samples * n_features)
            
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
        
        # 协方差矩阵 C = WW^T + σ²I
        C = self.W_ @ self.W_.T + self.sigma2_ * np.eye(n_features)
        
        # 多元高斯的对数似然
        sign, logdet = np.linalg.slogdet(C)
        inv_C = np.linalg.inv(C)
        
        ll = -0.5 * n_samples * (n_features * np.log(2 * np.pi) + logdet)
        for x in X_centered:
            ll -= 0.5 * x @ inv_C @ x
        
        return ll
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        推断潜在变量
        
        Args:
            X: 数据
            
        Returns:
            潜在变量的期望
        """
        X_centered = X - self.mean_
        M = self.W_.T @ self.W_ + self.sigma2_ * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)
        Z = X_centered @ self.W_ @ M_inv
        return Z
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        从潜在空间生成数据
        
        Args:
            Z: 潜在变量
            
        Returns:
            生成的数据
        """
        X = Z @ self.W_.T + self.mean_
        # 可以添加噪声：X += np.random.randn(*X.shape) * np.sqrt(self.sigma2_)
        return X
    
    def sample(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        从模型采样
        
        Args:
            n_samples: 样本数量
            
        Returns:
            (X, Z): 生成的数据和潜在变量
        """
        # 采样潜在变量
        Z = np.random.randn(n_samples, self.n_components)
        
        # 生成数据
        X = Z @ self.W_.T + self.mean_
        X += np.random.randn(*X.shape) * np.sqrt(self.sigma2_)
        
        return X, Z


class KernelPCA:
    """
    核主成分分析
    
    在特征空间中进行PCA，可处理非线性关系。
    """
    
    def __init__(self, n_components: int = 2,
                 kernel: str = 'rbf',
                 gamma: Optional[float] = None,
                 degree: int = 3,
                 coef0: float = 1.0):
        """
        初始化核PCA
        
        Args:
            n_components: 主成分数量
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: RBF/poly/sigmoid核的参数
            degree: 多项式核的度数
            coef0: poly/sigmoid核的常数项
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        # 拟合的参数
        self.X_fit_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        
    def _compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算核矩阵
        
        Args:
            X: 第一个数据集
            Y: 第二个数据集（如果None，计算X与自身的核）
            
        Returns:
            核矩阵
        """
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            K = X @ Y.T
        elif self.kernel == 'rbf':
            # RBF核：exp(-γ||x-y||²)
            if self.gamma is None:
                gamma = 1.0 / X.shape[1]
            else:
                gamma = self.gamma
            
            # 计算距离矩阵
            XX = np.sum(X * X, axis=1)[:, np.newaxis]
            YY = np.sum(Y * Y, axis=1)[np.newaxis, :]
            XY = X @ Y.T
            distances = XX + YY - 2 * XY
            K = np.exp(-gamma * distances)
        elif self.kernel == 'poly':
            if self.gamma is None:
                gamma = 1.0 / X.shape[1]
            else:
                gamma = self.gamma
            K = (gamma * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            if self.gamma is None:
                gamma = 1.0 / X.shape[1]
            else:
                gamma = self.gamma
            K = np.tanh(gamma * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"未知核函数: {self.kernel}")
        
        return K
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """
        拟合核PCA
        
        Args:
            X: 训练数据
            
        Returns:
            self
        """
        self.X_fit_ = X.copy()
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = self._compute_kernel(X)
        
        # 中心化核矩阵
        # K_centered = (I - 11^T/n) K (I - 11^T/n)
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # 特征分解
        eigenvalues, eigenvectors = eigh(K_centered)
        
        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 保留前n_components个
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.eigenvectors_ = eigenvectors[:, :self.n_components]
        
        # 归一化特征向量
        for i in range(self.n_components):
            self.eigenvectors_[:, i] /= np.sqrt(eigenvalues[i])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        投影到核主成分
        
        Args:
            X: 数据
            
        Returns:
            投影后的数据
        """
        # 计算与训练数据的核
        K = self._compute_kernel(X, self.X_fit_)
        
        # 中心化
        n_train = self.X_fit_.shape[0]
        n_test = X.shape[0]
        
        K_train = self._compute_kernel(self.X_fit_)
        one_n_train = np.ones((n_train, n_train)) / n_train
        one_n_test = np.ones((n_test, n_train)) / n_train
        
        K_centered = K - one_n_test @ K_train - K @ one_n_train + \
                    one_n_test @ K_train @ one_n_train
        
        # 投影
        X_transformed = K_centered @ self.eigenvectors_
        
        return X_transformed


def demonstrate_pca(show_plot: bool = True) -> None:
    """
    演示PCA
    """
    print("\n主成分分析(PCA)演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    
    # 生成椭圆形数据
    mean = [2, 3]
    cov = [[2, 1.5], [1.5, 1]]
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # 添加第三维（噪声）
    X = np.column_stack([X, np.random.randn(n_samples) * 0.5])
    
    print(f"数据集：{n_samples}个样本，3维")
    
    # 标准PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"\n解释方差比例：")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.3f}")
    print(f"  累积: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # 重构
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\n重构误差(MSE): {reconstruction_error:.4f}")
    
    if show_plot:
        fig = plt.figure(figsize=(15, 10))
        
        # 原始数据（3D）
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', s=20, alpha=0.6)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('X3')
        ax1.set_title('原始数据（3D）')
        
        # 主成分方向
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c='gray', s=10, alpha=0.3)
        
        # 绘制主成分方向
        origin = pca.mean_
        for i in range(2):
            direction = pca.components_[i] * np.sqrt(pca.explained_variance_[i]) * 3
            ax2.quiver(origin[0], origin[1], origin[2],
                      direction[0], direction[1], direction[2],
                      color=['red', 'green'][i], arrow_length_ratio=0.1,
                      linewidth=3, label=f'PC{i+1}')
        
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('X3')
        ax2.set_title('主成分方向')
        ax2.legend()
        
        # PCA投影（2D）
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(X_pca[:, 0], X_pca[:, 1], c='green', s=20, alpha=0.6)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('PCA投影（2D）')
        ax3.grid(True, alpha=0.3)
        
        # 解释方差
        ax4 = fig.add_subplot(2, 3, 4)
        components = range(1, len(pca.explained_variance_ratio_) + 1)
        ax4.bar(components, pca.explained_variance_ratio_, alpha=0.7, color='blue')
        ax4.plot(components, np.cumsum(pca.explained_variance_ratio_),
                'ro-', linewidth=2, markersize=8)
        ax4.set_xlabel('主成分')
        ax4.set_ylabel('解释方差比例')
        ax4.set_title('方差解释')
        ax4.grid(True, alpha=0.3)
        
        # 重构对比
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.scatter(X[:50, 0], X[:50, 1], X[:50, 2],
                   c='blue', s=30, alpha=0.6, label='原始')
        ax5.scatter(X_reconstructed[:50, 0], X_reconstructed[:50, 1],
                   X_reconstructed[:50, 2],
                   c='red', s=30, alpha=0.6, marker='^', label='重构')
        
        # 连线显示重构误差
        for i in range(20):
            ax5.plot([X[i, 0], X_reconstructed[i, 0]],
                    [X[i, 1], X_reconstructed[i, 1]],
                    [X[i, 2], X_reconstructed[i, 2]],
                    'k-', linewidth=0.5, alpha=0.3)
        
        ax5.set_xlabel('X1')
        ax5.set_ylabel('X2')
        ax5.set_zlabel('X3')
        ax5.set_title('重构对比（前50个点）')
        ax5.legend()
        
        # 载荷图
        ax6 = fig.add_subplot(2, 3, 6)
        loadings = pca.components_.T
        features = ['特征1', '特征2', '特征3']
        
        x = np.arange(len(features))
        width = 0.35
        
        ax6.bar(x - width/2, loadings[:, 0], width, label='PC1', alpha=0.7)
        ax6.bar(x + width/2, loadings[:, 1], width, label='PC2', alpha=0.7)
        
        ax6.set_xlabel('特征')
        ax6.set_ylabel('载荷')
        ax6.set_title('主成分载荷')
        ax6.set_xticks(x)
        ax6.set_xticklabels(features)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('主成分分析(PCA)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. PCA找到方差最大的方向")
    print("2. 前两个成分解释了大部分方差")
    print("3. 第三维主要是噪声")
    print("4. 重构误差较小")


def demonstrate_probabilistic_pca(show_plot: bool = True) -> None:
    """
    演示概率PCA
    """
    print("\n概率PCA演示")
    print("=" * 60)
    
    # 生成数据（带缺失值）
    np.random.seed(42)
    n_samples = 200
    
    # 真实的低维结构
    Z_true = np.random.randn(n_samples, 2)
    W_true = np.random.randn(5, 2)
    X_complete = Z_true @ W_true.T + np.random.randn(n_samples, 5) * 0.3
    
    # 添加缺失值
    missing_rate = 0.2
    mask = np.random.rand(n_samples, 5) < missing_rate
    X_missing = X_complete.copy()
    X_missing[mask] = np.nan
    
    print(f"数据集：{n_samples}个样本，5维")
    print(f"缺失率：{missing_rate:.1%}")
    
    # 概率PCA（处理完整数据）
    ppca = ProbabilisticPCA(n_components=2, max_iter=100)
    ppca.fit(X_complete)
    
    print(f"\n噪声方差σ²: {ppca.sigma2_:.4f}")
    print(f"EM迭代次数: {ppca.n_iter_}")
    
    # 生成新样本
    X_generated, Z_generated = ppca.sample(100)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 对数似然曲线
        ax1 = axes[0, 0]
        ax1.plot(ppca.ll_curve_, 'b-', linewidth=2)
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('对数似然')
        ax1.set_title('EM收敛')
        ax1.grid(True, alpha=0.3)
        
        # 潜在空间
        ax2 = axes[0, 1]
        Z_inferred = ppca.transform(X_complete)
        ax2.scatter(Z_true[:, 0], Z_true[:, 1], c='blue', s=20,
                   alpha=0.6, label='真实')
        ax2.scatter(Z_inferred[:, 0], Z_inferred[:, 1], c='red', s=20,
                   alpha=0.6, label='推断')
        ax2.set_xlabel('Z1')
        ax2.set_ylabel('Z2')
        ax2.set_title('潜在空间')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 载荷矩阵
        ax3 = axes[0, 2]
        im = ax3.imshow(ppca.W_, cmap='coolwarm', aspect='auto')
        ax3.set_xlabel('潜在维度')
        ax3.set_ylabel('观测维度')
        ax3.set_title('载荷矩阵W')
        plt.colorbar(im, ax=ax3)
        
        # 生成样本 vs 真实样本
        ax4 = axes[1, 0]
        # 使用前两个维度可视化
        ax4.scatter(X_complete[:, 0], X_complete[:, 1], c='blue',
                   s=20, alpha=0.6, label='真实数据')
        ax4.scatter(X_generated[:, 0], X_generated[:, 1], c='red',
                   s=20, alpha=0.6, label='生成数据')
        ax4.set_xlabel('维度1')
        ax4.set_ylabel('维度2')
        ax4.set_title('数据分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 重构误差分布
        ax5 = axes[1, 1]
        X_reconstructed = ppca.inverse_transform(Z_inferred)
        errors = np.sum((X_complete - X_reconstructed) ** 2, axis=1)
        ax5.hist(errors, bins=30, alpha=0.7, color='green')
        ax5.axvline(x=np.mean(errors), color='red', linestyle='--',
                   linewidth=2, label=f'均值={np.mean(errors):.3f}')
        ax5.set_xlabel('重构误差')
        ax5.set_ylabel('频数')
        ax5.set_title('重构误差分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 协方差结构
        ax6 = axes[1, 2]
        # 模型协方差：C = WW^T + σ²I
        C_model = ppca.W_ @ ppca.W_.T + ppca.sigma2_ * np.eye(5)
        C_empirical = np.cov(X_complete.T)
        
        vmax = max(np.abs(C_model).max(), np.abs(C_empirical).max())
        
        # 显示差异
        diff = C_model - C_empirical
        im = ax6.imshow(diff, cmap='RdBu_r', vmin=-vmax/2, vmax=vmax/2)
        ax6.set_title('协方差差异\n(模型 - 经验)')
        plt.colorbar(im, ax=ax6)
        
        plt.suptitle('概率PCA', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EM算法收敛到最大似然解")
    print("2. 正确推断潜在结构")
    print("3. 生成样本符合数据分布")
    print("4. 提供噪声水平估计")


def demonstrate_kernel_pca(show_plot: bool = True) -> None:
    """
    演示核PCA
    """
    print("\n核PCA演示")
    print("=" * 60)
    
    # 生成非线性数据（瑞士卷）
    np.random.seed(42)
    n_samples = 500
    
    # 生成瑞士卷
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = np.random.randn(n_samples) * 2
    X = np.column_stack([x, y, z])
    
    print(f"数据集：瑞士卷，{n_samples}个样本")
    
    # 标准PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 核PCA（RBF核）
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)
    X_kpca = kpca.fit_transform(X)
    
    if show_plot:
        fig = plt.figure(figsize=(15, 10))
        
        # 原始数据
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t,
                              cmap='viridis', s=20, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('原始数据（瑞士卷）')
        plt.colorbar(scatter1, ax=ax1)
        
        # 线性PCA
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=t,
                              cmap='viridis', s=20, alpha=0.6)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('线性PCA')
        plt.colorbar(scatter2, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # 核PCA
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t,
                              cmap='viridis', s=20, alpha=0.6)
        ax3.set_xlabel('KPC1')
        ax3.set_ylabel('KPC2')
        ax3.set_title('核PCA (RBF)')
        plt.colorbar(scatter3, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # 不同核函数比较
        kernels = ['linear', 'poly', 'sigmoid']
        for i, kernel in enumerate(kernels):
            ax = fig.add_subplot(2, 3, 4 + i)
            
            if kernel == 'poly':
                kpca_temp = KernelPCA(n_components=2, kernel=kernel, degree=3)
            else:
                kpca_temp = KernelPCA(n_components=2, kernel=kernel)
            
            X_temp = kpca_temp.fit_transform(X)
            
            scatter = ax.scatter(X_temp[:, 0], X_temp[:, 1], c=t,
                               cmap='viridis', s=20, alpha=0.6)
            ax.set_xlabel('KPC1')
            ax.set_ylabel('KPC2')
            ax.set_title(f'核PCA ({kernel})')
            plt.colorbar(scatter, ax=ax)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('核PCA比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 线性PCA无法展开瑞士卷")
    print("2. RBF核PCA成功展开非线性结构")
    print("3. 不同核函数效果不同")
    print("4. 核PCA保留了局部结构")