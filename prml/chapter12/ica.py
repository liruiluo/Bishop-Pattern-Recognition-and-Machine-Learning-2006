"""
12.4 独立成分分析 (Independent Component Analysis, ICA)
========================================================

ICA是一种寻找数据中统计独立成分的方法，主要用于盲源分离。

数学模型：
观测信号：x = As
其中s是独立源信号，A是混合矩阵。

目标：
在只知道x的情况下，恢复s和A。

关键假设：
1. 源信号统计独立
2. 源信号非高斯（最多一个高斯）
3. 混合矩阵可逆

ICA vs PCA：
- PCA：去相关（二阶统计量）
- ICA：独立（高阶统计量）
- PCA：正交约束
- ICA：非正交

算法：
1. FastICA：基于负熵最大化
2. Infomax：基于信息最大化
3. JADE：基于四阶累积量

FastICA算法：
1. 中心化和白化数据
2. 选择非线性函数g（如tanh, exp）
3. 固定点迭代：
   w ← E[xg(w^Tx)] - E[g'(w^Tx)]w
   w ← w/||w||

应用：
- 语音分离（鸡尾酒会问题）
- 脑电图(EEG)分析
- 图像处理
- 金融数据分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Optional, Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FastICA:
    """
    FastICA算法
    
    通过最大化非高斯性来寻找独立成分。
    """
    
    def __init__(self, n_components: Optional[int] = None,
                 algorithm: str = 'parallel',
                 whiten: bool = True,
                 fun: str = 'logcosh',
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        初始化FastICA
        
        Args:
            n_components: 独立成分数量
            algorithm: 算法类型 ('parallel', 'deflation')
            whiten: 是否白化预处理
            fun: 非线性函数 ('logcosh', 'exp', 'cube')
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 拟合的参数
        self.components_ = None  # 独立成分（解混矩阵W）
        self.mixing_ = None  # 混合矩阵A
        self.mean_ = None
        self.whitening_ = None  # 白化矩阵
        self.n_iter_ = None
        
    def _get_nonlinearity(self) -> Tuple[Callable, Callable]:
        """
        获取非线性函数及其导数
        
        Returns:
            (g, g_prime): 非线性函数和导数
        """
        if self.fun == 'logcosh':
            # g(u) = tanh(u)
            # g'(u) = 1 - tanh²(u)
            def g(x):
                return np.tanh(x)
            def g_prime(x):
                return 1 - np.tanh(x) ** 2
        elif self.fun == 'exp':
            # g(u) = u * exp(-u²/2)
            # g'(u) = (1 - u²) * exp(-u²/2)
            def g(x):
                return x * np.exp(-x**2 / 2)
            def g_prime(x):
                return (1 - x**2) * np.exp(-x**2 / 2)
        elif self.fun == 'cube':
            # g(u) = u³
            # g'(u) = 3u²
            def g(x):
                return x ** 3
            def g_prime(x):
                return 3 * x ** 2
        else:
            raise ValueError(f"未知的非线性函数: {self.fun}")
        
        return g, g_prime
    
    def _whiten_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        白化数据
        
        白化使得数据具有单位协方差矩阵。
        
        Args:
            X: 中心化数据
            
        Returns:
            (X_white, whitening_matrix): 白化数据和白化矩阵
        """
        # 计算协方差矩阵
        cov = np.cov(X.T)
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 白化矩阵
        # K = D^(-1/2) E^T
        D_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        whitening_matrix = D_sqrt_inv @ eigenvectors.T
        
        # 白化数据
        X_white = X @ whitening_matrix.T
        
        return X_white, whitening_matrix
    
    def _ica_parallel(self, X_white: np.ndarray) -> np.ndarray:
        """
        并行FastICA算法
        
        同时估计所有独立成分。
        
        Args:
            X_white: 白化数据
            
        Returns:
            解混矩阵W
        """
        n_samples, n_features = X_white.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化W为随机正交矩阵
        W = np.random.randn(self.n_components, n_features)
        # Gram-Schmidt正交化
        W = self._orthogonalize(W)
        
        g, g_prime = self._get_nonlinearity()
        
        for iteration in range(self.max_iter):
            W_old = W.copy()
            
            # 对每个成分应用固定点迭代
            for i in range(self.n_components):
                w = W[i]
                
                # 计算w^T x
                wx = X_white @ w
                
                # 固定点更新
                # w ← E[xg(w^Tx)] - E[g'(w^Tx)]w
                w_new = np.mean(X_white * g(wx)[:, np.newaxis], axis=0) - \
                       np.mean(g_prime(wx)) * w
                
                W[i] = w_new
            
            # 正交化W
            W = self._orthogonalize(W)
            
            # 检查收敛
            # 使用W和W_old的内积检查收敛
            convergence = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1))
            
            if convergence < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
            print(f"FastICA未收敛，达到最大迭代次数{self.max_iter}")
        
        return W
    
    def _ica_deflation(self, X_white: np.ndarray) -> np.ndarray:
        """
        逐次提取FastICA算法
        
        逐个估计独立成分。
        
        Args:
            X_white: 白化数据
            
        Returns:
            解混矩阵W
        """
        n_samples, n_features = X_white.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        W = np.zeros((self.n_components, n_features))
        g, g_prime = self._get_nonlinearity()
        
        for i in range(self.n_components):
            # 初始化w为随机向量
            w = np.random.randn(n_features)
            w /= np.linalg.norm(w)
            
            for iteration in range(self.max_iter):
                w_old = w.copy()
                
                # 计算w^T x
                wx = X_white @ w
                
                # 固定点更新
                w = np.mean(X_white * g(wx)[:, np.newaxis], axis=0) - \
                   np.mean(g_prime(wx)) * w
                
                # 去除已提取成分的影响
                w = w - W[:i].T @ (W[:i] @ w)
                
                # 归一化
                w /= np.linalg.norm(w)
                
                # 检查收敛
                if np.abs(np.abs(w @ w_old) - 1) < self.tol:
                    break
            
            W[i] = w
        
        self.n_iter_ = self.n_components  # 简化：不跟踪每个成分的迭代
        return W
    
    def _orthogonalize(self, W: np.ndarray) -> np.ndarray:
        """
        正交化矩阵（对称装饰）
        
        Args:
            W: 矩阵
            
        Returns:
            正交化后的矩阵
        """
        # W ← (WW^T)^(-1/2) W
        U, s, Vt = np.linalg.svd(W @ W.T)
        W_orth = U @ np.diag(1.0 / np.sqrt(s)) @ U.T @ W
        return W_orth
    
    def fit(self, X: np.ndarray) -> 'FastICA':
        """
        拟合ICA模型
        
        Args:
            X: 观测数据，shape (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # 确定成分数量
        if self.n_components is None:
            self.n_components = n_features
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 白化
        if self.whiten:
            X_white, self.whitening_ = self._whiten_data(X_centered)
        else:
            X_white = X_centered
            self.whitening_ = np.eye(n_features)
        
        # 运行ICA算法
        if self.algorithm == 'parallel':
            W = self._ica_parallel(X_white)
        elif self.algorithm == 'deflation':
            W = self._ica_deflation(X_white)
        else:
            raise ValueError(f"未知算法: {self.algorithm}")
        
        # 解混矩阵（在白化空间）
        self.components_ = W
        
        # 混合矩阵 A = (W^+)^T
        # 在原始空间：A_orig = whitening^(-1) @ A_white
        if self.whiten:
            # 从白化空间转换回原始空间
            self.mixing_ = np.linalg.pinv(self.whitening_) @ np.linalg.pinv(W).T
        else:
            self.mixing_ = np.linalg.pinv(W).T
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        提取独立成分
        
        Args:
            X: 观测数据
            
        Returns:
            独立成分S
        """
        # 中心化
        X_centered = X - self.mean_
        
        # 白化
        if self.whiten:
            X_white = X_centered @ self.whitening_.T
        else:
            X_white = X_centered
        
        # 提取独立成分
        S = X_white @ self.components_.T
        
        return S
    
    def inverse_transform(self, S: np.ndarray) -> np.ndarray:
        """
        从独立成分重构信号
        
        Args:
            S: 独立成分
            
        Returns:
            重构的观测信号
        """
        # X = AS + mean
        X = S @ self.mixing_.T + self.mean_
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并转换
        
        Args:
            X: 观测数据
            
        Returns:
            独立成分
        """
        self.fit(X)
        return self.transform(X)


def generate_sources(n_samples: int = 2000, 
                    n_sources: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成独立源信号
    
    Args:
        n_samples: 样本数
        n_sources: 源信号数量
        
    Returns:
        (sources, time): 源信号和时间轴
    """
    time = np.linspace(0, 8, n_samples)
    sources = []
    
    # 源1：正弦波
    s1 = np.sin(2 * time)
    sources.append(s1)
    
    # 源2：方波
    s2 = signal.square(3 * time)
    sources.append(s2)
    
    if n_sources >= 3:
        # 源3：锯齿波
        s3 = signal.sawtooth(2 * np.pi * time)
        sources.append(s3)
    
    if n_sources >= 4:
        # 源4：噪声
        np.random.seed(42)
        s4 = np.random.laplace(size=n_samples)
        s4 = s4 / np.std(s4)  # 标准化
        sources.append(s4)
    
    S = np.array(sources).T
    return S, time


def demonstrate_ica(show_plot: bool = True) -> None:
    """
    演示ICA（鸡尾酒会问题）
    """
    print("\n独立成分分析(ICA)演示 - 鸡尾酒会问题")
    print("=" * 60)
    
    # 生成独立源信号
    n_samples = 2000
    n_sources = 3
    S_true, time = generate_sources(n_samples, n_sources)
    
    print(f"源信号：{n_sources}个独立信号")
    
    # 生成混合矩阵
    np.random.seed(42)
    A_true = np.random.randn(n_sources, n_sources)
    
    # 混合信号
    X = S_true @ A_true.T
    
    print(f"观测信号：{n_sources}个混合信号")
    
    # 应用ICA
    ica = FastICA(n_components=n_sources, random_state=42)
    S_estimated = ica.fit_transform(X)
    
    print(f"\nFastICA收敛：{ica.n_iter_}次迭代")
    
    # 计算分离质量（使用相关系数）
    # 由于ICA的排列和符号不确定性，需要匹配
    correlation_matrix = np.abs(np.corrcoef(S_true.T, S_estimated.T))
    correlation_matrix = correlation_matrix[:n_sources, n_sources:]
    
    print("\n源信号与估计信号的相关系数：")
    for i in range(n_sources):
        max_corr_idx = np.argmax(correlation_matrix[i])
        max_corr = correlation_matrix[i, max_corr_idx]
        print(f"  源{i+1} -> 估计{max_corr_idx+1}: {max_corr:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        
        # 源信号
        for i in range(n_sources):
            ax = axes[0, i]
            ax.plot(time[:500], S_true[:500, i], 'b-', linewidth=1)
            ax.set_title(f'源信号 {i+1}')
            ax.set_xlabel('时间')
            ax.set_ylabel('幅度')
            ax.grid(True, alpha=0.3)
        
        # 混合信号
        for i in range(n_sources):
            ax = axes[1, i]
            ax.plot(time[:500], X[:500, i], 'r-', linewidth=1)
            ax.set_title(f'混合信号 {i+1}')
            ax.set_xlabel('时间')
            ax.set_ylabel('幅度')
            ax.grid(True, alpha=0.3)
        
        # 估计的独立成分
        for i in range(n_sources):
            ax = axes[2, i]
            ax.plot(time[:500], S_estimated[:500, i], 'g-', linewidth=1)
            ax.set_title(f'ICA估计 {i+1}')
            ax.set_xlabel('时间')
            ax.set_ylabel('幅度')
            ax.grid(True, alpha=0.3)
        
        # 混合矩阵和解混矩阵
        ax = axes[3, 0]
        im1 = ax.imshow(A_true, cmap='coolwarm', aspect='auto')
        ax.set_title('真实混合矩阵A')
        ax.set_xlabel('源')
        ax.set_ylabel('观测')
        plt.colorbar(im1, ax=ax)
        
        ax = axes[3, 1]
        im2 = ax.imshow(ica.mixing_, cmap='coolwarm', aspect='auto')
        ax.set_title('估计混合矩阵')
        ax.set_xlabel('源')
        ax.set_ylabel('观测')
        plt.colorbar(im2, ax=ax)
        
        ax = axes[3, 2]
        im3 = ax.imshow(ica.components_, cmap='coolwarm', aspect='auto')
        ax.set_title('解混矩阵W')
        ax.set_xlabel('观测')
        ax.set_ylabel('成分')
        plt.colorbar(im3, ax=ax)
        
        plt.suptitle('独立成分分析(ICA) - 盲源分离', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. ICA成功分离了混合信号")
    print("2. 存在排列和符号不确定性")
    print("3. 非高斯性是关键（高斯信号无法分离）")
    print("4. FastICA收敛快速")


def demonstrate_ica_vs_pca(show_plot: bool = True) -> None:
    """
    比较ICA和PCA
    """
    print("\nICA vs PCA比较")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 1000
    
    # 两个独立的非高斯源
    s1 = np.random.laplace(0, 1, n_samples)
    s2 = np.random.exponential(1, n_samples) - 1
    S = np.column_stack([s1, s2])
    
    # 混合
    angle = np.pi / 6
    A = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    X = S @ A.T
    
    print(f"数据：{n_samples}个样本，2个独立源")
    
    # PCA
    from .pca import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # ICA
    ica = FastICA(n_components=2, random_state=42)
    X_ica = ica.fit_transform(X)
    
    if show_plot:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 独立源
        ax = axes[0, 0]
        ax.scatter(S[:, 0], S[:, 1], s=5, alpha=0.5, c='blue')
        ax.set_xlabel('源1')
        ax.set_ylabel('源2')
        ax.set_title('独立源')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 混合数据
        ax = axes[0, 1]
        ax.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5, c='red')
        ax.set_xlabel('混合1')
        ax.set_ylabel('混合2')
        ax.set_title('观测数据')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # PCA结果
        ax = axes[0, 2]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.5, c='green')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA（去相关）')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # ICA结果
        ax = axes[0, 3]
        ax.scatter(X_ica[:, 0], X_ica[:, 1], s=5, alpha=0.5, c='purple')
        ax.set_xlabel('IC1')
        ax.set_ylabel('IC2')
        ax.set_title('ICA（独立）')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 边缘分布
        for i, (data, title, color) in enumerate([
            (S, '源', 'blue'),
            (X, '混合', 'red'),
            (X_pca, 'PCA', 'green'),
            (X_ica, 'ICA', 'purple')
        ]):
            ax = axes[1, i]
            ax.hist(data[:, 0], bins=30, alpha=0.5, color=color,
                   density=True, label='维度1')
            ax.hist(data[:, 1], bins=30, alpha=0.5, color='orange',
                   density=True, label='维度2')
            ax.set_xlabel('值')
            ax.set_ylabel('密度')
            ax.set_title(f'{title} - 边缘分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('ICA vs PCA：独立 vs 去相关', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 计算相关性和互信息
    print("\n统计量比较：")
    print("方法\t相关系数\t说明")
    print("-" * 40)
    
    for data, name in [(S, '源'), (X, '混合'), (X_pca, 'PCA'), (X_ica, 'ICA')]:
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        print(f"{name}\t{corr:.3f}\t\t", end="")
        if name == '源':
            print("独立非高斯")
        elif name == '混合':
            print("相关")
        elif name == 'PCA':
            print("去相关但不独立")
        else:
            print("独立")
    
    print("\n观察：")
    print("1. PCA只能去相关（二阶统计量）")
    print("2. ICA实现统计独立（高阶统计量）")
    print("3. PCA找最大方差方向")
    print("4. ICA找最大非高斯方向")