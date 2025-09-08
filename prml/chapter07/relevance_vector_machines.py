"""
7.2 相关向量机 (Relevance Vector Machines)
===========================================

相关向量机是SVM的贝叶斯替代方法。
它提供概率预测，并且通常比SVM更稀疏。

核心思想：
1. 贝叶斯框架：对权重使用稀疏先验
2. 自动相关性确定（ARD）：自动剪枝不相关的基函数
3. 概率输出：提供预测的不确定性

RVM回归模型：
y = Σ_i w_i φ_i(x) + ε
其中ε ~ N(0, β^(-1))

权重的先验（稀疏先验）：
p(w|α) = Π_i N(w_i|0, α_i^(-1))

其中α_i是每个权重的精度参数。
通过最大化边际似然自动确定α_i。

与SVM的比较：
1. RVM提供概率预测
2. RVM通常更稀疏（更少的相关向量）
3. RVM训练更慢（迭代优化）
4. RVM没有需要调节的C参数

缺点：
1. 训练时间长（O(N³)）
2. 不是凸优化（可能陷入局部最优）
3. 对初始化敏感
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from scipy.linalg import solve
from scipy.special import expit  # sigmoid函数
import warnings
warnings.filterwarnings('ignore')


class RVMRegressor:
    """
    相关向量机回归
    
    使用贝叶斯方法和稀疏先验实现自动相关性确定。
    """
    
    def __init__(self, kernel: str = 'rbf',
                 gamma: float = 0.1,
                 alpha_init: float = 1.0,
                 beta_init: float = 1.0,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 prune_threshold: float = 1e10,
                 verbose: bool = False):
        """
        初始化RVM回归器
        
        Args:
            kernel: 核函数类型 ('linear', 'rbf', 'poly')
            gamma: RBF核参数
            alpha_init: 权重精度的初始值
            beta_init: 噪声精度的初始值
            max_iter: 最大迭代次数
            tol: 收敛容差
            prune_threshold: 剪枝阈值（α > threshold时剪枝）
            verbose: 是否打印训练信息
        """
        self.kernel = kernel
        self.gamma = gamma
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.max_iter = max_iter
        self.tol = tol
        self.prune_threshold = prune_threshold
        self.verbose = verbose
        
        # 训练结果
        self.relevance_vectors_ = None  # 相关向量
        self.relevance_indices_ = None  # 相关向量索引
        self.weights_ = None  # 权重（后验均值）
        self.alpha_ = None  # 精度参数
        self.beta_ = None  # 噪声精度
        self.Sigma_ = None  # 权重的后验协方差
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        计算核矩阵
        
        Args:
            X1: 第一组样本, shape (n1, d)
            X2: 第二组样本, shape (n2, d)
            
        Returns:
            核矩阵, shape (n1, n2)
        """
        if self.kernel == 'linear':
            return X1 @ X2.T
        
        elif self.kernel == 'rbf':
            # RBF核
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            distances_sq = X1_norm + X2_norm - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * distances_sq)
        
        elif self.kernel == 'poly':
            return (1 + X1 @ X2.T) ** 3
        
        else:
            raise ValueError(f"未知的核函数: {self.kernel}")
    
    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        构造设计矩阵Φ
        
        每一行是一个样本的基函数值。
        使用所有训练样本作为基函数中心。
        
        Args:
            X: 输入数据, shape (n_samples, n_features)
            
        Returns:
            设计矩阵, shape (n_samples, n_basis)
        """
        # 添加偏置项
        Phi = self._kernel_function(X, self.X_train)
        Phi = np.column_stack([np.ones(len(X)), Phi])
        return Phi
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RVMRegressor':
        """
        训练RVM
        
        使用期望最大化(EM)算法优化超参数。
        
        算法步骤：
        1. E步：给定α和β，计算权重的后验分布
        2. M步：给定权重后验，更新α和β
        
        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练目标, shape (n_samples,)
            
        Returns:
            self
        """
        self.X_train = X
        y = y.ravel()
        n_samples = len(X)
        
        # 构造设计矩阵
        Phi = self._design_matrix(X)  # shape (n, n+1)
        n_basis = Phi.shape[1]
        
        # 初始化超参数
        alpha = np.ones(n_basis) * self.alpha_init  # 每个基函数的精度
        beta = self.beta_init  # 噪声精度
        
        # 记录有效的基函数索引
        active_indices = np.arange(n_basis)
        
        # EM迭代
        for iteration in range(self.max_iter):
            alpha_old = alpha.copy()
            
            # === E步：计算权重的后验分布 ===
            # 后验协方差: Σ = (A + βΦ^T Φ)^(-1)
            # 其中A = diag(α)
            # 注意：alpha已经是active_indices对应的子集
            A = np.diag(alpha)
            Phi_active = Phi[:, active_indices]
            
            # Σ = (A + βΦ^T Φ)^(-1)
            Sigma = np.linalg.inv(A + beta * Phi_active.T @ Phi_active)
            
            # 后验均值: μ = βΣΦ^T y
            mu = beta * Sigma @ Phi_active.T @ y
            
            # === M步：更新超参数 ===
            # 更新α_i = 1 / (μ_i² + Σ_ii)
            # 这实现了自动相关性确定
            gamma = 1 - alpha * np.diag(Sigma)  # 有效自由度
            alpha_new = gamma / (mu ** 2 + 1e-10)  # 添加小值避免除零
            
            # 更新β = (N - Σ_i γ_i) / ||y - Φμ||²
            y_pred = Phi_active @ mu
            residual = y - y_pred
            beta = (n_samples - np.sum(gamma)) / (np.sum(residual ** 2) + 1e-10)
            
            # 剪枝：移除α过大的基函数
            keep_indices = alpha_new < self.prune_threshold
            if not np.all(keep_indices):
                active_indices = active_indices[keep_indices]
                alpha = alpha_new[keep_indices]
                mu = mu[keep_indices]
                Sigma = Sigma[np.ix_(keep_indices, keep_indices)]
                
                if self.verbose:
                    print(f"迭代{iteration}: 剪枝到{len(active_indices)}个基函数")
            else:
                alpha = alpha_new
            
            # 检查收敛
            if len(active_indices) == len(alpha_old):
                alpha_change = np.max(np.abs(alpha - alpha_old))
            else:
                alpha_change = np.inf  # 维度改变，继续迭代
            if alpha_change < self.tol:
                if self.verbose:
                    print(f"在{iteration}次迭代后收敛")
                break
        
        # 保存结果
        self.relevance_indices_ = active_indices[1:] - 1  # 去除偏置项的索引调整
        self.relevance_vectors_ = X[self.relevance_indices_]
        self.weights_ = mu
        self.alpha_ = alpha
        self.beta_ = beta
        self.Sigma_ = Sigma
        self.active_indices_ = active_indices
        
        print(f"RVM训练完成:")
        print(f"  相关向量数: {len(self.relevance_vectors_)}/{n_samples}")
        print(f"  噪声精度β: {self.beta_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, 
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测
        
        提供预测均值和（可选）不确定性。
        
        预测分布：
        p(y*|x*, D) = N(y*|μ*, σ*²)
        其中：
        μ* = φ(x*)^T μ
        σ*² = β^(-1) + φ(x*)^T Σ φ(x*)
        
        Args:
            X: 测试数据, shape (n_samples, n_features)
            return_std: 是否返回标准差
            
        Returns:
            预测均值，或(均值, 标准差)
        """
        # 计算设计矩阵
        Phi = self._design_matrix(X)
        Phi_active = Phi[:, self.active_indices_]
        
        # 预测均值
        y_mean = Phi_active @ self.weights_
        
        if return_std:
            # 预测方差
            # σ*² = β^(-1) + φ^T Σ φ
            variance = 1/self.beta_ + np.sum((Phi_active @ self.Sigma_) * Phi_active, axis=1)
            y_std = np.sqrt(variance)
            return y_mean, y_std
        
        return y_mean
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算R²分数
        
        Args:
            X: 测试数据
            y: 真实目标
            
        Returns:
            R²分数
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class RVMClassifier:
    """
    相关向量机分类
    
    使用Laplace近似处理分类问题。
    """
    
    def __init__(self, kernel: str = 'rbf',
                 gamma: float = 0.1,
                 alpha_init: float = 1.0,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 prune_threshold: float = 1e10,
                 verbose: bool = False):
        """
        初始化RVM分类器
        
        Args:
            kernel: 核函数类型
            gamma: RBF核参数
            alpha_init: 权重精度的初始值
            max_iter: 最大迭代次数
            tol: 收敛容差
            prune_threshold: 剪枝阈值
            verbose: 是否打印训练信息
        """
        self.kernel = kernel
        self.gamma = gamma
        self.alpha_init = alpha_init
        self.max_iter = max_iter
        self.tol = tol
        self.prune_threshold = prune_threshold
        self.verbose = verbose
        
        # 训练结果
        self.relevance_vectors_ = None
        self.relevance_indices_ = None
        self.weights_ = None
        self.alpha_ = None
        self.Sigma_ = None
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            distances_sq = X1_norm + X2_norm - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * distances_sq)
        else:
            raise ValueError(f"未知的核函数: {self.kernel}")
    
    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """构造设计矩阵"""
        Phi = self._kernel_function(X, self.X_train)
        Phi = np.column_stack([np.ones(len(X)), Phi])
        return Phi
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return expit(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RVMClassifier':
        """
        训练RVM分类器
        
        使用Laplace近似和迭代重加权最小二乘(IRLS)。
        
        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签, shape (n_samples,), 值为0或1
            
        Returns:
            self
        """
        self.X_train = X
        y = y.ravel()
        
        # 确保标签是0和1
        unique_labels = np.unique(y)
        if not np.array_equal(sorted(unique_labels), [0, 1]):
            # 转换标签
            y = (y == unique_labels[1]).astype(int)
        
        n_samples = len(X)
        
        # 构造设计矩阵
        Phi = self._design_matrix(X)
        n_basis = Phi.shape[1]
        
        # 初始化
        alpha = np.ones(n_basis) * self.alpha_init
        w = np.zeros(n_basis)  # 权重
        active_indices = np.arange(n_basis)
        
        # 外循环：更新超参数
        for outer_iter in range(self.max_iter):
            alpha_old = alpha.copy()
            
            # 内循环：Laplace近似（IRLS）
            for inner_iter in range(100):
                Phi_active = Phi[:, active_indices]
                w_old = w.copy()
                
                # 计算预测概率
                y_pred = self._sigmoid(Phi_active @ w)
                
                # 计算权重矩阵（Hessian的对角元素）
                R = np.diag(y_pred * (1 - y_pred) + 1e-8)  # 添加小值避免奇异
                
                # 更新权重（Newton-Raphson）
                # w_new = w - H^(-1) g
                # 其中H = -Φ^T R Φ - A, g = Φ^T (y_pred - y) + A w
                A = np.diag(alpha[active_indices])
                H = Phi_active.T @ R @ Phi_active + A
                g = Phi_active.T @ (y_pred - y) + A @ w
                
                try:
                    w = w - solve(H, g)
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，添加正则化
                    H += np.eye(len(H)) * 1e-6
                    w = w - solve(H, g)
                
                # 检查内循环收敛
                if np.max(np.abs(w - w_old)) < 1e-4:
                    break
            
            # 计算后验协方差
            Sigma = np.linalg.inv(H)
            
            # 更新超参数α
            gamma = 1 - alpha[active_indices] * np.diag(Sigma)
            alpha[active_indices] = gamma / (w ** 2 + 1e-8)
            
            # 剪枝
            keep_indices = alpha[active_indices] < self.prune_threshold
            if not np.all(keep_indices):
                active_indices = active_indices[keep_indices]
                alpha = alpha[active_indices]
                w = w[keep_indices]
                Sigma = Sigma[np.ix_(keep_indices, keep_indices)]
                
                if self.verbose:
                    print(f"外循环{outer_iter}: 剪枝到{len(active_indices)}个基函数")
            
            # 检查外循环收敛
            alpha_change = np.max(np.abs(alpha - alpha_old[active_indices]))
            if alpha_change < self.tol:
                if self.verbose:
                    print(f"在{outer_iter}次迭代后收敛")
                break
        
        # 保存结果
        self.relevance_indices_ = active_indices[1:] - 1
        self.relevance_vectors_ = X[self.relevance_indices_]
        self.weights_ = w
        self.alpha_ = alpha
        self.Sigma_ = Sigma
        self.active_indices_ = active_indices
        
        print(f"RVM分类器训练完成:")
        print(f"  相关向量数: {len(self.relevance_vectors_)}/{n_samples}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        使用贝叶斯平均近似预测概率。
        
        Args:
            X: 测试数据, shape (n_samples, n_features)
            
        Returns:
            预测概率, shape (n_samples, 2)
        """
        Phi = self._design_matrix(X)
        Phi_active = Phi[:, self.active_indices_]
        
        # 预测均值
        y_mean = Phi_active @ self.weights_
        
        # 使用probit近似计算预测概率
        # p(y=1|x) ≈ σ(κ(σ²) * μ)
        # 其中κ(σ²) = (1 + πσ²/8)^(-1/2)
        
        # 计算预测方差
        y_var = np.sum((Phi_active @ self.Sigma_) * Phi_active, axis=1)
        
        # Probit近似
        kappa = 1 / np.sqrt(1 + np.pi * y_var / 8)
        prob_1 = self._sigmoid(kappa * y_mean)
        
        # 返回两类概率
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 测试数据
            
        Returns:
            预测标签
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demonstrate_rvm_regression(show_plot: bool = True) -> None:
    """
    演示RVM回归
    
    展示RVM的稀疏性和不确定性量化。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nRVM回归演示")
    print("=" * 60)
    
    # 生成回归数据
    np.random.seed(42)
    n_train = 50
    
    def true_function(x):
        return np.sinc(x)  # sinc(x) = sin(πx)/(πx)
    
    X_train = np.random.uniform(-5, 5, n_train).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + 0.1 * np.random.randn(n_train)
    
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    print(f"训练样本数: {n_train}")
    
    # 训练RVM
    print("\n训练RVM...")
    rvm = RVMRegressor(kernel='rbf', gamma=0.1, verbose=False)
    rvm.fit(X_train, y_train)
    
    # 预测
    y_pred, y_std = rvm.predict(X_test, return_std=True)
    
    # 计算误差
    mse = np.mean((y_pred - y_true) ** 2)
    print(f"测试MSE: {mse:.4f}")
    
    # 稀疏性比例
    sparsity = len(rvm.relevance_vectors_) / n_train
    print(f"稀疏性: {sparsity:.2%}")
    
    if show_plot:
        plt.figure(figsize=(12, 5))
        
        # 子图1：预测结果
        plt.subplot(1, 2, 1)
        plt.plot(X_test, y_true, 'g-', label='真实函数', linewidth=2, alpha=0.5)
        plt.plot(X_test, y_pred, 'r-', label='RVM预测', linewidth=2)
        plt.fill_between(X_test.ravel(),
                        y_pred - 2*y_std,
                        y_pred + 2*y_std,
                        alpha=0.3, color='red',
                        label='95%置信区间')
        plt.scatter(X_train, y_train, c='blue', s=30, 
                   alpha=0.5, label='训练数据')
        plt.scatter(rvm.relevance_vectors_, 
                   true_function(rvm.relevance_vectors_).ravel(),
                   c='red', s=100, marker='^', 
                   edgecolors='black', linewidth=2,
                   label=f'相关向量({len(rvm.relevance_vectors_)}个)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('RVM回归')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：超参数α的值
        plt.subplot(1, 2, 2)
        plt.semilogy(rvm.alpha_, 'o-')
        plt.xlabel('基函数索引')
        plt.ylabel('精度参数α (log scale)')
        plt.title('超参数α的值（大值表示被剪枝）')
        plt.axhline(y=rvm.prune_threshold, color='r', 
                   linestyle='--', label='剪枝阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. RVM自动选择相关向量")
    print("2. 通常比SVM更稀疏")
    print("3. 提供预测不确定性")
    print("4. 不需要交叉验证选择参数")


def demonstrate_rvm_classification(show_plot: bool = True) -> None:
    """
    演示RVM分类
    
    展示RVM分类的稀疏性和概率输出。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nRVM分类演示")
    print("=" * 60)
    
    # 生成二分类数据
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    # 分割数据
    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"训练样本数: {n_train}")
    print(f"测试样本数: {len(X_test)}")
    
    # 训练RVM分类器
    print("\n训练RVM分类器...")
    rvm = RVMClassifier(kernel='rbf', gamma=0.5, verbose=False)
    rvm.fit(X_train, y_train)
    
    # 评估
    train_acc = rvm.score(X_train, y_train)
    test_acc = rvm.score(X_test, y_test)
    
    print(f"训练准确率: {train_acc:.3f}")
    print(f"测试准确率: {test_acc:.3f}")
    
    # 稀疏性
    sparsity = len(rvm.relevance_vectors_) / n_train
    print(f"稀疏性: {sparsity:.2%}")
    
    if show_plot and X.shape[1] == 2:
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # 预测概率
        Z = rvm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12, 5))
        
        # 子图1：决策边界
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(label='P(y=1|x)')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        # 绘制数据点
        scatter = plt.scatter(X_train[:, 0], X_train[:, 1], 
                            c=y_train, cmap='coolwarm',
                            s=50, edgecolors='black', linewidth=1)
        
        # 标记相关向量
        plt.scatter(rvm.relevance_vectors_[:, 0],
                   rvm.relevance_vectors_[:, 1],
                   s=200, linewidth=2, facecolors='none',
                   edgecolors='green', 
                   label=f'相关向量({len(rvm.relevance_vectors_)}个)')
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('RVM分类 - 决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：预测不确定性
        plt.subplot(1, 2, 2)
        
        # 计算预测的熵（不确定性度量）
        proba = rvm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1)
        entropy = entropy.reshape(xx.shape)
        
        plt.contourf(xx, yy, entropy, levels=20, cmap='viridis')
        plt.colorbar(label='预测熵')
        
        # 绘制数据点
        plt.scatter(X_train[:, 0], X_train[:, 1],
                   c=y_train, cmap='coolwarm',
                   s=50, edgecolors='black', linewidth=1)
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('RVM分类 - 预测不确定性')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. RVM提供概率预测")
    print("2. 决策边界附近不确定性高")
    print("3. 自动选择相关向量")
    print("4. 通常比SVM更稀疏")


def compare_svm_rvm(show_plot: bool = True) -> None:
    """
    比较SVM和RVM
    
    展示两种方法的差异。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nSVM vs RVM比较")
    print("=" * 60)
    
    # 生成回归数据
    np.random.seed(42)
    n_samples = 100
    
    X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(n_samples)
    
    # 分割数据
    n_train = 70
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"训练样本数: {n_train}")
    print(f"测试样本数: {len(X_test)}")
    
    # 训练SVM (使用sklearn)
    print("\n训练SVM...")
    from sklearn.svm import SVR
    svm = SVR(kernel='rbf', C=10.0, gamma=0.1)
    svm.fit(X_train, y_train)
    
    # 训练RVM
    print("训练RVM...")
    rvm = RVMRegressor(kernel='rbf', gamma=0.1)
    rvm.fit(X_train, y_train)
    
    # 比较结果
    print("\n性能比较:")
    print("-" * 40)
    
    # SVM
    svm_pred = svm.predict(X_test)
    svm_mse = np.mean((svm_pred - y_test) ** 2)
    n_sv = len(svm.support_)
    
    print(f"SVM:")
    print(f"  支持向量数: {n_sv}/{n_train} ({n_sv/n_train:.1%})")
    print(f"  测试MSE: {svm_mse:.4f}")
    
    # RVM
    rvm_pred = rvm.predict(X_test)
    rvm_mse = np.mean((rvm_pred - y_test) ** 2)
    n_rv = len(rvm.relevance_vectors_)
    
    print(f"\nRVM:")
    print(f"  相关向量数: {n_rv}/{n_train} ({n_rv/n_train:.1%})")
    print(f"  测试MSE: {rvm_mse:.4f}")
    
    if show_plot:
        X_plot = np.linspace(-6, 6, 200).reshape(-1, 1)
        
        plt.figure(figsize=(12, 5))
        
        # SVM预测
        plt.subplot(1, 2, 1)
        svm_pred_plot = svm.predict(X_plot)
        plt.plot(X_plot, np.sin(X_plot), 'g-', 
                label='真实函数', linewidth=2, alpha=0.5)
        plt.plot(X_plot, svm_pred_plot, 'b-', 
                label='SVM预测', linewidth=2)
        plt.scatter(X_train, y_train, c='gray', s=20, alpha=0.3)
        plt.scatter(X_train[svm.support_], y_train[svm.support_],
                   c='blue', s=100, marker='^',
                   label=f'支持向量({n_sv}个)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'SVM (MSE={svm_mse:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RVM预测
        plt.subplot(1, 2, 2)
        rvm_pred_plot, rvm_std = rvm.predict(X_plot, return_std=True)
        plt.plot(X_plot, np.sin(X_plot), 'g-',
                label='真实函数', linewidth=2, alpha=0.5)
        plt.plot(X_plot, rvm_pred_plot, 'r-',
                label='RVM预测', linewidth=2)
        plt.fill_between(X_plot.ravel(),
                        rvm_pred_plot - 2*rvm_std,
                        rvm_pred_plot + 2*rvm_std,
                        alpha=0.3, color='red')
        plt.scatter(X_train, y_train, c='gray', s=20, alpha=0.3)
        plt.scatter(rvm.relevance_vectors_,
                   y_train[rvm.relevance_indices_],
                   c='red', s=100, marker='^',
                   label=f'相关向量({n_rv}个)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'RVM (MSE={rvm_mse:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('SVM vs RVM比较')
        plt.tight_layout()
        plt.show()
    
    print("\n主要区别：")
    print("1. 稀疏性：RVM通常更稀疏")
    print("2. 概率输出：RVM提供不确定性估计")
    print("3. 参数选择：RVM自动优化，SVM需要交叉验证")
    print("4. 训练速度：SVM更快（凸优化）")
    print("5. 理论基础：SVM基于结构风险最小化，RVM基于贝叶斯")