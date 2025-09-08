"""
3.3 贝叶斯线性回归 (Bayesian Linear Regression)
==============================================

频率派方法通过最大似然估计得到点估计w_ML，
贝叶斯方法则维护参数的完整分布p(w|D)。

贝叶斯方法的优势：
1. 不确定性量化：提供预测的置信区间
2. 自动正则化：通过先验避免过拟合
3. 模型选择：通过边际似然比较模型
4. 顺序学习：新数据到来时更新后验

核心思想：
参数不是固定值，而是随机变量。
我们的知识用概率分布表示。

贝叶斯推断的步骤：
1. 选择先验 p(w)
2. 计算似然 p(t|X,w)
3. 应用贝叶斯定理得到后验 p(w|t,X)
4. 做预测 p(t*|x*,t,X)

共轭先验：
当先验和后验属于同一分布族时，称为共轭。
对于高斯似然，高斯先验是共轭的。

数学推导：
先验：w ~ N(m₀, S₀)
似然：t|w ~ N(Φw, β⁻¹I)
后验：w|t ~ N(mₙ, Sₙ)

其中：
Sₙ⁻¹ = S₀⁻¹ + βΦᵀΦ
mₙ = Sₙ(S₀⁻¹m₀ + βΦᵀt)

这些公式展示了：
- 后验精度 = 先验精度 + 数据精度
- 后验均值 = 精度加权的先验和数据信息
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Union
from scipy import stats
from scipy.linalg import inv, cholesky
from .linear_basis_models import BasisFunction, PolynomialBasis, GaussianBasis
import warnings
warnings.filterwarnings('ignore')


class BayesianLinearRegression:
    """
    贝叶斯线性回归
    
    与频率派不同，贝叶斯方法：
    1. 参数w有先验分布
    2. 观测数据后更新为后验分布
    3. 预测时考虑参数不确定性
    
    假设：
    - 先验：w ~ N(m₀, S₀)
    - 噪声：ε ~ N(0, β⁻¹)
    - 模型：t = wᵀφ(x) + ε
    
    关键公式：
    后验分布 p(w|t) ∝ p(t|w)p(w)
    由于共轭性，后验也是高斯分布。
    
    预测分布：
    p(t*|x*) = ∫ p(t*|x*,w)p(w|t)dw
    这个积分有解析解，也是高斯分布。
    """
    
    def __init__(self, basis_function: BasisFunction,
                 alpha: float = 1.0,
                 beta: float = 25.0,
                 m0: Optional[np.ndarray] = None,
                 S0: Optional[np.ndarray] = None):
        """
        初始化贝叶斯线性回归
        
        Args:
            basis_function: 基函数
            alpha: 先验精度（权重的精度）
            beta: 噪声精度（1/σ²）
            m0: 先验均值
            S0: 先验协方差
        """
        self.basis_function = basis_function
        self.alpha = alpha  # 先验精度
        self.beta = beta    # 噪声精度
        
        # 特征维度（包括偏置）
        self.M = basis_function.n_basis + 1
        
        # 设置先验
        if m0 is None:
            # 默认先验均值为0
            self.m0 = np.zeros(self.M)
        else:
            self.m0 = m0
        
        if S0 is None:
            # 默认先验协方差：各向同性
            # S₀ = α⁻¹I
            self.S0 = np.eye(self.M) / alpha
        else:
            self.S0 = S0
        
        # 初始化后验参数（开始时等于先验）
        self.mN = self.m0.copy()
        self.SN = self.S0.copy()
        
        # 存储数据用于计算
        self.X_train = None
        self.y_train = None
        self.Phi_train = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        更新后验分布
        
        贝叶斯更新公式：
        后验 ∝ 似然 × 先验
        
        对于共轭高斯分布：
        Sₙ⁻¹ = S₀⁻¹ + βΦᵀΦ
        mₙ = Sₙ(S₀⁻¹m₀ + βΦᵀt)
        
        这可以理解为：
        - 精度相加（信息累积）
        - 均值是精度加权平均
        
        Args:
            X: 输入数据
            y: 目标值
            
        Returns:
            self
        """
        # 计算设计矩阵
        Phi = self.basis_function(X)
        
        # 存储训练数据
        self.X_train = X
        self.y_train = y
        self.Phi_train = Phi
        
        # 计算后验协方差
        # Sₙ⁻¹ = S₀⁻¹ + βΦᵀΦ
        S0_inv = inv(self.S0)
        SN_inv = S0_inv + self.beta * Phi.T @ Phi
        self.SN = inv(SN_inv)
        
        # 计算后验均值
        # mₙ = Sₙ(S₀⁻¹m₀ + βΦᵀt)
        self.mN = self.SN @ (S0_inv @ self.m0 + self.beta * Phi.T @ y)
        
        return self
    
    def predict(self, X: np.ndarray, 
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测新数据点
        
        预测分布是高斯分布：
        p(t*|x*) = N(t*|mₙᵀφ(x*), σ²(x*))
        
        其中：
        均值：mₙᵀφ(x*)
        方差：σ²(x*) = β⁻¹ + φ(x*)ᵀSₙφ(x*)
        
        方差有两部分：
        1. β⁻¹：观测噪声（不可约）
        2. φᵀSₙφ：参数不确定性（可约）
        
        Args:
            X: 输入数据
            return_std: 是否返回标准差
            
        Returns:
            预测均值，或(均值, 标准差)
        """
        # 计算基函数
        Phi = self.basis_function(X)
        
        # 预测均值
        # E[t*] = mₙᵀφ(x*)
        mean = Phi @ self.mN
        
        if return_std:
            # 预测方差
            # Var[t*] = β⁻¹ + φ(x*)ᵀSₙφ(x*)
            # 第一项：噪声方差
            # 第二项：参数不确定性
            variance = np.zeros(len(X))
            for i in range(len(X)):
                phi = Phi[i]
                # 参数不确定性贡献
                param_var = phi @ self.SN @ phi
                # 总方差
                variance[i] = 1/self.beta + param_var
            
            std = np.sqrt(variance)
            return mean, std
        
        return mean
    
    def sample_functions(self, X: np.ndarray, n_samples: int = 5) -> np.ndarray:
        """
        从后验分布采样函数
        
        展示后验分布代表的函数集合。
        每个采样是一个可能的函数。
        
        步骤：
        1. 从后验 w ~ N(mₙ, Sₙ) 采样权重
        2. 计算 y = wᵀφ(x)
        
        Args:
            X: 输入点
            n_samples: 采样数量
            
        Returns:
            采样的函数值，shape (n_samples, n_points)
        """
        Phi = self.basis_function(X)
        n_points = len(X)
        
        # 从后验采样权重
        # 使用Cholesky分解生成相关高斯样本
        # w = mₙ + L·z, 其中 Sₙ = L·Lᵀ, z ~ N(0,I)
        L = cholesky(self.SN, lower=True)
        
        samples = np.zeros((n_samples, n_points))
        for i in range(n_samples):
            # 采样权重
            z = np.random.randn(self.M)
            w = self.mN + L @ z
            
            # 计算函数值
            samples[i] = Phi @ w
        
        return samples
    
    def marginal_likelihood(self) -> float:
        """
        计算边际似然（证据）
        
        p(t|X) = ∫ p(t|X,w)p(w)dw
        
        这个积分有解析解：
        log p(t|X) = -0.5[N·log(2π) + log|C| + tᵀC⁻¹t]
        
        其中 C = β⁻¹I + ΦS₀Φᵀ
        
        边际似然用于：
        1. 模型比较
        2. 超参数优化
        
        Returns:
            对数边际似然
        """
        if self.Phi_train is None:
            raise ValueError("需要先调用fit方法")
        
        N = len(self.y_train)
        
        # 计算协方差矩阵
        # C = β⁻¹I + ΦS₀Φᵀ
        C = np.eye(N) / self.beta + self.Phi_train @ self.S0 @ self.Phi_train.T
        
        # 计算对数边际似然
        # 使用Cholesky分解提高数值稳定性
        L = cholesky(C, lower=True)
        
        # log|C| = 2·Σlog(L_ii)
        log_det_C = 2 * np.sum(np.log(np.diag(L)))
        
        # tᵀC⁻¹t 通过求解线性系统
        alpha = np.linalg.solve(L, self.y_train)
        quadratic = alpha @ alpha
        
        # 组合各项
        log_marginal = -0.5 * (N * np.log(2*np.pi) + log_det_C + quadratic)
        
        return log_marginal


def demonstrate_bayesian_regression(n_train: int = 5,
                                   basis_type: str = 'polynomial',
                                   n_basis: int = 9,
                                   alpha: float = 2.0,
                                   beta: float = 25.0,
                                   show_plot: bool = True) -> None:
    """
    演示贝叶斯线性回归
    
    展示：
    1. 预测均值和不确定性
    2. 后验采样的函数
    3. 与MLE的对比
    
    Args:
        n_train: 训练样本数
        basis_type: 基函数类型
        n_basis: 基函数数量
        alpha: 先验精度
        beta: 噪声精度
        show_plot: 是否绘图
    """
    print("\n贝叶斯线性回归演示")
    print("=" * 60)
    print(f"训练样本: {n_train}")
    print(f"基函数: {basis_type}, M={n_basis}")
    print(f"先验精度 α={alpha}, 噪声精度 β={beta}")
    print("-" * 60)
    
    # 生成数据
    np.random.seed(42)
    X_train = np.random.uniform(0, 1, n_train)
    noise_std = np.sqrt(1/beta)
    y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, noise_std, n_train)
    
    # 测试数据
    X_test = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * X_test)
    
    # 选择基函数
    if basis_type == 'polynomial':
        basis = PolynomialBasis(n_basis)
    else:
        basis = GaussianBasis(n_basis, 0, 1)
    
    # 贝叶斯回归
    bayesian_model = BayesianLinearRegression(basis, alpha, beta)
    bayesian_model.fit(X_train, y_train)
    
    # 预测
    y_pred, y_std = bayesian_model.predict(X_test, return_std=True)
    
    # 采样函数
    sampled_functions = bayesian_model.sample_functions(X_test, n_samples=5)
    
    # 计算边际似然
    log_evidence = bayesian_model.marginal_likelihood()
    print(f"对数边际似然: {log_evidence:.2f}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：预测分布
        ax1 = axes[0]
        
        # 真实函数
        ax1.plot(X_test, y_true, 'g-', label='真实函数', alpha=0.5)
        
        # 预测均值
        ax1.plot(X_test, y_pred, 'r-', label='预测均值', linewidth=2)
        
        # 置信区间（±2σ）
        ax1.fill_between(X_test, 
                         y_pred - 2*y_std, 
                         y_pred + 2*y_std,
                         alpha=0.3, color='red', 
                         label='95%置信区间')
        
        # 训练数据
        ax1.scatter(X_train, y_train, s=50, c='blue', 
                   zorder=5, label='训练数据')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('预测分布')
        ax1.legend()
        ax1.set_ylim([-2, 2])
        ax1.grid(True, alpha=0.3)
        
        # 右图：后验采样
        ax2 = axes[1]
        
        # 真实函数
        ax2.plot(X_test, y_true, 'g-', label='真实函数', 
                alpha=0.5, linewidth=2)
        
        # 采样的函数
        for i, sample in enumerate(sampled_functions):
            ax2.plot(X_test, sample, '-', alpha=0.5, 
                    label=f'采样{i+1}' if i < 3 else None)
        
        # 训练数据
        ax2.scatter(X_train, y_train, s=50, c='blue', 
                   zorder=5, label='训练数据')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('后验函数采样')
        ax2.legend()
        ax2.set_ylim([-2, 2])
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'贝叶斯线性回归 (N={n_train}, M={n_basis})', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 数据点处不确定性最小")
    print("2. 远离数据的地方不确定性增大")
    print("3. 采样函数都通过训练数据附近")
    print("4. 贝叶斯方法自动提供不确定性量化")


def sequential_bayesian_learning(basis_type: str = 'gaussian',
                                n_basis: int = 9,
                                alpha: float = 2.0,
                                beta: float = 25.0,
                                data_sequence: List[int] = [0, 1, 2, 5],
                                show_plot: bool = True) -> None:
    """
    顺序贝叶斯学习
    
    展示后验如何随数据增加而更新。
    
    贝叶斯学习的顺序性：
    昨天的后验是今天的先验。
    
    这展示了：
    1. 先验的影响逐渐减弱
    2. 不确定性逐渐减少
    3. 预测逐渐接近真实函数
    
    Args:
        basis_type: 基函数类型
        n_basis: 基函数数量
        alpha: 先验精度
        beta: 噪声精度
        data_sequence: 数据点数量序列
        show_plot: 是否绘图
    """
    print("\n顺序贝叶斯学习")
    print("=" * 60)
    print(f"观察数据点序列: {data_sequence}")
    print("-" * 60)
    
    # 生成完整数据集
    np.random.seed(42)
    max_n = max(data_sequence)
    X_all = np.random.uniform(0, 1, max_n)
    noise_std = np.sqrt(1/beta)
    y_all = np.sin(2 * np.pi * X_all) + np.random.normal(0, noise_std, max_n)
    
    # 测试数据
    X_test = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * X_test)
    
    # 基函数
    if basis_type == 'gaussian':
        basis = GaussianBasis(n_basis, 0, 1)
    else:
        basis = PolynomialBasis(n_basis)
    
    if show_plot:
        fig, axes = plt.subplots(2, len(data_sequence), 
                                figsize=(4*len(data_sequence), 8))
        
        for idx, n_data in enumerate(data_sequence):
            # 选择前n个数据点
            if n_data == 0:
                # 只有先验
                model = BayesianLinearRegression(basis, alpha, beta)
                X_train = np.array([])
                y_train = np.array([])
            else:
                X_train = X_all[:n_data]
                y_train = y_all[:n_data]
                
                model = BayesianLinearRegression(basis, alpha, beta)
                model.fit(X_train, y_train)
            
            # 预测
            if n_data == 0:
                # 先验预测
                Phi_test = basis(X_test)
                y_pred = Phi_test @ model.mN
                
                # 先验不确定性
                y_var = np.zeros(len(X_test))
                for i in range(len(X_test)):
                    phi = Phi_test[i]
                    y_var[i] = 1/beta + phi @ model.SN @ phi
                y_std = np.sqrt(y_var)
            else:
                y_pred, y_std = model.predict(X_test, return_std=True)
            
            # 采样函数
            sampled_functions = model.sample_functions(X_test, n_samples=5)
            
            # 上图：预测分布
            ax1 = axes[0, idx] if len(data_sequence) > 1 else axes[0]
            
            # 真实函数
            ax1.plot(X_test, y_true, 'g-', alpha=0.5, linewidth=1)
            
            # 预测均值和置信区间
            ax1.plot(X_test, y_pred, 'r-', linewidth=2)
            ax1.fill_between(X_test, 
                            y_pred - 2*y_std,
                            y_pred + 2*y_std,
                            alpha=0.3, color='red')
            
            # 数据点
            if n_data > 0:
                ax1.scatter(X_train, y_train, s=50, c='blue', zorder=5)
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_title(f'N={n_data}')
            ax1.set_ylim([-2, 2])
            ax1.grid(True, alpha=0.3)
            
            # 下图：采样函数
            ax2 = axes[1, idx] if len(data_sequence) > 1 else axes[1]
            
            # 真实函数
            ax2.plot(X_test, y_true, 'g-', alpha=0.5, linewidth=2)
            
            # 采样函数
            for sample in sampled_functions:
                ax2.plot(X_test, sample, '-', alpha=0.5)
            
            # 数据点
            if n_data > 0:
                ax2.scatter(X_train, y_train, s=50, c='blue', zorder=5)
            
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_title(f'后验采样 (N={n_data})')
            ax2.set_ylim([-2, 2])
            ax2.grid(True, alpha=0.3)
            
            print(f"N={n_data}: ", end="")
            if n_data == 0:
                print("只有先验，高不确定性")
            else:
                log_evidence = model.marginal_likelihood()
                print(f"log p(D)={log_evidence:.2f}")
        
        plt.suptitle('顺序贝叶斯学习', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. N=0：只有先验，不确定性均匀")
    print("2. N增加：不确定性在数据点处减小")
    print("3. 采样函数逐渐收敛到真实函数附近")
    print("4. 后验是先验和数据的折中")


def compare_bayesian_vs_mle(n_train_list: List[int] = [5, 20, 100],
                           n_basis: int = 9,
                           alpha: float = 2.0,
                           beta: float = 25.0,
                           show_plot: bool = True) -> None:
    """
    比较贝叶斯方法和最大似然估计
    
    展示：
    1. 小样本时贝叶斯的优势
    2. 大样本时两者趋于一致
    3. 不确定性量化的重要性
    
    Args:
        n_train_list: 训练样本数列表
        n_basis: 基函数数量
        alpha: 先验精度
        beta: 噪声精度
        show_plot: 是否绘图
    """
    print("\n贝叶斯 vs 最大似然估计")
    print("=" * 60)
    
    from .linear_basis_models import LinearRegression, PolynomialBasis
    
    # 测试数据
    X_test = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * X_test)
    
    basis = PolynomialBasis(n_basis)
    
    results = []
    
    for n_train in n_train_list:
        # 生成训练数据
        np.random.seed(42)
        X_train = np.random.uniform(0, 1, n_train)
        noise_std = np.sqrt(1/beta)
        y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, noise_std, n_train)
        
        # MLE
        mle_model = LinearRegression(basis, regularization=0.0)
        mle_model.fit(X_train, y_train)
        y_mle = mle_model.predict(X_test)
        
        # 贝叶斯
        bayes_model = BayesianLinearRegression(basis, alpha, beta)
        bayes_model.fit(X_train, y_train)
        y_bayes, y_std = bayes_model.predict(X_test, return_std=True)
        
        # 计算误差
        mle_error = np.sqrt(np.mean((y_mle - y_true)**2))
        bayes_error = np.sqrt(np.mean((y_bayes - y_true)**2))
        
        results.append({
            'n_train': n_train,
            'mle_error': mle_error,
            'bayes_error': bayes_error,
            'mle_pred': y_mle,
            'bayes_pred': y_bayes,
            'bayes_std': y_std,
            'X_train': X_train,
            'y_train': y_train
        })
        
        print(f"\nN={n_train}:")
        print(f"  MLE RMSE: {mle_error:.4f}")
        print(f"  Bayes RMSE: {bayes_error:.4f}")
        print(f"  改进: {(mle_error - bayes_error)/mle_error*100:.1f}%")
    
    if show_plot:
        fig, axes = plt.subplots(2, len(n_train_list), 
                                figsize=(5*len(n_train_list), 10))
        
        for idx, result in enumerate(results):
            # 上图：MLE
            ax1 = axes[0, idx] if len(n_train_list) > 1 else axes[0]
            
            ax1.plot(X_test, y_true, 'g-', label='真实', alpha=0.5)
            ax1.plot(X_test, result['mle_pred'], 'b-', 
                    label=f'MLE (RMSE={result["mle_error"]:.3f})', 
                    linewidth=2)
            ax1.scatter(result['X_train'], result['y_train'], 
                       s=50, c='red', zorder=5)
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_title(f'MLE (N={result["n_train"]})')
            ax1.legend()
            ax1.set_ylim([-2, 2])
            ax1.grid(True, alpha=0.3)
            
            # 下图：贝叶斯
            ax2 = axes[1, idx] if len(n_train_list) > 1 else axes[1]
            
            ax2.plot(X_test, y_true, 'g-', label='真实', alpha=0.5)
            ax2.plot(X_test, result['bayes_pred'], 'r-',
                    label=f'Bayes (RMSE={result["bayes_error"]:.3f})',
                    linewidth=2)
            ax2.fill_between(X_test,
                            result['bayes_pred'] - 2*result['bayes_std'],
                            result['bayes_pred'] + 2*result['bayes_std'],
                            alpha=0.3, color='red')
            ax2.scatter(result['X_train'], result['y_train'],
                       s=50, c='blue', zorder=5)
            
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_title(f'贝叶斯 (N={result["n_train"]})')
            ax2.legend()
            ax2.set_ylim([-2, 2])
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'贝叶斯 vs MLE (M={n_basis})', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n关键观察：")
    print("1. 小样本：贝叶斯显著优于MLE（正则化效果）")
    print("2. 大样本：两者性能接近（先验影响减弱）")
    print("3. 贝叶斯提供不确定性量化")
    print("4. MLE容易过拟合，特别是M大N小时")


def demonstrate_evidence_approximation(X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      n_basis: int = 9,
                                      show_plot: bool = True) -> Dict:
    """
    证据近似（经验贝叶斯）
    
    自动确定超参数α和β。
    
    思路：
    最大化边际似然（证据）来选择超参数。
    这避免了交叉验证。
    
    迭代更新公式：
    α_new = γ / (m_N^T m_N)
    β_new = (N - γ) / ||t - Φm_N||²
    
    其中γ = Σᵢ λᵢ/(α + λᵢ) 是有效参数数量。
    
    Args:
        X_train: 训练输入
        y_train: 训练目标
        n_basis: 基函数数量
        show_plot: 是否绘图
        
    Returns:
        优化结果
    """
    print("\n证据近似（自动确定超参数）")
    print("=" * 60)
    
    from scipy.linalg import eigh
    
    # 基函数
    basis = PolynomialBasis(n_basis)
    Phi = basis(X_train)
    N, M = Phi.shape
    
    # 初始化超参数
    alpha = 1.0
    beta = 1.0
    
    # 存储历史
    alpha_history = [alpha]
    beta_history = [beta]
    evidence_history = []
    
    max_iter = 100
    tol = 1e-4
    
    print("迭代优化超参数...")
    print("-" * 40)
    
    for iteration in range(max_iter):
        # E步：计算后验
        # S_N^{-1} = αI + βΦ^TΦ
        SN_inv = alpha * np.eye(M) + beta * Phi.T @ Phi
        SN = inv(SN_inv)
        # m_N = βS_NΦ^Tt
        mN = beta * SN @ Phi.T @ y_train
        
        # 计算有效参数数量γ
        # γ = Σᵢ λᵢ/(α + λᵢ)
        # 其中λᵢ是βΦ^TΦ的特征值
        eigenvalues = eigh(beta * Phi.T @ Phi, eigvals_only=True)
        gamma = np.sum(eigenvalues / (alpha + eigenvalues))
        
        # M步：更新超参数
        # α = γ / ||m_N||²
        alpha_new = gamma / (mN @ mN)
        
        # β = (N - γ) / ||t - Φm_N||²
        residual = y_train - Phi @ mN
        beta_new = (N - gamma) / (residual @ residual)
        
        # 计算边际似然
        model = BayesianLinearRegression(basis, alpha, beta)
        model.fit(X_train, y_train)
        log_evidence = model.marginal_likelihood()
        evidence_history.append(log_evidence)
        
        # 检查收敛
        alpha_change = abs(alpha_new - alpha) / alpha
        beta_change = abs(beta_new - beta) / beta
        
        if iteration % 10 == 0:
            print(f"Iter {iteration:3d}: α={alpha:.4f}, β={beta:.4f}, "
                  f"γ={gamma:.2f}, log p(D)={log_evidence:.2f}")
        
        if alpha_change < tol and beta_change < tol:
            print(f"\n收敛于第{iteration}次迭代")
            break
        
        # 更新
        alpha = alpha_new
        beta = beta_new
        alpha_history.append(alpha)
        beta_history.append(beta)
    
    print("-" * 40)
    print(f"最优超参数: α={alpha:.4f}, β={beta:.4f}")
    print(f"有效参数数: γ={gamma:.2f} (总参数M={M})")
    print(f"噪声标准差估计: σ={1/np.sqrt(beta):.4f}")
    
    # 使用最优超参数的模型
    final_model = BayesianLinearRegression(basis, alpha, beta)
    final_model.fit(X_train, y_train)
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 超参数收敛
        ax1 = axes[0, 0]
        ax1.plot(alpha_history, 'b-', label='α')
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('α')
        ax1.set_title('先验精度α的收敛')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = axes[0, 1]
        ax2.plot(beta_history, 'r-', label='β')
        ax2.set_xlabel('迭代')
        ax2.set_ylabel('β')
        ax2.set_title('噪声精度β的收敛')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 证据
        ax3 = axes[1, 0]
        ax3.plot(evidence_history, 'g-')
        ax3.set_xlabel('迭代')
        ax3.set_ylabel('log p(D)')
        ax3.set_title('对数边际似然')
        ax3.grid(True, alpha=0.3)
        
        # 最终预测
        ax4 = axes[1, 1]
        X_test = np.linspace(0, 1, 100)
        y_true = np.sin(2 * np.pi * X_test)
        y_pred, y_std = final_model.predict(X_test, return_std=True)
        
        ax4.plot(X_test, y_true, 'g-', label='真实', alpha=0.5)
        ax4.plot(X_test, y_pred, 'r-', label='预测', linewidth=2)
        ax4.fill_between(X_test,
                         y_pred - 2*y_std,
                         y_pred + 2*y_std,
                         alpha=0.3, color='red')
        ax4.scatter(X_train, y_train, s=50, c='blue', zorder=5)
        
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_title(f'最优模型 (α={alpha:.3f}, β={beta:.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('证据近似（经验贝叶斯）', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'model': final_model,
        'log_evidence': log_evidence
    }


def demonstrate_model_comparison(X_train: np.ndarray,
                                y_train: np.ndarray,
                                model_orders: List[int] = [1, 3, 5, 7, 9],
                                alpha: float = 2.0,
                                beta: float = 25.0,
                                show_plot: bool = True) -> None:
    """
    贝叶斯模型比较
    
    使用边际似然选择最佳模型复杂度。
    
    边际似然自动实现奥卡姆剃刀：
    - 太简单的模型拟合不好（似然小）
    - 太复杂的模型预测分散（积分小）
    - 最优模型在两者之间平衡
    
    Args:
        X_train: 训练输入
        y_train: 训练目标
        model_orders: 要比较的模型阶数
        alpha: 先验精度
        beta: 噪声精度
        show_plot: 是否绘图
    """
    print("\n贝叶斯模型比较")
    print("=" * 60)
    print(f"比较模型阶数: {model_orders}")
    print("-" * 60)
    
    results = []
    
    for M in model_orders:
        # 创建模型
        basis = PolynomialBasis(M)
        model = BayesianLinearRegression(basis, alpha, beta)
        model.fit(X_train, y_train)
        
        # 计算边际似然
        log_evidence = model.marginal_likelihood()
        
        # 测试误差
        X_test = np.linspace(0, 1, 100)
        y_true = np.sin(2 * np.pi * X_test)
        y_pred = model.predict(X_test)
        test_error = np.sqrt(np.mean((y_pred - y_true)**2))
        
        results.append({
            'M': M,
            'log_evidence': log_evidence,
            'test_error': test_error,
            'model': model
        })
        
        print(f"M={M:2d}: log p(D)={log_evidence:8.2f}, "
              f"测试RMSE={test_error:.4f}")
    
    # 找到最佳模型
    best_idx = np.argmax([r['log_evidence'] for r in results])
    best_M = results[best_idx]['M']
    print(f"\n最佳模型: M={best_M}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 边际似然
        ax1 = axes[0]
        Ms = [r['M'] for r in results]
        evidences = [r['log_evidence'] for r in results]
        ax1.plot(Ms, evidences, 'bo-', linewidth=2, markersize=8)
        ax1.scatter(best_M, results[best_idx]['log_evidence'],
                   color='red', s=200, zorder=5, marker='*')
        ax1.set_xlabel('模型阶数 M')
        ax1.set_ylabel('log p(D|M)')
        ax1.set_title('边际似然（模型证据）')
        ax1.grid(True, alpha=0.3)
        
        # 测试误差
        ax2 = axes[1]
        errors = [r['test_error'] for r in results]
        ax2.plot(Ms, errors, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('模型阶数 M')
        ax2.set_ylabel('测试RMSE')
        ax2.set_title('泛化误差')
        ax2.grid(True, alpha=0.3)
        
        # 最佳模型的预测
        ax3 = axes[2]
        X_test = np.linspace(0, 1, 100)
        y_true = np.sin(2 * np.pi * X_test)
        
        best_model = results[best_idx]['model']
        y_pred, y_std = best_model.predict(X_test, return_std=True)
        
        ax3.plot(X_test, y_true, 'g-', label='真实', alpha=0.5)
        ax3.plot(X_test, y_pred, 'r-', label=f'M={best_M}', linewidth=2)
        ax3.fill_between(X_test,
                         y_pred - 2*y_std,
                         y_pred + 2*y_std,
                         alpha=0.3, color='red')
        ax3.scatter(X_train, y_train, s=50, c='blue', zorder=5)
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_title(f'最佳模型 (M={best_M})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('贝叶斯模型比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 边际似然实现自动模型选择")
    print("2. 不需要验证集或交叉验证")
    print("3. 奥卡姆剃刀：偏好简单但足够的模型")
    print("4. 边际似然平衡拟合优度和模型复杂度")