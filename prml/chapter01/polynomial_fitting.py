"""
1.1 多项式曲线拟合 (Polynomial Curve Fitting)
==============================================

这是Bishop PRML书中最经典的例子之一。通过这个简单的例子，
我们可以理解机器学习中的许多核心概念：

1. 过拟合 (Overfitting)：当模型过于复杂时，会完美拟合训练数据，
   但在新数据上表现很差。这就像记住了所有考试题的答案，
   但无法解决新问题。

2. 欠拟合 (Underfitting)：当模型过于简单时，连训练数据都无法
   很好地拟合。这就像用直线去拟合明显的曲线数据。

3. 正则化 (Regularization)：通过在损失函数中加入惩罚项，
   限制模型的复杂度，防止过拟合。

4. 模型选择 (Model Selection)：如何选择合适的模型复杂度？
   这需要在偏差和方差之间找到平衡。

让我们通过代码来深入理解这些概念。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from scipy.linalg import lstsq


def generate_sinusoidal_data(
    n_train: int = 10,
    n_test: int = 100,
    noise_std: float = 0.3,
    x_min: float = 0.0,
    x_max: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成用于多项式拟合的正弦函数数据
    
    这个函数生成的数据来自于函数 y = sin(2πx) + ε，
    其中ε是高斯噪声。这是PRML书中的经典例子。
    
    为什么选择正弦函数？
    1. 正弦函数是非线性的，但又不会过于复杂
    2. 它有明确的周期性，容易观察拟合效果
    3. 在[0,1]区间内恰好是一个完整周期
    
    Args:
        n_train: 训练样本数量。较少的训练样本更容易观察到过拟合
        n_test: 测试样本数量。用于评估模型的泛化能力
        noise_std: 噪声的标准差。噪声模拟了真实世界数据的不确定性
        x_min, x_max: x的取值范围
        seed: 随机种子，确保结果可重复
        
    Returns:
        X_train: 训练集输入，shape (n_train, 1)
        y_train: 训练集目标值，shape (n_train,)
        X_test: 测试集输入，shape (n_test, 1)
        y_test: 测试集目标值，shape (n_test,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成训练数据
    # 注意：训练数据是随机采样的，这更接近真实场景
    X_train = np.random.uniform(x_min, x_max, (n_train, 1))
    
    # 真实函数：y = sin(2πx)
    # 这是我们要学习的"真相"，但在真实世界中我们通常不知道这个函数
    y_train_true = np.sin(2 * np.pi * X_train).ravel()
    
    # 添加高斯噪声
    # 噪声代表了测量误差、环境干扰等不可控因素
    noise_train = np.random.normal(0, noise_std, n_train)
    y_train = y_train_true + noise_train
    
    # 生成测试数据
    # 测试数据是均匀分布的，这样可以更好地评估整个函数的拟合效果
    X_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)
    y_test_true = np.sin(2 * np.pi * X_test).ravel()
    
    # 测试数据也添加噪声，模拟真实的测试场景
    noise_test = np.random.normal(0, noise_std, n_test)
    y_test = y_test_true + noise_test
    
    return X_train, y_train, X_test, y_test


class PolynomialCurveFitting:
    """
    多项式曲线拟合类
    
    这个类实现了多项式回归，包括：
    1. 普通最小二乘法
    2. 带正则化的最小二乘法（岭回归）
    
    多项式回归的核心思想：
    将输入x转换为[1, x, x², x³, ..., x^M]的形式，
    然后用线性回归来拟合这些特征。
    """
    
    def __init__(self, order: int, regularization: float = 0.0):
        """
        初始化多项式拟合器
        
        Args:
            order: 多项式的阶数M。M=0是常数，M=1是直线，M=2是抛物线...
            regularization: 正则化系数λ。λ=0表示不使用正则化
        """
        self.order = order
        self.regularization = regularization
        self.weights = None  # 多项式系数，将在fit时计算
        
    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        生成多项式特征
        
        将输入X转换为多项式特征矩阵Φ。
        例如，如果order=3，x=0.5，则转换为[1, 0.5, 0.25, 0.125]
        
        这个转换是多项式回归的关键：
        y = w₀ + w₁x + w₂x² + ... + wₘx^M
        可以写成：y = w^T φ(x)
        其中φ(x) = [1, x, x², ..., x^M]^T
        
        Args:
            X: 输入数据，shape (n_samples, 1)
            
        Returns:
            Φ: 多项式特征矩阵，shape (n_samples, order+1)
        """
        n_samples = X.shape[0]
        # 创建范德蒙德矩阵
        # 每一列是X的不同次幂：[X^0, X^1, X^2, ..., X^order]
        Phi = np.zeros((n_samples, self.order + 1))
        for i in range(self.order + 1):
            Phi[:, i] = X.ravel() ** i
        return Phi
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialCurveFitting':
        """
        拟合多项式模型
        
        使用最小二乘法求解多项式系数。
        
        不带正则化时，最小化：
        E(w) = 1/2 Σ(y_n - w^T φ(x_n))²
        
        带正则化时，最小化：
        E(w) = 1/2 Σ(y_n - w^T φ(x_n))² + λ/2 ||w||²
        
        解析解为：
        w = (Φ^T Φ + λI)^(-1) Φ^T y
        
        Args:
            X: 训练输入，shape (n_samples, 1)
            y: 训练目标，shape (n_samples,)
            
        Returns:
            self: 返回自身，支持链式调用
        """
        # 生成多项式特征
        Phi = self._polynomial_features(X)
        
        # 计算 Φ^T Φ
        # 这是正规方程的核心部分
        gram_matrix = Phi.T @ Phi  # shape: (order+1, order+1)
        
        # 添加正则化项
        # λI 是对角矩阵，对角线上都是λ
        # 注意：通常不对偏置项w₀进行正则化，但这里为了简单起见对所有参数都正则化
        if self.regularization > 0:
            # 添加λI到格拉姆矩阵
            gram_matrix += self.regularization * np.eye(self.order + 1)
        
        # 计算 Φ^T y
        moment_vector = Phi.T @ y  # shape: (order+1,)
        
        # 求解线性方程组：(Φ^T Φ + λI) w = Φ^T y
        # 使用np.linalg.solve比直接求逆更稳定、更高效
        try:
            self.weights = np.linalg.solve(gram_matrix, moment_vector)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘法的备用方法
            print(f"警告：格拉姆矩阵奇异，使用伪逆求解")
            self.weights = np.linalg.pinv(gram_matrix) @ moment_vector
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合的多项式进行预测
        
        Args:
            X: 输入数据，shape (n_samples, 1)
            
        Returns:
            y_pred: 预测值，shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("模型还未拟合，请先调用fit方法")
        
        Phi = self._polynomial_features(X)
        # y = Φw
        return Phi @ self.weights
    
    def compute_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算均方根误差 (RMSE)
        
        RMSE = sqrt(1/N Σ(y_n - y_pred_n)²)
        
        这是评估回归模型的常用指标，单位与y相同，
        因此比均方误差(MSE)更容易解释。
        
        Args:
            X: 输入数据
            y: 真实目标值
            
        Returns:
            RMSE值
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return np.sqrt(mse)


def plot_polynomial_fits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    orders: List[int],
    regularization: float = 0.0,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    绘制不同阶数多项式的拟合效果
    
    这个函数展示了模型复杂度如何影响拟合效果：
    - 低阶多项式（如M=0,1）通常欠拟合
    - 高阶多项式（如M=9）在小数据集上容易过拟合
    - 中等阶数（如M=3）可能达到好的平衡
    
    Args:
        X_train: 训练输入
        y_train: 训练目标
        X_test: 用于绘制拟合曲线的测试输入
        orders: 要测试的多项式阶数列表
        regularization: 正则化系数
        figsize: 图片尺寸
    """
    n_orders = len(orders)
    fig, axes = plt.subplots(1, n_orders, figsize=figsize)
    
    if n_orders == 1:
        axes = [axes]
    
    # 用于绘制的x值（更密集，使曲线更平滑）
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    
    for idx, (order, ax) in enumerate(zip(orders, axes)):
        # 拟合多项式
        model = PolynomialCurveFitting(order, regularization)
        model.fit(X_train, y_train)
        
        # 预测
        y_plot = model.predict(x_plot)
        
        # 绘制真实函数
        y_true = np.sin(2 * np.pi * x_plot).ravel()
        ax.plot(x_plot, y_true, 'g-', label='真实函数', alpha=0.7, linewidth=2)
        
        # 绘制拟合曲线
        ax.plot(x_plot, y_plot, 'r-', label=f'M={order}', linewidth=2)
        
        # 绘制训练数据点
        ax.scatter(X_train, y_train, s=40, c='blue', 
                  edgecolors='black', label='训练数据', zorder=5)
        
        # 计算训练和测试误差
        train_error = model.compute_error(X_train, y_train)
        
        # 设置图形属性
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'M={order}, RMSE={train_error:.3f}')
        ax.set_ylim([-1.5, 1.5])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.suptitle(f'多项式拟合效果对比 (λ={regularization})', fontsize=14)
    plt.tight_layout()
    plt.show()


def demonstrate_overfitting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    orders: List[int] = [0, 1, 3, 9],
    show_plot: bool = True
) -> None:
    """
    演示过拟合现象
    
    这个函数展示了经典的过拟合现象：
    1. 随着模型复杂度增加，训练误差持续下降
    2. 但测试误差先下降后上升，形成U型曲线
    3. 最优模型复杂度在U型曲线的最低点
    
    这就是偏差-方差权衡(Bias-Variance Tradeoff)的体现：
    - 简单模型：高偏差，低方差
    - 复杂模型：低偏差，高方差
    """
    print("\n演示过拟合现象：")
    print("-" * 50)
    print(f"{'多项式阶数':<12} {'训练误差':<12} {'测试误差':<12} {'状态'}")
    print("-" * 50)
    
    train_errors = []
    test_errors = []
    
    for order in orders:
        # 拟合模型
        model = PolynomialCurveFitting(order, regularization=0.0)
        model.fit(X_train, y_train)
        
        # 计算误差
        train_error = model.compute_error(X_train, y_train)
        test_error = model.compute_error(X_test, y_test)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # 判断状态
        if order <= 1:
            status = "欠拟合"
        elif order >= 9 and len(X_train) <= 10:
            status = "过拟合"
        else:
            status = "适中"
        
        print(f"M={order:<10} {train_error:<12.4f} {test_error:<12.4f} {status}")
    
    if show_plot:
        plot_polynomial_fits(X_train, y_train, X_test, orders)
        
        # 绘制误差曲线
        plt.figure(figsize=(8, 5))
        plt.plot(orders, train_errors, 'b-o', label='训练误差', linewidth=2)
        plt.plot(orders, test_errors, 'r-o', label='测试误差', linewidth=2)
        plt.xlabel('多项式阶数 M')
        plt.ylabel('RMSE')
        plt.title('训练误差 vs 测试误差：过拟合的特征')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def demonstrate_regularization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    order: int = 9,
    lambda_values: List[float] = [0, 1e-18, 1e-4, 1],
    show_plot: bool = True
) -> None:
    """
    演示正则化的效果
    
    正则化通过在损失函数中添加惩罚项来控制模型复杂度：
    E(w) = E_D(w) + λE_W(w)
    
    其中：
    - E_D(w) 是数据拟合项（最小化预测误差）
    - E_W(w) 是正则化项（限制权重大小）
    - λ 控制两者的平衡
    
    λ的影响：
    - λ=0：无正则化，可能过拟合
    - λ很小：轻微正则化，减少过拟合
    - λ适中：好的偏差-方差平衡
    - λ很大：强正则化，可能欠拟合
    """
    print(f"\n正则化对{order}阶多项式的影响：")
    print("-" * 60)
    print(f"{'λ值':<15} {'训练误差':<12} {'测试误差':<12} {'权重范数':<12}")
    print("-" * 60)
    
    if show_plot:
        fig, axes = plt.subplots(1, len(lambda_values), figsize=(15, 5))
        if len(lambda_values) == 1:
            axes = [axes]
    
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    
    for idx, (lambda_val, ax) in enumerate(zip(lambda_values, axes if show_plot else [None]*len(lambda_values))):
        # 拟合模型
        model = PolynomialCurveFitting(order, regularization=lambda_val)
        model.fit(X_train, y_train)
        
        # 计算误差
        train_error = model.compute_error(X_train, y_train)
        test_error = model.compute_error(X_test, y_test)
        
        # 计算权重的L2范数
        weight_norm = np.linalg.norm(model.weights)
        
        # 格式化λ值的显示
        if lambda_val == 0:
            lambda_str = "0"
        elif lambda_val < 1e-10:
            lambda_str = f"{lambda_val:.0e}"
        else:
            lambda_str = f"{lambda_val:.4f}"
        
        print(f"λ={lambda_str:<13} {train_error:<12.4f} {test_error:<12.4f} {weight_norm:<12.4f}")
        
        if show_plot and ax is not None:
            # 预测
            y_plot = model.predict(x_plot)
            
            # 绘制真实函数
            y_true = np.sin(2 * np.pi * x_plot).ravel()
            ax.plot(x_plot, y_true, 'g-', label='真实函数', alpha=0.7, linewidth=2)
            
            # 绘制拟合曲线
            ax.plot(x_plot, y_plot, 'r-', label=f'拟合(M={order})', linewidth=2)
            
            # 绘制训练数据
            ax.scatter(X_train, y_train, s=40, c='blue',
                      edgecolors='black', label='训练数据', zorder=5)
            
            # 设置图形属性
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'λ={lambda_str}')
            ax.set_ylim([-1.5, 1.5])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
    
    if show_plot:
        plt.suptitle(f'正则化效果：{order}阶多项式在不同λ值下的表现', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("- λ=0时，模型可能过拟合（训练误差小但测试误差大）")
    print("- 适当的λ值可以改善测试误差")
    print("- λ过大时，模型变得过于简单（欠拟合）")
    print("- 权重范数随λ增大而减小，说明正则化在限制权重大小")