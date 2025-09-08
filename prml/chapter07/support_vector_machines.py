"""
7.1 支持向量机 (Support Vector Machines)
=========================================

支持向量机是最成功的机器学习算法之一。
它基于统计学习理论，具有坚实的理论基础。

核心思想：
1. 最大间隔原理：寻找具有最大间隔的决策边界
2. 核技巧：通过核函数隐式地在高维空间中工作
3. 稀疏性：只有支持向量对决策函数有贡献

线性SVM的原始问题：
min_{w,b} (1/2)||w||² 
s.t. y_i(w^T φ(x_i) + b) ≥ 1, ∀i

对偶问题：
max_α Σ_i α_i - (1/2)Σ_{i,j} α_i α_j y_i y_j k(x_i, x_j)
s.t. 0 ≤ α_i ≤ C, Σ_i α_i y_i = 0

其中：
- α_i：拉格朗日乘子
- C：软间隔参数（控制错分样本的惩罚）
- k(x_i, x_j)：核函数

决策函数：
f(x) = Σ_i α_i y_i k(x_i, x) + b

KKT条件：
- α_i = 0 → y_i f(x_i) > 1（正确分类，非支持向量）
- 0 < α_i < C → y_i f(x_i) = 1（在边界上的支持向量）  
- α_i = C → y_i f(x_i) ≤ 1（错分或在边界内的样本）

优点：
1. 全局最优解（凸优化问题）
2. 泛化能力强（最大间隔原理）
3. 稀疏解（只依赖支持向量）
4. 可处理高维数据

缺点：
1. 对大规模数据计算复杂度高
2. 对核函数和参数敏感
3. 不直接提供概率输出
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union, Callable
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')


class SVM(BaseEstimator, ClassifierMixin):
    """
    支持向量机分类器
    
    实现了软间隔SVM的训练和预测。
    使用凸优化求解对偶问题。
    """
    
    def __init__(self, C: float = 1.0, 
                 kernel: str = 'linear',
                 gamma: float = 0.1,
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 max_iter: int = 1000):
        """
        初始化SVM
        
        Args:
            C: 软间隔参数，控制错分样本的惩罚
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: RBF/poly/sigmoid核的参数
            degree: 多项式核的次数
            coef0: poly/sigmoid核的常数项
            tol: 收敛容差
            max_iter: 最大迭代次数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        # 训练结果
        self.alpha = None  # 拉格朗日乘子
        self.b = None  # 偏置
        self.support_vectors_ = None  # 支持向量
        self.support_vector_indices_ = None  # 支持向量索引
        self.X_train = None
        self.y_train = None
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        计算核函数
        
        Args:
            X1: 第一组样本, shape (n1, d)
            X2: 第二组样本, shape (n2, d)
            
        Returns:
            核矩阵, shape (n1, n2)
        """
        if self.kernel == 'linear':
            # 线性核: k(x, y) = x^T y
            return X1 @ X2.T
        
        elif self.kernel == 'rbf':
            # RBF核: k(x, y) = exp(-γ||x-y||²)
            # 计算欧氏距离的平方
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)  # shape (n1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)  # shape (1, n2)
            distances_sq = X1_norm + X2_norm - 2 * (X1 @ X2.T)  # shape (n1, n2)
            return np.exp(-self.gamma * distances_sq)
        
        elif self.kernel == 'poly':
            # 多项式核: k(x, y) = (γx^T y + r)^d
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'sigmoid':
            # Sigmoid核: k(x, y) = tanh(γx^T y + r)
            return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        
        else:
            raise ValueError(f"未知的核函数: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        训练SVM
        
        通过求解对偶问题得到支持向量和决策函数。
        
        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签, shape (n_samples,), 值为-1或1
            
        Returns:
            self
        """
        n_samples = len(X)
        self.X_train = X
        self.y_train = y
        
        # 确保标签是-1和1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM只支持二分类")
        if not np.array_equal(sorted(unique_labels), [-1, 1]):
            # 转换标签到-1和1
            self.y_train = np.where(y == unique_labels[0], -1, 1)
        
        # 计算核矩阵
        K = self._kernel_function(X, X)  # shape (n, n)
        
        # 使用cvxpy求解对偶问题
        # max_α Σ_i α_i - (1/2)Σ_{i,j} α_i α_j y_i y_j K_{ij}
        # s.t. 0 ≤ α_i ≤ C, Σ_i α_i y_i = 0
        
        # 使用scipy的optimize代替cvxpy，避免DCP问题
        from scipy.optimize import minimize as scipy_minimize
        
        # 构造目标函数
        # Q_{ij} = y_i y_j K_{ij}
        Q = np.outer(self.y_train, self.y_train) * K  # shape (n, n)
        
        # 添加正则化保证数值稳定
        Q = Q + np.eye(n_samples) * 1e-5
        
        # 定义目标函数和梯度
        def objective_func(alpha):
            # 最小化: (1/2)α^T Q α - Σα_i
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)
        
        def gradient(alpha):
            return Q @ alpha - np.ones(n_samples)
        
        # 定义约束
        constraints = [
            # Σα_i y_i = 0
            {'type': 'eq', 'fun': lambda a: np.dot(a, self.y_train)}
        ]
        
        # 定义边界 0 <= α_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # 初始猜测
        alpha0 = np.zeros(n_samples)
        
        # 求解
        result = scipy_minimize(
            objective_func,
            alpha0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 200}
        )
        
        if not result.success:
            print(f"警告：优化未完全收敛: {result.message}")
        
        # 保存拉格朗日乘子
        self.alpha = result.x
        
        # 找出支持向量（α > tol）
        sv_indices = self.alpha > self.tol
        self.support_vector_indices_ = np.where(sv_indices)[0]
        self.support_vectors_ = X[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]
        self.y_sv = self.y_train[sv_indices]
        
        print(f"找到 {len(self.support_vectors_)} 个支持向量")
        
        # 计算偏置b
        # 使用在边界上的支持向量（0 < α < C）计算b
        margin_sv_indices = np.logical_and(
            self.alpha > self.tol,
            self.alpha < self.C - self.tol
        )
        
        if np.any(margin_sv_indices):
            # 对于边界上的支持向量，y_i(Σα_j y_j K_{ij} + b) = 1
            # 所以 b = y_i - Σα_j y_j K_{ij}
            margin_indices = np.where(margin_sv_indices)[0]
            
            # 计算每个边界支持向量对应的b
            b_values = []
            for idx in margin_indices:
                # 计算 Σα_j y_j K_{ij}
                kernel_sum = np.sum(
                    self.alpha[sv_indices] * 
                    self.y_train[sv_indices] * 
                    K[idx, sv_indices]
                )
                b_values.append(self.y_train[idx] - kernel_sum)
            
            # 取平均值作为最终的b
            self.b = np.mean(b_values)
        else:
            # 如果没有边界支持向量，使用所有支持向量的平均
            self.b = np.mean(
                self.y_train[sv_indices] - 
                K[sv_indices][:, sv_indices] @ (self.alpha_sv * self.y_sv)
            )
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算决策函数值
        
        f(x) = Σ_i α_i y_i k(x_i, x) + b
        
        Args:
            X: 输入数据, shape (n_samples, n_features)
            
        Returns:
            决策函数值, shape (n_samples,)
        """
        # 计算测试样本与支持向量的核矩阵
        K = self._kernel_function(X, self.support_vectors_)  # shape (n_test, n_sv)
        
        # 计算决策函数
        decision = K @ (self.alpha_sv * self.y_sv) + self.b  # shape (n_test,)
        
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据, shape (n_samples, n_features)
            
        Returns:
            预测标签, shape (n_samples,)
        """
        return np.sign(self.decision_function(X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算分类准确率
        
        Args:
            X: 测试数据
            y: 真实标签
            
        Returns:
            准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class SimplifiedSMO:
    """
    简化版SMO算法
    
    Sequential Minimal Optimization是求解SVM的高效算法。
    每次选择两个变量进行优化，其他变量固定。
    
    选择策略：
    1. 外循环：选择违反KKT条件的α_i
    2. 内循环：选择使目标函数增长最大的α_j
    """
    
    def __init__(self, C: float = 1.0, 
                 kernel: str = 'rbf',
                 gamma: float = 0.1,
                 tol: float = 1e-3,
                 max_iter: int = 100):
        """
        初始化SMO算法
        
        Args:
            C: 软间隔参数
            kernel: 核函数类型
            gamma: RBF核参数
            tol: 容差
            max_iter: 最大迭代次数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """计算两个样本的核函数值"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError(f"不支持的核函数: {self.kernel}")
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._kernel_function(X[i], X[j])
                K[j, i] = K[i, j]
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimplifiedSMO':
        """
        使用简化SMO算法训练SVM
        
        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签, shape (n_samples,), 值为-1或1
            
        Returns:
            self
        """
        n_samples = len(X)
        self.X = X
        self.y = y
        
        # 初始化
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # 计算核矩阵
        self.K = self._compute_kernel_matrix(X)
        
        # SMO主循环
        num_changed = 0
        examine_all = True
        iter_count = 0
        
        while (num_changed > 0 or examine_all) and iter_count < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # 遍历所有样本
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # 只遍历非边界样本（0 < α < C）
                non_bound = np.logical_and(
                    self.alpha > self.tol,
                    self.alpha < self.C - self.tol
                )
                for i in np.where(non_bound)[0]:
                    num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iter_count += 1
        
        # 找出支持向量
        sv_indices = self.alpha > self.tol
        self.support_vectors_ = X[sv_indices]
        self.support_vector_indices_ = np.where(sv_indices)[0]
        
        print(f"SMO算法收敛，迭代{iter_count}次")
        print(f"找到{len(self.support_vectors_)}个支持向量")
        
        return self
    
    def _examine_example(self, i2: int) -> int:
        """
        检查并优化一个样本
        
        Args:
            i2: 要检查的样本索引
            
        Returns:
            是否进行了优化（0或1）
        """
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self._compute_error(i2)
        r2 = E2 * y2
        
        # 检查KKT条件
        if ((r2 < -self.tol and alpha2 < self.C) or 
            (r2 > self.tol and alpha2 > 0)):
            
            # 选择第二个变量
            i1 = self._select_second_alpha(i2, E2)
            if i1 >= 0:
                if self._take_step(i1, i2):
                    return 1
            
            # 如果上面的选择失败，尝试其他非边界样本
            non_bound = np.logical_and(
                self.alpha > self.tol,
                self.alpha < self.C - self.tol
            )
            for i1 in np.random.permutation(np.where(non_bound)[0]):
                if self._take_step(i1, i2):
                    return 1
            
            # 如果还是失败，尝试所有样本
            for i1 in np.random.permutation(len(self.X)):
                if self._take_step(i1, i2):
                    return 1
        
        return 0
    
    def _select_second_alpha(self, i2: int, E2: float) -> int:
        """
        选择第二个要优化的变量
        
        启发式：选择使|E1 - E2|最大的α1
        
        Args:
            i2: 第一个变量的索引
            E2: 第一个变量的误差
            
        Returns:
            第二个变量的索引
        """
        non_zero_alpha = self.alpha > self.tol
        if np.sum(non_zero_alpha) > 1:
            # 计算所有非零α的误差
            errors = np.array([self._compute_error(i) 
                              for i in range(len(self.X))])
            
            # 选择使|E1 - E2|最大的
            if E2 > 0:
                i1 = np.argmin(errors)
            else:
                i1 = np.argmax(errors)
            
            return i1
        
        return -1
    
    def _take_step(self, i1: int, i2: int) -> bool:
        """
        优化两个拉格朗日乘子
        
        Args:
            i1: 第一个变量索引
            i2: 第二个变量索引
            
        Returns:
            是否成功优化
        """
        if i1 == i2:
            return False
        
        alpha1_old = self.alpha[i1]
        alpha2_old = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        s = y1 * y2
        
        # 计算α2的上下界
        if s < 0:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha2_old + alpha1_old - self.C)
            H = min(self.C, alpha2_old + alpha1_old)
        
        if L == H:
            return False
        
        # 计算η（二阶导数）
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            # 计算新的α2
            alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
            
            # 裁剪到[L, H]
            if alpha2_new < L:
                alpha2_new = L
            elif alpha2_new > H:
                alpha2_new = H
        else:
            # η <= 0的特殊情况（罕见）
            return False
        
        # 检查变化是否足够大
        if abs(alpha2_new - alpha2_old) < self.tol * (alpha2_new + alpha2_old + self.tol):
            return False
        
        # 计算新的α1
        alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new)
        
        # 更新偏置b
        b1 = E1 + y1 * (alpha1_new - alpha1_old) * k11 + \
             y2 * (alpha2_new - alpha2_old) * k12 + self.b
        b2 = E2 + y1 * (alpha1_new - alpha1_old) * k12 + \
             y2 * (alpha2_new - alpha2_old) * k22 + self.b
        
        if 0 < alpha1_new < self.C:
            self.b = -b1
        elif 0 < alpha2_new < self.C:
            self.b = -b2
        else:
            self.b = -(b1 + b2) / 2
        
        # 更新α
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        
        return True
    
    def _compute_error(self, i: int) -> float:
        """
        计算第i个样本的误差
        
        E_i = f(x_i) - y_i
        
        Args:
            i: 样本索引
            
        Returns:
            误差值
        """
        # f(x_i) = Σ_j α_j y_j K_{ij} + b
        f_xi = np.sum(self.alpha * self.y * self.K[i, :]) + self.b
        return f_xi - self.y[i]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值"""
        n_test = len(X)
        decision = np.zeros(n_test)
        
        for i in range(n_test):
            for j in self.support_vector_indices_:
                decision[i] += self.alpha[j] * self.y[j] * \
                              self._kernel_function(X[i], self.X[j])
            decision[i] += self.b
        
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return np.sign(self.decision_function(X))


def visualize_svm_decision_boundary(svm_model, X: np.ndarray, y: np.ndarray,
                                   title: str = "SVM决策边界",
                                   show_plot: bool = True) -> None:
    """
    可视化SVM的决策边界
    
    展示决策边界、间隔和支持向量。
    
    Args:
        svm_model: 训练好的SVM模型
        X: 数据点, shape (n_samples, 2)
        y: 标签
        title: 图标题
        show_plot: 是否显示图形
    """
    if not show_plot or X.shape[1] != 2:
        return
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 计算网格点的决策函数值
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制决策边界和间隔
    plt.contourf(xx, yy, Z, levels=[-np.inf, -1, 0, 1, np.inf],
                 colors=['blue', 'lightblue', 'white', 'lightcoral', 'red'],
                 alpha=0.3)
    
    # 绘制决策边界（f(x) = 0）
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    # 绘制间隔边界（f(x) = ±1）
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', 
                linewidths=1, linestyles='--')
    
    # 绘制训练样本
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                         s=50, edgecolors='black', linewidth=1)
    
    # 标记支持向量
    if hasattr(svm_model, 'support_vectors_'):
        plt.scatter(svm_model.support_vectors_[:, 0],
                   svm_model.support_vectors_[:, 1],
                   s=200, linewidth=2, facecolors='none',
                   edgecolors='green', label='支持向量')
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.colorbar(scatter, label='类别')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def demonstrate_svm_classification(show_plot: bool = True) -> None:
    """
    演示SVM分类
    
    展示不同核函数和参数对SVM的影响。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nSVM分类演示")
    print("=" * 60)
    
    # 生成二分类数据
    from sklearn.datasets import make_moons, make_circles
    
    datasets = {
        'Two Moons': make_moons(n_samples=200, noise=0.1, random_state=42),
        'Circles': make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    }
    
    for dataset_name, (X, y) in datasets.items():
        # 将标签转换为-1和1
        y = np.where(y == 0, -1, 1)
        
        print(f"\n数据集: {dataset_name}")
        print("-" * 40)
        
        # 不同的核函数
        kernels = ['linear', 'rbf', 'poly']
        
        for kernel in kernels:
            print(f"\n核函数: {kernel}")
            
            # 训练SVM
            if kernel == 'linear':
                svm = SVM(C=1.0, kernel=kernel)
            elif kernel == 'rbf':
                svm = SVM(C=1.0, kernel=kernel, gamma=0.5)
            else:  # poly
                svm = SVM(C=1.0, kernel=kernel, degree=3, gamma=1.0)
            
            svm.fit(X, y)
            
            # 计算训练准确率
            train_acc = svm.score(X, y)
            print(f"  训练准确率: {train_acc:.3f}")
            print(f"  支持向量数: {len(svm.support_vectors_)}/{len(X)}")
            
            # 可视化
            if show_plot:
                visualize_svm_decision_boundary(
                    svm, X, y,
                    title=f"SVM - {dataset_name} - {kernel}核",
                    show_plot=True
                )


def compare_c_parameter(show_plot: bool = True) -> None:
    """
    比较不同C参数的影响
    
    C控制软间隔的程度：
    - C大：硬间隔，容易过拟合
    - C小：软间隔，容易欠拟合
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nC参数影响分析")
    print("=" * 60)
    
    # 生成有噪声的线性可分数据
    np.random.seed(42)
    n_samples = 100
    
    # 生成两类数据
    X1 = np.random.randn(n_samples // 2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # 添加一些噪声点（错分样本）
    X[0] = [1, 0]  # 本应在类-1，但在类1的区域
    X[-1] = [-1, 0]  # 本应在类1，但在类-1的区域
    
    # 不同的C值
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = {}
    
    for C in C_values:
        print(f"\nC = {C}")
        
        # 训练SVM
        svm = SVM(C=C, kernel='linear')
        svm.fit(X, y)
        
        # 统计结果
        n_sv = len(svm.support_vectors_)
        train_acc = svm.score(X, y)
        
        results[C] = {
            'n_sv': n_sv,
            'train_acc': train_acc,
            'model': svm
        }
        
        print(f"  支持向量数: {n_sv}")
        print(f"  训练准确率: {train_acc:.3f}")
        
        # 可视化
        if show_plot:
            visualize_svm_decision_boundary(
                svm, X, y,
                title=f"SVM - C={C}",
                show_plot=True
            )
    
    print("\n观察：")
    print("1. C小：更多支持向量，软间隔，允许错分")
    print("2. C大：更少支持向量，硬间隔，不允许错分")
    print("3. C过大可能导致过拟合")
    print("4. C过小可能导致欠拟合")


def demonstrate_smo_algorithm(show_plot: bool = True) -> None:
    """
    演示SMO算法
    
    SMO是求解SVM的高效算法。
    
    Args:
        show_plot: 是否显示图形
    """
    print("\nSMO算法演示")
    print("=" * 60)
    
    # 生成简单的线性可分数据
    np.random.seed(42)
    n_samples = 50
    
    X1 = np.random.randn(n_samples // 2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))
    
    print("训练数据：")
    print(f"  样本数: {n_samples}")
    print(f"  特征维度: 2")
    
    # 使用SMO算法训练
    print("\n使用SMO算法训练...")
    smo = SimplifiedSMO(C=1.0, kernel='rbf', gamma=0.5, max_iter=100)
    smo.fit(X, y)
    
    # 计算准确率
    y_pred = smo.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"训练准确率: {accuracy:.3f}")
    
    # 可视化
    if show_plot:
        visualize_svm_decision_boundary(
            smo, X, y,
            title="SMO算法 - RBF核",
            show_plot=True
        )
    
    print("\nSMO算法特点：")
    print("1. 每次优化两个变量")
    print("2. 使用启发式选择变量")
    print("3. 避免大规模二次规划")
    print("4. 适合大规模数据集")