"""
1.3 模型选择 (Model Selection)
================================

模型选择是机器学习中的核心问题：如何选择合适的模型复杂度？
太简单会欠拟合，太复杂会过拟合。

主要方法：
1. 保留验证集 (Hold-out Validation)
2. 交叉验证 (Cross-Validation)
3. 信息准则 (AIC, BIC等)

本节重点介绍交叉验证，这是实践中最常用的方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from .polynomial_fitting import PolynomialCurveFitting


def train_validation_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据分割为训练集、验证集和测试集
    
    数据分割的原则：
    - 训练集：用于训练模型参数
    - 验证集：用于选择模型（如超参数调优）
    - 测试集：用于最终评估，只能使用一次！
    
    Args:
        X: 输入特征
        y: 目标值
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    # 计算分割点
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # 分割数据
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def cross_validation_demo(
    X: np.ndarray,
    y: np.ndarray,
    orders: List[int],
    n_folds: int = 5,
    show_plot: bool = True
) -> int:
    """
    使用k折交叉验证选择最优多项式阶数
    
    k折交叉验证的步骤：
    1. 将数据分成k个大小相似的子集（折）
    2. 对于每个模型：
       - 用k-1个折训练
       - 用剩余1个折验证
       - 重复k次，每个折都当过验证集
    3. 平均k次的验证误差作为模型性能估计
    
    优点：
    - 充分利用数据
    - 减少评估的方差
    - 对小数据集特别有效
    
    Args:
        X: 输入特征
        y: 目标值
        orders: 要测试的多项式阶数列表
        n_folds: 折数
        show_plot: 是否显示图形
        
    Returns:
        最优的多项式阶数
    """
    n_samples = len(X)
    fold_size = n_samples // n_folds
    
    cv_errors_mean = []
    cv_errors_std = []
    
    print(f"\n执行{n_folds}折交叉验证...")
    print("-" * 50)
    
    for order in orders:
        fold_errors = []
        
        for fold in range(n_folds):
            # 确定验证集的索引
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            
            # 创建训练和验证的索引
            val_indices = list(range(val_start, val_end))
            train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
            
            # 分割数据
            X_train_cv = X[train_indices]
            y_train_cv = y[train_indices]
            X_val_cv = X[val_indices]
            y_val_cv = y[val_indices]
            
            # 训练模型
            model = PolynomialCurveFitting(order, regularization=0.0)
            model.fit(X_train_cv, y_train_cv)
            
            # 计算验证误差
            val_error = model.compute_error(X_val_cv, y_val_cv)
            fold_errors.append(val_error)
        
        # 计算平均误差和标准差
        mean_error = np.mean(fold_errors)
        std_error = np.std(fold_errors)
        cv_errors_mean.append(mean_error)
        cv_errors_std.append(std_error)
        
        print(f"阶数 M={order}: CV误差 = {mean_error:.4f} ± {std_error:.4f}")
    
    # 选择最优阶数
    best_idx = int(np.argmin(cv_errors_mean))  # 转换为Python int类型
    best_order = orders[best_idx]
    
    print("-" * 50)
    print(f"最优多项式阶数: M={best_order}")
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # 绘制误差曲线
        plt.errorbar(orders, cv_errors_mean, yerr=cv_errors_std,
                    marker='o', linewidth=2, capsize=5, capthick=2,
                    markersize=8, label='CV误差')
        
        # 标记最优点
        plt.scatter(best_order, cv_errors_mean[best_idx],
                   color='red', s=200, zorder=5,
                   marker='*', label=f'最优 (M={best_order})')
        
        plt.xlabel('多项式阶数 M')
        plt.ylabel('交叉验证误差 (RMSE)')
        plt.title(f'{n_folds}折交叉验证：选择最优模型复杂度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加说明文字
        plt.text(0.02, 0.98, 
                f'最优阶数: M={best_order}\n'
                f'CV误差: {cv_errors_mean[best_idx]:.4f}',
                transform=plt.gca().transAxes,
                va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    return best_order