"""
3.2 偏差-方差分解 (The Bias-Variance Decomposition)
====================================================

机器学习的核心权衡之一：偏差-方差权衡

期望损失的分解：
E[(y - ŷ)²] = Bias² + Variance + Noise

其中：
- Bias²：偏差的平方，模型的系统性错误
- Variance：方差，模型对训练数据的敏感度
- Noise：不可约噪声，数据本身的随机性

关键洞察：
1. 简单模型：高偏差，低方差（欠拟合）
2. 复杂模型：低偏差，高方差（过拟合）
3. 最优模型：在偏差和方差之间平衡

偏差来源：
- 模型假设错误
- 模型表达能力不足

方差来源：
- 训练数据的有限性
- 模型对数据的过度敏感

这个分解帮助我们理解：
- 为什么增加模型复杂度不总是好的
- 为什么正则化有效
- 为什么集成方法（如bagging）能减少方差
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from .linear_basis_models import (
    PolynomialBasis, 
    LinearRegression,
    generate_synthetic_data
)
import warnings
warnings.filterwarnings('ignore')


def true_function(x: np.ndarray) -> np.ndarray:
    """
    真实的目标函数
    
    我们使用sin(2πx)作为真实函数，
    这是Bishop书中的经典例子。
    
    Args:
        x: 输入值
        
    Returns:
        真实函数值
    """
    return np.sin(2 * np.pi * x)


def generate_datasets(n_datasets: int, 
                     n_samples: int,
                     noise_std: float = 0.3,
                     x_min: float = 0,
                     x_max: float = 1,
                     seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成多个数据集
    
    用于偏差-方差分析，需要多个独立的数据集
    来估计模型的期望行为和方差。
    
    Args:
        n_datasets: 数据集数量
        n_samples: 每个数据集的样本数
        noise_std: 噪声标准差
        x_min, x_max: 输入范围
        seed: 随机种子
        
    Returns:
        数据集列表，每个元素是(X, y)
    """
    if seed is not None:
        np.random.seed(seed)
    
    datasets = []
    for _ in range(n_datasets):
        X = np.random.uniform(x_min, x_max, n_samples)
        # 真实函数 + 噪声
        y = true_function(X) + np.random.normal(0, noise_std, n_samples)
        datasets.append((X, y))
    
    return datasets


def compute_bias_variance(models_predictions: np.ndarray,
                         true_values: np.ndarray,
                         noise_variance: float) -> dict:
    """
    计算偏差和方差
    
    给定多个模型在同一点的预测，计算：
    - 偏差：平均预测与真实值的差异
    - 方差：预测的离散程度
    
    数学公式：
    - 平均预测：ȳ = E[ŷ]
    - 偏差²：Bias² = (ȳ - y_true)²
    - 方差：Var = E[(ŷ - ȳ)²]
    - 期望损失：E[(y - ŷ)²] = Bias² + Var + σ²
    
    Args:
        models_predictions: shape (n_models, n_points)
        true_values: shape (n_points,)
        noise_variance: 噪声方差σ²
        
    Returns:
        包含偏差、方差等的字典
    """
    # 平均预测
    mean_prediction = np.mean(models_predictions, axis=0)
    
    # 偏差²：(平均预测 - 真实值)²
    bias_squared = (mean_prediction - true_values) ** 2
    
    # 方差：预测围绕平均预测的方差
    variance = np.var(models_predictions, axis=0)
    
    # 期望损失的各个组成部分
    # 注意：实际的期望损失还包括噪声项
    expected_loss = bias_squared + variance + noise_variance
    
    return {
        'mean_prediction': mean_prediction,
        'bias_squared': bias_squared,
        'variance': variance,
        'noise_variance': noise_variance,
        'expected_loss': expected_loss,
        'avg_bias_squared': np.mean(bias_squared),
        'avg_variance': np.mean(variance),
        'avg_expected_loss': np.mean(expected_loss)
    }


def bias_variance_experiment(n_datasets: int = 100,
                            n_samples: int = 25,
                            polynomial_orders: List[int] = [1, 3, 9],
                            noise_std: float = 0.3,
                            test_points: Optional[np.ndarray] = None,
                            show_plot: bool = True) -> dict:
    """
    偏差-方差实验
    
    这个实验展示了模型复杂度如何影响偏差和方差：
    1. 生成多个数据集
    2. 在每个数据集上训练模型
    3. 计算模型集合的偏差和方差
    4. 观察随复杂度的变化
    
    Args:
        n_datasets: 数据集数量
        n_samples: 每个数据集的样本数
        polynomial_orders: 要测试的多项式阶数
        noise_std: 噪声标准差
        test_points: 测试点，None则自动生成
        show_plot: 是否绘图
        
    Returns:
        实验结果字典
    """
    print("\n偏差-方差分解实验")
    print("=" * 60)
    print(f"数据集数量: {n_datasets}")
    print(f"每个数据集样本数: {n_samples}")
    print(f"噪声标准差: {noise_std}")
    print(f"测试的多项式阶数: {polynomial_orders}")
    print("-" * 60)
    
    # 生成数据集
    datasets = generate_datasets(n_datasets, n_samples, noise_std, seed=42)
    
    # 测试点
    if test_points is None:
        test_points = np.linspace(0, 1, 100)
    
    # 真实函数值
    true_values = true_function(test_points)
    
    # 噪声方差
    noise_variance = noise_std ** 2
    
    results = {}
    
    for order in polynomial_orders:
        # 存储所有模型的预测
        all_predictions = []
        
        # 在每个数据集上训练模型
        for X_train, y_train in datasets:
            # 创建并训练模型
            basis = PolynomialBasis(order)
            model = LinearRegression(basis, regularization=0.0)
            model.fit(X_train, y_train)
            
            # 在测试点上预测
            predictions = model.predict(test_points)
            all_predictions.append(predictions)
        
        # 转换为数组：shape (n_datasets, n_test_points)
        all_predictions = np.array(all_predictions)
        
        # 计算偏差和方差
        analysis = compute_bias_variance(all_predictions, true_values, noise_variance)
        
        results[order] = analysis
        
        print(f"\nM={order}:")
        print(f"  平均偏差²: {analysis['avg_bias_squared']:.4f}")
        print(f"  平均方差: {analysis['avg_variance']:.4f}")
        print(f"  噪声方差: {noise_variance:.4f}")
        print(f"  期望损失: {analysis['avg_expected_loss']:.4f}")
    
    if show_plot:
        visualize_bias_variance(results, test_points, true_values, 
                               datasets[:20], polynomial_orders)
    
    return results


def visualize_bias_variance(results: dict, 
                           test_points: np.ndarray,
                           true_values: np.ndarray,
                           sample_datasets: List[Tuple],
                           polynomial_orders: List[int]) -> None:
    """
    可视化偏差-方差分解
    
    展示：
    1. 不同复杂度模型的拟合
    2. 偏差和方差随位置的变化
    3. 总体偏差-方差权衡
    
    Args:
        results: 分析结果
        test_points: 测试点
        true_values: 真实函数值
        sample_datasets: 用于展示的样本数据集
        polynomial_orders: 多项式阶数
    """
    n_orders = len(polynomial_orders)
    fig, axes = plt.subplots(3, n_orders, figsize=(5*n_orders, 12))
    
    for idx, order in enumerate(polynomial_orders):
        analysis = results[order]
        
        # 第1行：模型拟合示例
        ax1 = axes[0, idx] if n_orders > 1 else axes[0]
        
        # 绘制真实函数
        ax1.plot(test_points, true_values, 'g-', linewidth=2, 
                label='真实函数', alpha=0.7)
        
        # 绘制平均预测
        ax1.plot(test_points, analysis['mean_prediction'], 'r-', 
                linewidth=2, label='平均预测')
        
        # 绘制几个模型示例
        for i in range(min(5, len(sample_datasets))):
            X_train, y_train = sample_datasets[i]
            basis = PolynomialBasis(order)
            model = LinearRegression(basis)
            model.fit(X_train, y_train)
            pred = model.predict(test_points)
            ax1.plot(test_points, pred, 'b-', alpha=0.1, linewidth=0.5)
        
        # 显示一个数据集的点
        if len(sample_datasets) > 0:
            X_show, y_show = sample_datasets[0]
            ax1.scatter(X_show, y_show, s=20, c='blue', alpha=0.5, zorder=5)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'M={order} 的多个模型')
        ax1.set_ylim([-2, 2])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 第2行：偏差²和方差
        ax2 = axes[1, idx] if n_orders > 1 else axes[1]
        
        ax2.plot(test_points, analysis['bias_squared'], 'r-', 
                linewidth=2, label='偏差²')
        ax2.plot(test_points, analysis['variance'], 'b-', 
                linewidth=2, label='方差')
        ax2.axhline(y=analysis['noise_variance'], color='gray', 
                   linestyle='--', label='噪声方差')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('值')
        ax2.set_title(f'偏差²和方差 (M={order})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, max(0.5, np.max(analysis['bias_squared']) * 1.2)])
        
        # 第3行：期望损失分解
        ax3 = axes[2, idx] if n_orders > 1 else axes[2]
        
        # 堆叠条形图显示各部分贡献
        width = 0.6
        x_pos = [0]
        
        # 偏差²
        ax3.bar(x_pos, [analysis['avg_bias_squared']], width, 
               label='偏差²', color='red', alpha=0.7)
        
        # 方差（堆叠在偏差上）
        ax3.bar(x_pos, [analysis['avg_variance']], width,
               bottom=[analysis['avg_bias_squared']],
               label='方差', color='blue', alpha=0.7)
        
        # 噪声（堆叠在偏差+方差上）
        ax3.bar(x_pos, [analysis['noise_variance']], width,
               bottom=[analysis['avg_bias_squared'] + analysis['avg_variance']],
               label='噪声', color='gray', alpha=0.7)
        
        ax3.set_ylabel('贡献')
        ax3.set_title(f'期望损失分解 (M={order})')
        ax3.set_xticks([])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        total = analysis['avg_expected_loss']
        ax3.text(0, total + 0.01, f'总损失\n{total:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('偏差-方差分解分析', fontsize=14)
    plt.tight_layout()
    plt.show()


def bias_variance_vs_complexity(n_datasets: int = 100,
                               n_samples: int = 25,
                               max_order: int = 9,
                               noise_std: float = 0.3,
                               show_plot: bool = True) -> dict:
    """
    偏差和方差随模型复杂度的变化
    
    这个实验展示了经典的U型曲线：
    - 偏差随复杂度递减
    - 方差随复杂度递增
    - 总误差先降后升（最优复杂度）
    
    Args:
        n_datasets: 数据集数量
        n_samples: 每个数据集的样本数
        max_order: 最大多项式阶数
        noise_std: 噪声标准差
        show_plot: 是否绘图
        
    Returns:
        结果字典
    """
    print("\n偏差-方差 vs 模型复杂度")
    print("=" * 60)
    
    orders = list(range(0, max_order + 1))
    
    # 生成数据集
    datasets = generate_datasets(n_datasets, n_samples, noise_std, seed=42)
    
    # 测试点
    test_points = np.linspace(0, 1, 100)
    true_values = true_function(test_points)
    noise_variance = noise_std ** 2
    
    # 存储结果
    bias_squared_list = []
    variance_list = []
    total_error_list = []
    
    for order in orders:
        # 收集所有预测
        all_predictions = []
        
        for X_train, y_train in datasets:
            basis = PolynomialBasis(order)
            model = LinearRegression(basis)
            model.fit(X_train, y_train)
            predictions = model.predict(test_points)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # 计算偏差和方差
        analysis = compute_bias_variance(all_predictions, true_values, noise_variance)
        
        bias_squared_list.append(analysis['avg_bias_squared'])
        variance_list.append(analysis['avg_variance'])
        total_error_list.append(analysis['avg_expected_loss'])
    
    results = {
        'orders': orders,
        'bias_squared': bias_squared_list,
        'variance': variance_list,
        'noise': [noise_variance] * len(orders),
        'total_error': total_error_list
    }
    
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：偏差和方差曲线
        ax1.plot(orders, bias_squared_list, 'r-o', linewidth=2, 
                label='偏差²', markersize=8)
        ax1.plot(orders, variance_list, 'b-s', linewidth=2, 
                label='方差', markersize=8)
        ax1.axhline(y=noise_variance, color='gray', linestyle='--', 
                   linewidth=2, label='噪声')
        
        ax1.set_xlabel('多项式阶数 M')
        ax1.set_ylabel('值')
        ax1.set_title('偏差-方差权衡')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.5, max_order + 0.5])
        
        # 右图：总误差
        ax2.plot(orders, total_error_list, 'g-o', linewidth=2, 
                markersize=8, label='期望损失')
        
        # 标记最优点
        min_idx = np.argmin(total_error_list)
        ax2.scatter(orders[min_idx], total_error_list[min_idx], 
                   color='red', s=200, zorder=5, marker='*',
                   label=f'最优 (M={orders[min_idx]})')
        
        ax2.set_xlabel('多项式阶数 M')
        ax2.set_ylabel('期望损失')
        ax2.set_title('模型选择')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.5, max_order + 0.5])
        
        # 添加区域标注
        ax2.axvspan(-0.5, 2.5, alpha=0.2, color='red', label='欠拟合')
        ax2.axvspan(6.5, max_order + 0.5, alpha=0.2, color='blue', label='过拟合')
        
        plt.suptitle(f'偏差-方差权衡 (N={n_samples}, L={n_datasets})', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 打印最优模型
    min_idx = np.argmin(total_error_list)
    print(f"\n最优模型: M={orders[min_idx]}")
    print(f"  偏差²: {bias_squared_list[min_idx]:.4f}")
    print(f"  方差: {variance_list[min_idx]:.4f}")
    print(f"  总误差: {total_error_list[min_idx]:.4f}")
    
    print("\n关键观察：")
    print("1. 低复杂度：高偏差（欠拟合），低方差")
    print("2. 高复杂度：低偏差，高方差（过拟合）")
    print("3. 最优复杂度：偏差和方差的最佳平衡")
    print("4. 噪声是不可约的下界")
    
    return results


def demonstrate_regularization_bias_variance(n_datasets: int = 100,
                                            n_samples: int = 25,
                                            polynomial_order: int = 9,
                                            lambda_values: List[float] = None,
                                            noise_std: float = 0.3,
                                            show_plot: bool = True) -> dict:
    """
    正则化对偏差-方差的影响
    
    展示正则化如何：
    - 增加偏差（限制模型灵活性）
    - 减少方差（降低对数据的敏感度）
    - 可能降低总误差
    
    Args:
        n_datasets: 数据集数量
        n_samples: 每个数据集的样本数
        polynomial_order: 多项式阶数
        lambda_values: 正则化系数列表
        noise_std: 噪声标准差
        show_plot: 是否绘图
        
    Returns:
        结果字典
    """
    print("\n正则化对偏差-方差的影响")
    print("=" * 60)
    print(f"多项式阶数: M={polynomial_order}")
    
    if lambda_values is None:
        lambda_values = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    
    # 生成数据集
    datasets = generate_datasets(n_datasets, n_samples, noise_std, seed=42)
    
    # 测试点
    test_points = np.linspace(0, 1, 100)
    true_values = true_function(test_points)
    noise_variance = noise_std ** 2
    
    results = {}
    
    for lambda_val in lambda_values:
        # 收集所有预测
        all_predictions = []
        
        for X_train, y_train in datasets:
            basis = PolynomialBasis(polynomial_order)
            model = LinearRegression(basis, regularization=lambda_val)
            model.fit(X_train, y_train)
            predictions = model.predict(test_points)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # 计算偏差和方差
        analysis = compute_bias_variance(all_predictions, true_values, noise_variance)
        results[lambda_val] = analysis
        
        print(f"\nλ={lambda_val:.4f}:")
        print(f"  偏差²: {analysis['avg_bias_squared']:.4f}")
        print(f"  方差: {analysis['avg_variance']:.4f}")
        print(f"  总误差: {analysis['avg_expected_loss']:.4f}")
    
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 提取数据
        lambdas = list(results.keys())
        bias_squared = [results[l]['avg_bias_squared'] for l in lambdas]
        variance = [results[l]['avg_variance'] for l in lambdas]
        total_error = [results[l]['avg_expected_loss'] for l in lambdas]
        
        # 左图：偏差和方差
        ax1.semilogx(lambdas, bias_squared, 'r-o', linewidth=2, 
                    label='偏差²', markersize=8)
        ax1.semilogx(lambdas, variance, 'b-s', linewidth=2, 
                    label='方差', markersize=8)
        ax1.axhline(y=noise_variance, color='gray', linestyle='--', 
                   linewidth=2, label='噪声')
        
        ax1.set_xlabel('正则化系数 λ (log scale)')
        ax1.set_ylabel('值')
        ax1.set_title(f'正则化的偏差-方差效应 (M={polynomial_order})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：总误差
        ax2.semilogx(lambdas, total_error, 'g-o', linewidth=2, 
                    markersize=8, label='期望损失')
        
        # 标记最优点
        min_idx = np.argmin(total_error)
        ax2.scatter(lambdas[min_idx], total_error[min_idx], 
                   color='red', s=200, zorder=5, marker='*',
                   label=f'最优 (λ={lambdas[min_idx]:.4f})')
        
        ax2.set_xlabel('正则化系数 λ (log scale)')
        ax2.set_ylabel('期望损失')
        ax2.set_title('正则化参数选择')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'正则化与偏差-方差权衡', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. λ=0：无正则化，低偏差，高方差")
    print("2. λ增大：偏差增加，方差减少")
    print("3. 存在最优λ，使总误差最小")
    print("4. 正则化通过增加偏差来减少方差")
    
    return results