"""
2.5 非参数方法 (Nonparametric Methods)
======================================

参数方法假设数据来自特定分布族（如高斯分布），
只需要估计有限个参数。

非参数方法不做这种假设，让数据自己决定分布形状。
优点：灵活，能适应任意分布
缺点：需要更多数据，计算复杂度高

主要方法：
1. 直方图 (Histogram)：最简单的密度估计
2. 核密度估计 (KDE)：平滑的密度估计
3. K近邻 (KNN)：基于局部邻域的估计

核心思想：
概率密度 ≈ (区域内的样本数) / (样本总数 × 区域体积)

当我们让区域体积随数据自适应变化时，
就得到了不同的非参数方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


class HistogramDensityEstimator:
    """
    直方图密度估计
    
    最简单的非参数密度估计方法。
    将数据空间划分为等宽的箱(bins)，
    统计每个箱中的数据点数。
    
    密度估计：
    p(x) ≈ (箱中点数) / (总点数 × 箱宽度)
    
    优点：
    - 简单直观
    - 计算快速
    
    缺点：
    - 不连续（阶梯状）
    - 依赖于箱的位置和宽度
    - 高维时的维度诅咒
    
    箱宽度选择：
    - 太窄：方差大（过拟合）
    - 太宽：偏差大（过平滑）
    需要在偏差-方差之间权衡
    """
    
    def __init__(self, bin_width: Optional[float] = None, 
                 n_bins: Optional[int] = None):
        """
        初始化直方图估计器
        
        Args:
            bin_width: 箱宽度
            n_bins: 箱数量（与bin_width二选一）
        """
        self.bin_width = bin_width
        self.n_bins = n_bins
        self.bins = None
        self.densities = None
        self.data_range = None
    
    def fit(self, data: np.ndarray) -> 'HistogramDensityEstimator':
        """
        拟合直方图
        
        Args:
            data: 训练数据，shape (n_samples,) 或 (n_samples, 1)
            
        Returns:
            self
        """
        data = data.ravel()
        n_samples = len(data)
        
        # 确定数据范围
        self.data_range = (data.min(), data.max())
        range_width = self.data_range[1] - self.data_range[0]
        
        # 确定箱的数量和宽度
        if self.n_bins is not None:
            # 使用指定的箱数量
            self.bin_width = range_width / self.n_bins
        elif self.bin_width is not None:
            # 使用指定的箱宽度
            self.n_bins = int(np.ceil(range_width / self.bin_width))
        else:
            # 使用Sturges规则：n_bins = ⌈log₂(n) + 1⌉
            # 这是一个经验规则，适用于正态分布
            self.n_bins = int(np.ceil(np.log2(n_samples) + 1))
            self.bin_width = range_width / self.n_bins
        
        # 计算直方图
        counts, bin_edges = np.histogram(data, bins=self.n_bins, 
                                        range=self.data_range)
        
        # 计算密度
        # 密度 = 计数 / (总样本数 × 箱宽度)
        self.densities = counts / (n_samples * self.bin_width)
        self.bins = bin_edges
        
        return self
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算概率密度
        
        Args:
            x: 查询点
            
        Returns:
            密度估计值
        """
        if self.bins is None:
            raise ValueError("需要先调用fit方法")
        
        x = np.asarray(x).ravel()
        pdf_values = np.zeros_like(x)
        
        # 对每个查询点，找到对应的箱
        for i, xi in enumerate(x):
            if xi < self.bins[0] or xi > self.bins[-1]:
                pdf_values[i] = 0
            else:
                # 找到所属的箱
                bin_idx = np.searchsorted(self.bins[:-1], xi) - 1
                bin_idx = np.clip(bin_idx, 0, len(self.densities) - 1)
                pdf_values[i] = self.densities[bin_idx]
        
        return pdf_values
    
    def plot(self, ax: Optional[plt.Axes] = None, 
             show_data: bool = True, data: Optional[np.ndarray] = None):
        """绘制直方图"""
        if ax is None:
            fig, ax = plt.subplots()
        
        # 绘制直方图
        for i in range(len(self.densities)):
            ax.bar(self.bins[i], self.densities[i], 
                  width=self.bin_width, align='edge',
                  alpha=0.7, edgecolor='black')
        
        # 显示数据点
        if show_data and data is not None:
            ax.scatter(data, np.zeros_like(data), 
                      marker='|', s=100, c='red', alpha=0.5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('密度')
        ax.set_title(f'直方图密度估计 (bins={self.n_bins})')


class KernelDensityEstimator:
    """
    核密度估计 (Kernel Density Estimation, KDE)
    
    在每个数据点放置一个核函数，然后求和。
    
    密度估计：
    p(x) = (1/n) Σᵢ K_h(x - xᵢ)
    
    其中：
    - K_h(u) = (1/h) K(u/h) 是缩放的核函数
    - h 是带宽（平滑参数）
    - K 是核函数（如高斯核）
    
    可以理解为：
    1. 在每个数据点放置一个"山峰"（核）
    2. 所有山峰叠加形成密度估计
    3. 带宽控制山峰的宽度
    
    核函数选择：
    - 高斯核：最常用，平滑
    - Epanechnikov核：理论最优（最小均方误差）
    - 均匀核：矩形窗口
    
    带宽选择：
    - 太小：欠平滑（方差大）
    - 太大：过平滑（偏差大）
    - 经验规则：Silverman规则，Scott规则
    """
    
    def __init__(self, kernel: str = 'gaussian', bandwidth: Optional[float] = None):
        """
        初始化KDE估计器
        
        Args:
            kernel: 核函数类型 ('gaussian', 'epanechnikov', 'uniform')
            bandwidth: 带宽参数h
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None
        
        # 定义核函数
        self.kernel_functions = {
            'gaussian': lambda u: np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi),
            'epanechnikov': lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1),
            'uniform': lambda u: 0.5 * (np.abs(u) <= 1)
        }
    
    def fit(self, data: np.ndarray) -> 'KernelDensityEstimator':
        """
        拟合KDE（实际上只是存储数据）
        
        Args:
            data: 训练数据
            
        Returns:
            self
        """
        self.data = np.asarray(data).ravel()
        n_samples = len(self.data)
        
        # 如果没有指定带宽，使用Silverman规则
        # h = 1.06 * σ * n^(-1/5)
        # 这是基于最小化MISE（均方积分误差）的经验规则
        if self.bandwidth is None:
            sigma = np.std(self.data)
            self.bandwidth = 1.06 * sigma * (n_samples ** (-0.2))
        
        return self
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算概率密度
        
        KDE的计算：在每个查询点，计算所有核的贡献之和
        
        Args:
            x: 查询点
            
        Returns:
            密度估计值
        """
        if self.data is None:
            raise ValueError("需要先调用fit方法")
        
        x = np.asarray(x).ravel()
        n_samples = len(self.data)
        pdf_values = np.zeros_like(x)
        
        kernel_func = self.kernel_functions[self.kernel]
        
        for i, xi in enumerate(x):
            # 计算xi到所有数据点的标准化距离
            u = (xi - self.data) / self.bandwidth
            # 计算所有核的贡献
            kernel_values = kernel_func(u)
            # 求平均并归一化
            pdf_values[i] = np.mean(kernel_values) / self.bandwidth
        
        return pdf_values
    
    def plot(self, ax: Optional[plt.Axes] = None, 
             x_range: Optional[Tuple[float, float]] = None,
             show_kernels: bool = False):
        """
        绘制KDE
        
        Args:
            ax: 绘图轴
            x_range: x轴范围
            show_kernels: 是否显示单个核函数
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if x_range is None:
            margin = 3 * self.bandwidth
            x_range = (self.data.min() - margin, self.data.max() + margin)
        
        x_plot = np.linspace(x_range[0], x_range[1], 200)
        pdf_values = self.pdf(x_plot)
        
        # 绘制总体密度
        ax.plot(x_plot, pdf_values, 'b-', linewidth=2, 
               label=f'KDE ({self.kernel}, h={self.bandwidth:.3f})')
        
        # 显示单个核函数
        if show_kernels and len(self.data) <= 20:
            kernel_func = self.kernel_functions[self.kernel]
            for xi in self.data:
                # 单个核的贡献
                kernel_x = x_plot
                u = (kernel_x - xi) / self.bandwidth
                kernel_y = kernel_func(u) / (len(self.data) * self.bandwidth)
                ax.plot(kernel_x, kernel_y, 'g--', alpha=0.3, linewidth=0.5)
        
        # 显示数据点
        ax.scatter(self.data, np.zeros_like(self.data), 
                  marker='|', s=100, c='red', alpha=0.5, label='数据点')
        
        ax.set_xlabel('x')
        ax.set_ylabel('密度')
        ax.set_title(f'核密度估计 ({self.kernel}核)')
        ax.legend()


class KNearestNeighborsDensity:
    """
    K近邻密度估计
    
    与KDE不同，KNN固定邻居数K，让体积V自适应。
    
    密度估计：
    p(x) ≈ K / (n × V(x))
    
    其中V(x)是包含K个最近邻的球的体积。
    
    特点：
    - 在数据密集区域，V小，密度高
    - 在数据稀疏区域，V大，密度低
    - 自适应带宽
    
    与KDE的对比：
    - KDE：固定带宽，让K变化
    - KNN：固定K，让带宽变化
    
    应用：
    - 密度估计
    - 分类（KNN分类器）
    - 异常检测（局部异常因子LOF）
    """
    
    def __init__(self, k: int = 5):
        """
        初始化KNN密度估计器
        
        Args:
            k: 近邻数
        """
        self.k = k
        self.data = None
        self.n_samples = 0
        self.dim = 1
    
    def fit(self, data: np.ndarray) -> 'KNearestNeighborsDensity':
        """
        拟合KNN估计器
        
        Args:
            data: 训练数据，shape (n_samples,) 或 (n_samples, n_features)
            
        Returns:
            self
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.data = data
        self.n_samples, self.dim = data.shape
        
        if self.k > self.n_samples:
            raise ValueError(f"K={self.k}不能大于样本数{self.n_samples}")
        
        return self
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算概率密度
        
        对每个查询点：
        1. 找到K个最近邻
        2. 计算到第K个邻居的距离（半径）
        3. 计算包含K个邻居的球的体积
        4. 密度 = K / (n × 体积)
        
        Args:
            x: 查询点，shape (n_queries,) 或 (n_queries, n_features)
            
        Returns:
            密度估计值
        """
        if self.data is None:
            raise ValueError("需要先调用fit方法")
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_queries = x.shape[0]
        pdf_values = np.zeros(n_queries)
        
        # 计算距离矩阵
        distances = cdist(x, self.data)
        
        for i in range(n_queries):
            # 找到K个最近邻
            dist_to_neighbors = distances[i]
            k_nearest_distances = np.sort(dist_to_neighbors)[:self.k]
            
            # 第K个邻居的距离作为半径
            radius = k_nearest_distances[-1]
            
            # 避免半径为0（查询点恰好是数据点）
            if radius == 0:
                radius = k_nearest_distances[k_nearest_distances > 0][0] \
                        if np.any(k_nearest_distances > 0) else 1e-10
            
            # 计算d维球的体积
            # V = c_d × r^d
            # 其中c_d是d维单位球的体积
            if self.dim == 1:
                volume = 2 * radius  # 1维是线段长度
            elif self.dim == 2:
                volume = np.pi * radius**2  # 2维是圆面积
            else:
                # d维球体积公式
                from scipy.special import gamma
                c_d = np.pi**(self.dim/2) / gamma(self.dim/2 + 1)
                volume = c_d * radius**self.dim
            
            # 密度估计
            pdf_values[i] = self.k / (self.n_samples * volume)
        
        return pdf_values
    
    def plot_1d(self, ax: Optional[plt.Axes] = None,
                x_range: Optional[Tuple[float, float]] = None):
        """绘制1维KNN密度估计"""
        if self.dim != 1:
            raise ValueError("只能绘制1维数据")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if x_range is None:
            margin = 0.1 * (self.data.max() - self.data.min())
            x_range = (self.data.min() - margin, self.data.max() + margin)
        
        x_plot = np.linspace(x_range[0], x_range[1], 200)
        pdf_values = self.pdf(x_plot)
        
        ax.plot(x_plot, pdf_values, 'b-', linewidth=2, 
               label=f'KNN (K={self.k})')
        ax.scatter(self.data, np.zeros_like(self.data), 
                  marker='|', s=100, c='red', alpha=0.5, label='数据点')
        
        ax.set_xlabel('x')
        ax.set_ylabel('密度')
        ax.set_title(f'K近邻密度估计 (K={self.k})')
        ax.legend()


def compare_density_estimators(true_distribution: str = 'mixture',
                              n_samples: int = 100,
                              show_plot: bool = True) -> None:
    """
    比较不同的密度估计方法
    
    通过一个混合高斯分布的例子，展示：
    1. 直方图：简单但不平滑
    2. KDE：平滑但带宽选择重要
    3. KNN：自适应但在边界处有问题
    
    Args:
        true_distribution: 真实分布类型
        n_samples: 样本数量
        show_plot: 是否显示图形
    """
    print("\n密度估计方法比较")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成数据：双峰混合高斯
    if true_distribution == 'mixture':
        # 混合高斯：0.4×N(-2,0.5) + 0.6×N(1,0.8)
        n1 = int(0.4 * n_samples)
        n2 = n_samples - n1
        data1 = np.random.normal(-2, 0.5, n1)
        data2 = np.random.normal(1, 0.8, n2)
        data = np.concatenate([data1, data2])
        
        # 真实密度函数
        def true_pdf(x):
            return (0.4 * stats.norm.pdf(x, -2, 0.5) + 
                   0.6 * stats.norm.pdf(x, 1, 0.8))
    else:
        # 标准正态分布
        data = np.random.normal(0, 1, n_samples)
        true_pdf = lambda x: stats.norm.pdf(x, 0, 1)
    
    print(f"数据：{n_samples}个样本")
    print(f"真实分布：{true_distribution}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        x_range = (data.min() - 1, data.max() + 1)
        x_plot = np.linspace(x_range[0], x_range[1], 200)
        true_density = true_pdf(x_plot)
        
        # 1. 直方图（不同bin数）
        for idx, n_bins in enumerate([10, 20, 30]):
            ax = axes[0, idx]
            
            hist_estimator = HistogramDensityEstimator(n_bins=n_bins)
            hist_estimator.fit(data)
            
            # 绘制直方图
            hist_estimator.plot(ax, show_data=True, data=data[:20])
            
            # 绘制真实密度
            ax.plot(x_plot, true_density, 'g-', linewidth=2, 
                   alpha=0.7, label='真实密度')
            
            ax.set_title(f'直方图 (bins={n_bins})')
            ax.legend()
            ax.set_ylim([0, max(true_density) * 1.2])
        
        # 2. KDE（不同带宽）
        bandwidths = [0.1, 0.3, 0.5]
        for idx, h in enumerate(bandwidths):
            ax = axes[1, idx]
            
            kde_estimator = KernelDensityEstimator(kernel='gaussian', bandwidth=h)
            kde_estimator.fit(data)
            
            # 绘制KDE
            kde_density = kde_estimator.pdf(x_plot)
            ax.plot(x_plot, kde_density, 'b-', linewidth=2, 
                   label=f'KDE (h={h})')
            
            # 绘制真实密度
            ax.plot(x_plot, true_density, 'g-', linewidth=2, 
                   alpha=0.7, label='真实密度')
            
            # 显示数据点
            ax.scatter(data[:20], np.zeros(min(20, len(data))), 
                      marker='|', s=100, c='red', alpha=0.5)
            
            ax.set_xlabel('x')
            ax.set_ylabel('密度')
            ax.set_title(f'核密度估计 (h={h})')
            ax.legend()
            ax.set_ylim([0, max(true_density) * 1.2])
        
        plt.suptitle(f'密度估计方法比较 (n={n_samples})', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 计算误差
    print("\n估计误差（L2范数）：")
    print("-" * 40)
    
    x_eval = np.linspace(x_range[0], x_range[1], 100)
    true_eval = true_pdf(x_eval)
    
    # 直方图误差
    hist_est = HistogramDensityEstimator(n_bins=20)
    hist_est.fit(data)
    hist_eval = hist_est.pdf(x_eval)
    hist_error = np.sqrt(np.mean((hist_eval - true_eval)**2))
    print(f"直方图 (20 bins): {hist_error:.4f}")
    
    # KDE误差（最优带宽）
    kde_est = KernelDensityEstimator(kernel='gaussian')
    kde_est.fit(data)
    kde_eval = kde_est.pdf(x_eval)
    kde_error = np.sqrt(np.mean((kde_eval - true_eval)**2))
    print(f"KDE (Silverman带宽): {kde_error:.4f}")
    
    # KNN误差
    knn_est = KNearestNeighborsDensity(k=int(np.sqrt(n_samples)))
    knn_est.fit(data.reshape(-1, 1))
    knn_eval = knn_est.pdf(x_eval.reshape(-1, 1))
    knn_error = np.sqrt(np.mean((knn_eval - true_eval)**2))
    print(f"KNN (K=√n): {knn_error:.4f}")
    
    print("\n观察：")
    print("1. 直方图简单但不平滑")
    print("2. KDE平滑但需要选择合适的带宽")
    print("3. 带宽选择是偏差-方差权衡")
    print("4. 样本量越大，估计越准确")


def demonstrate_bandwidth_selection() -> None:
    """
    演示带宽选择的重要性
    
    带宽是KDE中最重要的参数：
    - 太小：欠平滑，看到太多噪声
    - 太大：过平滑，丢失重要特征
    
    常用的带宽选择方法：
    1. Silverman规则
    2. Scott规则
    3. 交叉验证
    """
    print("\n带宽选择的影响")
    print("=" * 60)
    
    # 生成双峰数据
    np.random.seed(42)
    n_samples = 200
    data1 = np.random.normal(-2, 0.5, n_samples // 2)
    data2 = np.random.normal(2, 0.5, n_samples // 2)
    data = np.concatenate([data1, data2])
    
    # 不同的带宽选择规则
    sigma = np.std(data)
    n = len(data)
    
    # Silverman规则
    h_silverman = 1.06 * sigma * (n ** (-0.2))
    
    # Scott规则
    h_scott = 3.49 * sigma * (n ** (-1/3))
    
    # 极端情况
    h_undersmooth = 0.05
    h_oversmooth = 2.0
    
    bandwidths = {
        '欠平滑 (h=0.05)': h_undersmooth,
        f'Scott (h={h_scott:.3f})': h_scott,
        f'Silverman (h={h_silverman:.3f})': h_silverman,
        '过平滑 (h=2.0)': h_oversmooth
    }
    
    print("带宽选择规则：")
    print("-" * 40)
    for name, h in bandwidths.items():
        print(f"{name:<25} h = {h:.3f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    x_plot = np.linspace(-6, 6, 200)
    
    for idx, (name, h) in enumerate(bandwidths.items()):
        ax = axes[idx]
        
        # KDE估计
        kde = KernelDensityEstimator(kernel='gaussian', bandwidth=h)
        kde.fit(data)
        pdf_values = kde.pdf(x_plot)
        
        # 绘制
        ax.plot(x_plot, pdf_values, 'b-', linewidth=2)
        ax.fill_between(x_plot, pdf_values, alpha=0.3)
        ax.scatter(data, np.zeros_like(data), 
                  marker='|', s=50, c='red', alpha=0.3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('密度')
        ax.set_title(name)
        ax.set_ylim([0, 0.5])
        ax.grid(True, alpha=0.3)
        
        # 标记两个峰
        peaks_true = [-2, 2]
        for peak in peaks_true:
            ax.axvline(x=peak, color='green', linestyle='--', 
                      alpha=0.5, linewidth=1)
    
    plt.suptitle('带宽选择对KDE的影响', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n关键观察：")
    print("1. 欠平滑：能看到数据的细节，但太多噪声")
    print("2. 过平滑：丢失了双峰结构")
    print("3. 合适的带宽：保持了主要特征，同时平滑了噪声")
    print("4. 没有普适的最优带宽，需要根据具体问题选择")