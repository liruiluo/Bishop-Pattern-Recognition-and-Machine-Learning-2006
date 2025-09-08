"""
1.4 维度诅咒 (The Curse of Dimensionality)
============================================

维度诅咒是高维数据的本质困难。随着维度增加：
1. 数据变得稀疏
2. 距离度量失效
3. 需要指数级增长的数据量

这就是为什么深度学习等方法在高维问题上如此重要。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.special import gamma


def volume_of_hypersphere(d: int, r: float = 1.0) -> float:
    """
    计算d维超球的体积
    
    d维球的体积公式：
    V_d(r) = (π^(d/2) / Γ(d/2 + 1)) * r^d
    
    其中Γ是gamma函数
    
    有趣的事实：随着维度增加，单位球的体积趋近于0！
    
    Args:
        d: 维度
        r: 半径
        
    Returns:
        体积
    """
    # 使用gamma函数计算
    volume = (np.pi ** (d/2) / gamma(d/2 + 1)) * (r ** d)
    return volume


def demonstrate_curse_of_dimensionality(
    dimensions: List[int],
    n_samples: int = 1000,
    show_plot: bool = True
) -> None:
    """
    演示维度诅咒的几个方面
    
    1. 超球体积随维度的变化
    2. 点到点距离的分布
    3. 数据在高维空间的稀疏性
    
    Args:
        dimensions: 要测试的维度列表
        n_samples: 每个维度的样本数
        show_plot: 是否显示图形
    """
    print("\n维度诅咒的演示：")
    print("=" * 50)
    
    # 1. 计算不同维度下单位球的体积
    volumes = [volume_of_hypersphere(d) for d in dimensions]
    
    print("\n1. 单位超球体积随维度的变化：")
    print("-" * 40)
    for d, v in zip(dimensions[:6], volumes[:6]):  # 只显示前6个
        print(f"  {d}维: {v:.6f}")
    print("  ...")
    print(f"  注意：体积在高维时趋近于0！")
    
    # 2. 计算不同维度下随机点对的距离分布
    min_distances = []
    max_distances = []
    mean_distances = []
    
    print("\n2. 随机点对间的欧氏距离：")
    print("-" * 40)
    
    for d in dimensions:
        # 在[0,1]^d超立方体中生成随机点
        points = np.random.uniform(0, 1, (n_samples, d))
        
        # 计算所有点对的距离
        distances = []
        for i in range(min(100, n_samples)):  # 限制计算量
            for j in range(i+1, min(100, n_samples)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        min_distances.append(np.min(distances))
        max_distances.append(np.max(distances))
        mean_distances.append(np.mean(distances))
        
        if d in [1, 2, 10, 50]:
            print(f"  {d}维: 平均={np.mean(distances):.3f}, "
                  f"最小={np.min(distances):.3f}, "
                  f"最大={np.max(distances):.3f}")
    
    print("\n观察：")
    print("- 随着维度增加，点对间的距离越来越相似")
    print("- 这使得基于距离的方法（如KNN）在高维下失效")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1：超球体积
        ax1 = axes[0, 0]
        ax1.semilogy(dimensions, volumes, 'b-o', linewidth=2)
        ax1.set_xlabel('维度 d')
        ax1.set_ylabel('单位球体积 (log scale)')
        ax1.set_title('单位超球体积随维度的变化')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：距离分布
        ax2 = axes[0, 1]
        ax2.plot(dimensions, min_distances, 'g-o', label='最小距离')
        ax2.plot(dimensions, mean_distances, 'b-o', label='平均距离')
        ax2.plot(dimensions, max_distances, 'r-o', label='最大距离')
        ax2.set_xlabel('维度 d')
        ax2.set_ylabel('欧氏距离')
        ax2.set_title('随机点对间的距离分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：超立方体与内切球的体积比
        ax3 = axes[1, 0]
        cube_volume = 1  # 单位超立方体体积始终为1
        sphere_ratios = [volume_of_hypersphere(d, 0.5) for d in dimensions]
        ax3.semilogy(dimensions, sphere_ratios, 'r-o', linewidth=2)
        ax3.set_xlabel('维度 d')
        ax3.set_ylabel('内切球/立方体体积比 (log scale)')
        ax3.set_title('内切球体积占比随维度的变化')
        ax3.grid(True, alpha=0.3)
        
        # 子图4：数据点到中心的距离分布
        ax4 = axes[1, 1]
        center_distances = []
        for d in [1, 2, 5, 10, 20]:
            points = np.random.uniform(-1, 1, (1000, d))
            center = np.zeros(d)
            dists = np.linalg.norm(points - center, axis=1)
            ax4.hist(dists, bins=30, alpha=0.5, density=True, label=f'd={d}')
        
        ax4.set_xlabel('到原点的距离')
        ax4.set_ylabel('概率密度')
        ax4.set_title('不同维度下点到中心的距离分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('维度诅咒的多个方面', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n维度诅咒的启示：")
    print("1. 高维空间中大部分体积集中在边界附近")
    print("2. 需要指数级增长的数据才能充分采样高维空间")
    print("3. 降维和特征选择在高维问题中至关重要")