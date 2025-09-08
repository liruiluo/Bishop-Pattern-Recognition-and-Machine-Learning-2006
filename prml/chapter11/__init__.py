"""
Chapter 11: Sampling Methods (采样方法)
=======================================

本章介绍从概率分布中生成样本的方法。

主要内容：
1. 基础采样方法 (11.1)
   - 逆变换采样
   - 拒绝采样
   - 重要性采样
   - Box-Muller变换

2. 马尔可夫链蒙特卡罗 (11.2-11.3)
   - Metropolis-Hastings算法
   - Gibbs采样
   - 哈密顿蒙特卡罗(HMC)

3. 切片采样 (11.4)
   - 单变量切片采样
   - 多变量切片采样

核心概念：
采样是贝叶斯推理和蒙特卡罗方法的基础。
MCMC通过构建马尔可夫链来采样复杂分布。

关键性质：
- 详细平衡：π(x)T(x'|x) = π(x')T(x|x')
- 遍历性：从任意初始状态可达任意状态
- 平稳分布：长时间后收敛到目标分布

收敛诊断：
- R̂统计量
- 有效样本大小(ESS)
- 自相关函数

应用：
- 贝叶斯后验采样
- 统计物理模拟
- 概率图模型推理
- 深度生成模型
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .basic_sampling import (
    RejectionSampler,
    ImportanceSampler,
    box_muller_transform,
    inverse_transform_sampling,
    demonstrate_rejection_sampling,
    demonstrate_importance_sampling,
    demonstrate_box_muller
)

from .mcmc import (
    MetropolisHastings,
    GibbsSampler,
    HamiltonianMC,
    compute_effective_sample_size,
    compute_rhat,
    demonstrate_metropolis_hastings,
    demonstrate_gibbs_sampling,
    demonstrate_hmc
)


def run_chapter11(cfg: DictConfig) -> None:
    """
    运行第11章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第11章：采样方法 (Sampling Methods)")
    print("="*80)
    
    # 11.1 基础采样方法
    print("\n" + "-"*60)
    print("11.1 基础采样方法")
    print("-"*60)
    
    # 拒绝采样
    demonstrate_rejection_sampling(
        show_plot=cfg.visualization.show_plots
    )
    
    # 重要性采样
    demonstrate_importance_sampling(
        show_plot=cfg.visualization.show_plots
    )
    
    # Box-Muller变换
    demonstrate_box_muller(
        show_plot=cfg.visualization.show_plots
    )
    
    # 11.2-11.3 马尔可夫链蒙特卡罗
    print("\n" + "-"*60)
    print("11.2-11.3 马尔可夫链蒙特卡罗")
    print("-"*60)
    
    # Metropolis-Hastings
    demonstrate_metropolis_hastings(
        show_plot=cfg.visualization.show_plots
    )
    
    # Gibbs采样
    demonstrate_gibbs_sampling(
        show_plot=cfg.visualization.show_plots
    )
    
    # 哈密顿蒙特卡罗
    demonstrate_hmc(
        show_plot=cfg.visualization.show_plots
    )
    
    # 11.4 切片采样
    print("\n" + "-"*60)
    print("11.4 切片采样")
    print("-"*60)
    
    demonstrate_slice_sampling(
        show_plot=cfg.visualization.show_plots
    )
    
    # MCMC比较
    compare_mcmc_methods(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第11章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 基础方法简单但可能低效")
    print("2. MCMC可处理复杂高维分布")
    print("3. 不同MCMC方法有不同权衡")
    print("4. HMC利用梯度信息提高效率")
    print("5. 收敛诊断确保结果可靠")
    print("6. 采样是贝叶斯推理的核心工具")


def demonstrate_slice_sampling(show_plot: bool = True) -> None:
    """
    演示切片采样
    """
    print("\n切片采样演示")
    print("=" * 60)
    
    # 目标分布：混合高斯
    def target_pdf(x):
        from scipy import stats
        return 0.4 * stats.norm.pdf(x, -1, 0.5) + 0.6 * stats.norm.pdf(x, 2, 0.8)
    
    # 切片采样实现
    def slice_sample(target_pdf, n_samples, initial_x=0, width=2.0):
        """
        简单的切片采样实现
        
        Args:
            target_pdf: 目标概率密度
            n_samples: 样本数
            initial_x: 初始值
            width: 初始切片宽度
        """
        samples = []
        x = initial_x
        
        for _ in range(n_samples):
            # Step 1: 采样y ~ Uniform(0, f(x))
            y = np.random.uniform(0, target_pdf(x))
            
            # Step 2: 找到切片S = {x': f(x') > y}
            # 使用stepping out过程
            left = x - width * np.random.rand()
            right = left + width
            
            # 扩展左边界
            while target_pdf(left) > y:
                left -= width
            
            # 扩展右边界
            while target_pdf(right) > y:
                right += width
            
            # Step 3: 从切片中均匀采样
            while True:
                x_new = np.random.uniform(left, right)
                if target_pdf(x_new) > y:
                    x = x_new
                    break
                # 收缩区间
                if x_new < x:
                    left = x_new
                else:
                    right = x_new
            
            samples.append(x)
        
        return np.array(samples)
    
    # 生成样本
    n_samples = 5000
    samples = slice_sample(target_pdf, n_samples + 1000)[1000:]  # 丢弃预烧期
    
    print(f"样本均值: {np.mean(samples):.3f}")
    print(f"样本标准差: {np.std(samples):.3f}")
    
    # 理论值
    true_mean = 0.4 * (-1) + 0.6 * 2
    print(f"真实均值: {true_mean:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 样本直方图
        ax1 = axes[0, 0]
        ax1.hist(samples, bins=50, density=True, alpha=0.6,
                color='green', label='切片采样')
        x = np.linspace(-4, 5, 1000)
        y = [target_pdf(xi) for xi in x]
        ax1.plot(x, y, 'b-', linewidth=2, label='目标分布')
        ax1.set_xlabel('值')
        ax1.set_ylabel('概率密度')
        ax1.set_title('切片采样结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 轨迹图
        ax2 = axes[0, 1]
        ax2.plot(samples[:500], 'b-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('迭代')
        ax2.set_ylabel('值')
        ax2.set_title('采样轨迹')
        ax2.grid(True, alpha=0.3)
        
        # 切片可视化
        ax3 = axes[1, 0]
        # 显示几个切片
        x_point = 1.0
        y_levels = [0.05, 0.1, 0.15, 0.2]
        
        ax3.plot(x, y, 'b-', linewidth=2, label='PDF')
        ax3.axvline(x=x_point, color='red', linestyle='--', label=f'x={x_point}')
        
        for y_level in y_levels:
            # 找到切片边界
            slice_x = x[np.array(y) > y_level]
            if len(slice_x) > 0:
                ax3.fill_between(x, 0, y_level, 
                                where=np.array(y) > y_level,
                                alpha=0.2, label=f'y={y_level}')
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('概率密度')
        ax3.set_title('切片示意图')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 自相关
        ax4 = axes[1, 1]
        lags = range(50)
        autocorr = []
        for lag in lags:
            if lag == 0:
                autocorr.append(1.0)
            else:
                c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
                autocorr.append(c)
        
        ax4.bar(lags, autocorr, alpha=0.7, color='blue')
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('滞后')
        ax4.set_ylabel('自相关')
        ax4.set_title('自相关函数')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('切片采样', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 切片采样自动调节步长")
    print("2. 不需要调节参数")
    print("3. 总是接受新样本")
    print("4. 适合单峰或多峰分布")


def compare_mcmc_methods(show_plot: bool = True) -> None:
    """
    比较不同MCMC方法
    """
    print("\n不同MCMC方法比较")
    print("=" * 60)
    
    # 目标分布：标准正态
    def target_log_pdf(x):
        return -0.5 * x**2
    
    def grad_log_pdf(x):
        return -x
    
    n_samples = 5000
    
    # 1. Random Walk Metropolis
    mh_rw = MetropolisHastings(
        target_log_pdf=target_log_pdf,
        proposal_sampler=lambda x: x + np.random.normal(0, 0.5)
    )
    samples_mh = mh_rw.sample(n_samples, burn_in=500, random_state=42)
    ess_mh = compute_effective_sample_size(samples_mh)
    
    # 2. HMC
    hmc = HamiltonianMC(
        target_log_pdf=target_log_pdf,
        grad_log_pdf=grad_log_pdf,
        step_size=0.1,
        n_leapfrog=10
    )
    samples_hmc = hmc.sample(n_samples, burn_in=500, random_state=42)
    ess_hmc = compute_effective_sample_size(samples_hmc)
    
    # 3. Gibbs (对于1D退化为直接采样)
    # 使用2D示例
    def sample_2d_gaussian(state, dim):
        # 2D高斯的条件分布
        if dim == 0:
            return np.random.normal(0.5 * state[1], np.sqrt(0.75))
        else:
            return np.random.normal(0.5 * state[0], np.sqrt(0.75))
    
    gibbs = GibbsSampler(
        conditional_samplers=[sample_2d_gaussian, sample_2d_gaussian]
    )
    samples_gibbs = gibbs.sample(n_samples, burn_in=500, random_state=42)
    ess_gibbs = compute_effective_sample_size(samples_gibbs[:, 0])
    
    print("方法比较：")
    print("┌──────────────┬──────────┬──────────┬──────────┐")
    print("│ 方法         │ ESS/n    │ 接受率   │ 调参难度 │")
    print("├──────────────┼──────────┼──────────┼──────────┤")
    print(f"│ Random Walk  │ {ess_mh/n_samples:.3f}    │ {mh_rw.get_acceptance_rate():.3f}    │ 中等     │")
    print(f"│ HMC          │ {ess_hmc/n_samples:.3f}    │ {hmc.get_acceptance_rate():.3f}    │ 困难     │")
    print(f"│ Gibbs        │ {ess_gibbs/n_samples:.3f}    │ 1.000    │ 简单     │")
    print("└──────────────┴──────────┴──────────┴──────────┘")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 轨迹比较
        ax1 = axes[0, 0]
        ax1.plot(samples_mh[:200], 'b-', linewidth=1, alpha=0.7, label='MH')
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('值')
        ax1.set_title('Random Walk MH')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(samples_hmc[:200], 'r-', linewidth=1, alpha=0.7, label='HMC')
        ax2.set_xlabel('迭代')
        ax2.set_ylabel('值')
        ax2.set_title('HMC')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        ax3.plot(samples_gibbs[:200, 0], 'g-', linewidth=1, alpha=0.7, label='Gibbs')
        ax3.set_xlabel('迭代')
        ax3.set_ylabel('值')
        ax3.set_title('Gibbs')
        ax3.grid(True, alpha=0.3)
        
        # 自相关比较
        ax4 = axes[1, 0]
        lags = range(30)
        
        for samples, label, color in [
            (samples_mh, 'MH', 'blue'),
            (samples_hmc, 'HMC', 'red'),
            (samples_gibbs[:, 0], 'Gibbs', 'green')
        ]:
            autocorr = []
            for lag in lags:
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
                    autocorr.append(c)
            ax4.plot(lags, autocorr, linewidth=2, label=label, color=color, marker='o', markersize=4)
        
        ax4.axhline(y=0, color='gray', linestyle='--')
        ax4.set_xlabel('滞后')
        ax4.set_ylabel('自相关')
        ax4.set_title('自相关比较')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ESS比较（条形图）
        ax5 = axes[1, 1]
        methods = ['MH', 'HMC', 'Gibbs']
        ess_values = [ess_mh, ess_hmc, ess_gibbs]
        colors = ['blue', 'red', 'green']
        
        bars = ax5.bar(methods, ess_values, color=colors, alpha=0.7)
        ax5.set_ylabel('有效样本大小')
        ax5.set_title('ESS比较')
        ax5.axhline(y=n_samples, color='gray', linestyle='--', label=f'总样本数={n_samples}')
        
        # 添加数值标签
        for bar, ess in zip(bars, ess_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ess:.0f}\n({ess/n_samples:.2%})',
                    ha='center', va='bottom')
        
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 方法特点总结
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = """
        MCMC方法选择指南：
        
        Random Walk MH:
        • 简单通用
        • 高维效率低
        • 需调节步长
        
        HMC:
        • 高维高效
        • 需要梯度
        • 参数敏感
        
        Gibbs:
        • 无需调参
        • 需条件分布
        • 高相关性时慢
        
        切片采样:
        • 自适应
        • 单变量高效
        • 实现复杂
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('MCMC方法比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n选择建议：")
    print("1. 低维简单分布：Random Walk MH")
    print("2. 高维连续分布：HMC")
    print("3. 条件分布已知：Gibbs")
    print("4. 单维复杂分布：切片采样")
    print("5. 实践中常组合使用")