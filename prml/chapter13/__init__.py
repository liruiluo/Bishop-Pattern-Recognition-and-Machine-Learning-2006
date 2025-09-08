"""
Chapter 13: Sequential Data (序列数据)
======================================

本章介绍处理序列数据的概率模型。

主要内容：
1. 马尔可夫模型 (13.1)
   - 马尔可夫链
   - 状态空间模型

2. 隐马尔可夫模型 (13.2)
   - 前向-后向算法
   - Viterbi算法
   - Baum-Welch算法

3. 线性动态系统 (13.3)
   - 卡尔曼滤波器
   - RTS平滑
   - 扩展卡尔曼滤波

核心概念：
序列模型处理时间相关的数据，考虑时序依赖关系。

马尔可夫假设：
未来只依赖于现在，不依赖于过去。
P(x_t|x_1,...,x_{t-1}) = P(x_t|x_{t-1})

状态空间模型：
- 隐状态演化：z_t ~ P(z_t|z_{t-1})
- 观测生成：x_t ~ P(x_t|z_t)

推理任务：
1. 滤波：P(z_t|x_{1:t})
2. 平滑：P(z_t|x_{1:T})
3. 预测：P(z_{t+k}|x_{1:t})

应用：
- 语音识别
- 自然语言处理
- 目标跟踪
- 金融预测
- 生物信息学
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 导入各节的实现
from .hmm import (
    HiddenMarkovModel,
    GaussianHMM,
    demonstrate_discrete_hmm,
    demonstrate_baum_welch
)

from .kalman_filter import (
    KalmanFilter,
    ExtendedKalmanFilter,
    demonstrate_kalman_filter,
    demonstrate_nonlinear_tracking
)


def run_chapter13(cfg: DictConfig) -> None:
    """
    运行第13章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第13章：序列数据 (Sequential Data)")
    print("="*80)
    
    # 13.2 隐马尔可夫模型
    print("\n" + "-"*60)
    print("13.2 隐马尔可夫模型 (HMM)")
    print("-"*60)
    
    # 离散HMM
    demonstrate_discrete_hmm(
        show_plot=cfg.visualization.show_plots
    )
    
    # Baum-Welch学习
    demonstrate_baum_welch(
        show_plot=cfg.visualization.show_plots
    )
    
    # 语音识别示例
    demonstrate_speech_hmm(
        show_plot=cfg.visualization.show_plots
    )
    
    # 13.3 线性动态系统
    print("\n" + "-"*60)
    print("13.3 线性动态系统 (卡尔曼滤波)")
    print("-"*60)
    
    # 标准卡尔曼滤波
    demonstrate_kalman_filter(
        show_plot=cfg.visualization.show_plots
    )
    
    # 扩展卡尔曼滤波
    demonstrate_nonlinear_tracking(
        show_plot=cfg.visualization.show_plots
    )
    
    # 粒子滤波
    demonstrate_particle_filter(
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较不同方法
    compare_sequential_models(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第13章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. HMM处理离散隐状态")
    print("2. 卡尔曼滤波处理线性高斯系统")
    print("3. EKF处理弱非线性系统")
    print("4. 粒子滤波处理任意分布")
    print("5. 平滑优于滤波")
    print("6. 选择取决于问题特性")


def demonstrate_speech_hmm(show_plot: bool = True) -> None:
    """
    演示HMM在语音识别中的应用
    """
    print("\nHMM语音识别演示（简化）")
    print("=" * 60)
    
    # 模拟简单的语音识别：识别数字"one", "two", "three"
    # 每个词用一个HMM建模
    
    # 特征：简化为4个离散特征
    # 0: 静音, 1: 低频, 2: 中频, 3: 高频
    
    # 创建三个词的HMM模型
    word_models = {}
    
    # "one"的模型（3个状态）
    hmm_one = HiddenMarkovModel(n_states=3, n_observations=4, random_state=42)
    hmm_one.pi = np.array([1.0, 0.0, 0.0])
    hmm_one.A = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])
    hmm_one.B = np.array([
        [0.1, 0.7, 0.2, 0.0],  # 开始：低频为主
        [0.1, 0.2, 0.6, 0.1],  # 中间：中频
        [0.3, 0.1, 0.1, 0.5]   # 结束：高频
    ])
    word_models['one'] = hmm_one
    
    # "two"的模型
    hmm_two = HiddenMarkovModel(n_states=3, n_observations=4, random_state=43)
    hmm_two.pi = np.array([1.0, 0.0, 0.0])
    hmm_two.A = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])
    hmm_two.B = np.array([
        [0.1, 0.2, 0.7, 0.0],  # 开始：中频
        [0.1, 0.6, 0.2, 0.1],  # 中间：低频
        [0.3, 0.2, 0.2, 0.3]   # 结束：混合
    ])
    word_models['two'] = hmm_two
    
    # "three"的模型
    hmm_three = HiddenMarkovModel(n_states=3, n_observations=4, random_state=44)
    hmm_three.pi = np.array([1.0, 0.0, 0.0])
    hmm_three.A = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])
    hmm_three.B = np.array([
        [0.1, 0.1, 0.1, 0.7],  # 开始：高频
        [0.1, 0.3, 0.5, 0.1],  # 中间：中频
        [0.2, 0.5, 0.2, 0.1]   # 结束：低频
    ])
    word_models['three'] = hmm_three
    
    # 生成测试序列
    test_words = ['one', 'two', 'three', 'one', 'three']
    test_sequences = []
    true_labels = []
    
    print("生成测试序列...")
    for word in test_words:
        _, obs = word_models[word].sample(15)  # 每个词15帧
        test_sequences.append(obs)
        true_labels.append(word)
    
    # 识别
    print("\n识别结果：")
    correct = 0
    for i, obs in enumerate(test_sequences):
        # 计算每个模型的似然
        likelihoods = {}
        for word, model in word_models.items():
            _, ll = model.forward(obs)
            likelihoods[word] = ll
        
        # 选择最大似然的词
        predicted = max(likelihoods, key=likelihoods.get)
        
        print(f"  序列{i+1}: 真实='{true_labels[i]}', 预测='{predicted}', "
              f"似然={likelihoods[predicted]:.2f}")
        
        if predicted == true_labels[i]:
            correct += 1
    
    accuracy = correct / len(test_sequences)
    print(f"\n识别准确率: {accuracy:.1%}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 显示每个词的HMM参数
        for idx, (word, model) in enumerate(word_models.items()):
            ax = axes[0, idx]
            im = ax.imshow(model.B, cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title(f'词"{word}"的发射概率')
            ax.set_xlabel('观测(特征)')
            ax.set_ylabel('状态')
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['静音', '低频', '中频', '高频'])
            plt.colorbar(im, ax=ax)
        
        # 显示测试序列和识别结果
        for idx in range(min(3, len(test_sequences))):
            ax = axes[1, idx]
            obs = test_sequences[idx]
            
            # 计算每个模型的前向变量
            alphas = {}
            for word, model in word_models.items():
                alpha, _ = model.forward(obs)
                alphas[word] = alpha
            
            # 显示最可能模型的前向变量
            likelihoods = {}
            for word, model in word_models.items():
                _, ll = model.forward(obs)
                likelihoods[word] = ll
            best_word = max(likelihoods, key=likelihoods.get)
            
            im = ax.imshow(alphas[best_word].T, aspect='auto', cmap='viridis')
            ax.set_title(f'序列{idx+1}: {true_labels[idx]}→{best_word}')
            ax.set_xlabel('时间(帧)')
            ax.set_ylabel('状态')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('HMM语音识别（简化演示）', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 每个词用一个HMM建模")
    print("2. 识别通过比较似然实现")
    print("3. 实际语音识别使用连续特征")
    print("4. 需要大量训练数据")


class ParticleFilter:
    """
    粒子滤波器
    
    使用蒙特卡罗方法处理非线性非高斯系统。
    """
    
    def __init__(self, n_particles: int,
                 state_dim: int,
                 f: callable,
                 h: callable,
                 process_noise: callable,
                 measurement_likelihood: callable):
        """
        初始化粒子滤波器
        
        Args:
            n_particles: 粒子数
            state_dim: 状态维度
            f: 状态转移函数
            h: 观测函数
            process_noise: 过程噪声采样函数
            measurement_likelihood: 测量似然函数
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.f = f
        self.h = h
        self.process_noise = process_noise
        self.measurement_likelihood = measurement_likelihood
        
        # 粒子和权重
        self.particles = None
        self.weights = None
        
    def initialize(self, initial_distribution: callable) -> None:
        """初始化粒子"""
        self.particles = initial_distribution(self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict(self) -> None:
        """预测步：传播粒子"""
        for i in range(self.n_particles):
            self.particles[i] = self.f(self.particles[i]) + self.process_noise()
    
    def update(self, y: np.ndarray) -> None:
        """更新步：更新权重"""
        # 计算似然
        for i in range(self.n_particles):
            self.weights[i] *= self.measurement_likelihood(y, self.particles[i])
        
        # 归一化权重
        self.weights /= np.sum(self.weights)
        
        # 重采样（如果有效样本数太低）
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n_particles / 2:
            self.resample()
    
    def resample(self) -> None:
        """系统重采样"""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate(self) -> np.ndarray:
        """计算加权平均估计"""
        return np.average(self.particles, weights=self.weights, axis=0)


def demonstrate_particle_filter(show_plot: bool = True) -> None:
    """
    演示粒子滤波
    """
    print("\n粒子滤波演示 - 非线性非高斯跟踪")
    print("=" * 60)
    
    # 非线性系统
    dt = 0.1
    
    def f(x):
        # 非线性状态转移
        return np.array([
            x[0] + dt * x[2],
            x[1] + dt * x[3],
            x[2] + 0.1 * np.sin(x[0]),  # 非线性
            x[3] - 0.1 * np.cos(x[1])   # 非线性
        ])
    
    def h(x):
        # 非线性观测
        return np.array([
            np.sqrt(x[0]**2 + x[1]**2),  # 距离
            np.arctan2(x[1], x[0])        # 角度
        ])
    
    def process_noise():
        return 0.1 * np.random.randn(4)
    
    def measurement_likelihood(y, x):
        # 计算测量似然
        y_pred = h(x)
        diff = y - y_pred
        # 处理角度环绕
        diff[1] = np.arctan2(np.sin(diff[1]), np.cos(diff[1]))
        return np.exp(-0.5 * np.sum(diff**2 / np.array([0.1, 0.01])))
    
    # 创建粒子滤波器
    pf = ParticleFilter(
        n_particles=1000,
        state_dim=4,
        f=f,
        h=h,
        process_noise=process_noise,
        measurement_likelihood=measurement_likelihood
    )
    
    # 初始化
    def initial_distribution(n):
        particles = np.zeros((n, 4))
        particles[:, 0] = 10 + 0.5 * np.random.randn(n)
        particles[:, 1] = 0 + 0.5 * np.random.randn(n)
        particles[:, 2] = 0 + 0.1 * np.random.randn(n)
        particles[:, 3] = 1 + 0.1 * np.random.randn(n)
        return particles
    
    pf.initialize(initial_distribution)
    
    # 生成数据
    T = 100
    true_states = []
    observations = []
    pf_estimates = []
    
    true_state = np.array([10, 0, 0, 1])
    
    np.random.seed(42)
    for t in range(T):
        # 真实状态
        true_state = f(true_state) + process_noise()
        true_states.append(true_state)
        
        # 观测
        obs = h(true_state) + np.array([
            0.3 * np.random.randn(),
            0.1 * np.random.randn()
        ])
        observations.append(obs)
        
        # 粒子滤波
        pf.predict()
        pf.update(obs)
        pf_estimates.append(pf.estimate())
    
    true_states = np.array(true_states)
    pf_estimates = np.array(pf_estimates)
    
    # 计算误差
    pf_error = np.mean(np.linalg.norm(
        pf_estimates[:, :2] - true_states[:, :2], axis=1
    ))
    
    print(f"粒子滤波位置误差: {pf_error:.3f}")
    print(f"粒子数: {pf.n_particles}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 2D轨迹
        ax1 = axes[0, 0]
        ax1.plot(true_states[:, 0], true_states[:, 1], 'b-',
                linewidth=2, label='真实轨迹')
        ax1.plot(pf_estimates[:, 0], pf_estimates[:, 1], 'r--',
                linewidth=2, label='粒子滤波估计')
        
        # 显示最后时刻的粒子
        ax1.scatter(pf.particles[:, 0], pf.particles[:, 1],
                   c='gray', s=1, alpha=0.3, label='粒子')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('粒子滤波跟踪')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 位置误差
        ax2 = axes[0, 1]
        time = np.arange(T) * dt
        errors = np.linalg.norm(
            pf_estimates[:, :2] - true_states[:, :2], axis=1
        )
        ax2.plot(time, errors, 'r-', linewidth=2)
        ax2.set_xlabel('时间')
        ax2.set_ylabel('位置误差')
        ax2.set_title('估计误差')
        ax2.grid(True, alpha=0.3)
        
        # X坐标跟踪
        ax3 = axes[1, 0]
        ax3.plot(time, true_states[:, 0], 'b-', label='真实')
        ax3.plot(time, pf_estimates[:, 0], 'r--', label='粒子滤波')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('X位置')
        ax3.set_title('X坐标跟踪')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 权重分布
        ax4 = axes[1, 1]
        ax4.hist(pf.weights, bins=50, alpha=0.7, color='blue')
        ax4.set_xlabel('权重')
        ax4.set_ylabel('粒子数')
        ax4.set_title(f'权重分布 (ESS={1.0/np.sum(pf.weights**2):.1f})')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('粒子滤波 - 非线性非高斯系统', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 粒子滤波处理任意非线性")
    print("2. 不需要高斯假设")
    print("3. 计算量随粒子数增加")
    print("4. 可能退化（粒子贫化）")


def compare_sequential_models(show_plot: bool = True) -> None:
    """
    比较不同的序列模型
    """
    print("\n序列模型比较")
    print("=" * 60)
    
    print("\n方法比较：")
    print("┌────────────┬────────────┬────────────┬────────────┐")
    print("│ 方法       │ 状态空间   │ 观测空间   │ 适用场景   │")
    print("├────────────┼────────────┼────────────┼────────────┤")
    print("│ HMM        │ 离散       │ 离散/连续  │ 语音/NLP   │")
    print("│ 卡尔曼     │ 连续高斯   │ 连续高斯   │ 线性跟踪   │")
    print("│ EKF        │ 连续高斯   │ 连续       │ 弱非线性   │")
    print("│ 粒子滤波   │ 任意       │ 任意       │ 强非线性   │")
    print("└────────────┴────────────┴────────────┴────────────┘")
    
    print("\n计算复杂度：")
    print("- HMM: O(TN²) - N是状态数")
    print("- 卡尔曼: O(Td³) - d是状态维度")
    print("- 粒子滤波: O(TNp) - Np是粒子数")
    
    if show_plot:
        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 准确度vs计算量
        ax1 = axes[0, 0]
        methods = ['HMM', '卡尔曼', 'EKF', '粒子滤波']
        accuracy = [0.7, 0.9, 0.85, 0.95]
        computation = [0.3, 0.4, 0.5, 0.9]
        
        ax1.scatter(computation, accuracy, s=200, alpha=0.6)
        for i, txt in enumerate(methods):
            ax1.annotate(txt, (computation[i], accuracy[i]),
                        ha='center', va='center')
        
        ax1.set_xlabel('计算复杂度（相对）')
        ax1.set_ylabel('精度（相对）')
        ax1.set_title('精度 vs 计算量')
        ax1.grid(True, alpha=0.3)
        
        # 适用范围
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        # 创建文本表格
        table_data = [
            ['场景', '推荐方法'],
            ['线性高斯', '卡尔曼滤波'],
            ['弱非线性', 'EKF/UKF'],
            ['强非线性', '粒子滤波'],
            ['离散状态', 'HMM'],
            ['大状态空间', '变分方法']
        ]
        
        table = ax2.table(cellText=table_data,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        for i in range(6):
            if i == 0:
                table[(i, 0)].set_facecolor('#40466e')
                table[(i, 1)].set_facecolor('#40466e')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('应用场景选择')
        
        # 收敛性
        ax3 = axes[1, 0]
        T = 100
        t = np.arange(T)
        
        # 模拟收敛曲线
        kf_error = 0.5 * np.exp(-0.1 * t) + 0.05
        pf_error = 0.3 * np.exp(-0.05 * t) + 0.08
        hmm_error = 0.4 * np.exp(-0.08 * t) + 0.1
        
        ax3.plot(t, kf_error, 'b-', label='卡尔曼')
        ax3.plot(t, pf_error, 'r-', label='粒子滤波')
        ax3.plot(t, hmm_error, 'g-', label='HMM')
        
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('估计误差')
        ax3.set_title('收敛速度比较')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 鲁棒性
        ax4 = axes[1, 1]
        noise_levels = np.linspace(0, 1, 50)
        
        # 模拟鲁棒性
        kf_robust = 1 / (1 + 10 * noise_levels**2)
        pf_robust = 1 / (1 + 3 * noise_levels)
        ekf_robust = 1 / (1 + 8 * noise_levels**1.5)
        
        ax4.plot(noise_levels, kf_robust, 'b-', label='卡尔曼')
        ax4.plot(noise_levels, pf_robust, 'r-', label='粒子滤波')
        ax4.plot(noise_levels, ekf_robust, 'g-', label='EKF')
        
        ax4.set_xlabel('噪声水平')
        ax4.set_ylabel('性能')
        ax4.set_title('鲁棒性比较')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('序列模型综合比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n选择建议：")
    print("1. 线性高斯 → 卡尔曼滤波")
    print("2. 离散状态 → HMM")
    print("3. 弱非线性 → EKF")
    print("4. 强非线性/非高斯 → 粒子滤波")
    print("5. 计算资源有限 → 简单方法")
    print("6. 精度要求高 → 复杂方法")