"""
13.2 隐马尔可夫模型 (Hidden Markov Model, HMM)
===============================================

HMM是用于建模序列数据的概率模型，假设观测由隐藏的马尔可夫链生成。

模型参数：
- π: 初始状态概率
- A: 状态转移矩阵 A[i,j] = P(z_{t+1}=j|z_t=i)
- B: 发射概率 B[j,k] = P(x_t=k|z_t=j)

三个基本问题：
1. 评估问题：给定模型和观测序列，计算概率P(X|λ)
   解法：前向算法或后向算法

2. 解码问题：给定模型和观测序列，找最可能的状态序列
   解法：Viterbi算法

3. 学习问题：给定观测序列，估计模型参数
   解法：Baum-Welch算法（EM算法的特例）

前向算法：
α_t(i) = P(x_1,...,x_t, z_t=i|λ)
递推：α_{t+1}(j) = [Σ_i α_t(i)A_{ij}] B_j(x_{t+1})

后向算法：
β_t(i) = P(x_{t+1},...,x_T|z_t=i, λ)
递推：β_t(i) = Σ_j A_{ij}B_j(x_{t+1})β_{t+1}(j)

Baum-Welch算法：
E步：计算γ_t(i) = P(z_t=i|X,λ) 和 ξ_t(i,j) = P(z_t=i,z_{t+1}=j|X,λ)
M步：更新π, A, B

应用：
- 语音识别
- 词性标注
- 生物序列分析
- 金融时间序列
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class HiddenMarkovModel:
    """
    隐马尔可夫模型
    
    处理离散观测的HMM。
    """
    
    def __init__(self, n_states: int, n_observations: int,
                 init_method: str = 'random',
                 random_state: Optional[int] = None):
        """
        初始化HMM
        
        Args:
            n_states: 隐状态数
            n_observations: 观测符号数
            init_method: 初始化方法 ('random', 'uniform')
            random_state: 随机种子
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.init_method = init_method
        self.random_state = random_state
        
        # 模型参数
        self.pi = None  # 初始状态概率
        self.A = None   # 状态转移矩阵
        self.B = None   # 发射概率矩阵
        
        # 初始化参数
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """初始化模型参数"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init_method == 'random':
            # 随机初始化
            self.pi = np.random.dirichlet(np.ones(self.n_states))
            
            self.A = np.random.dirichlet(np.ones(self.n_states), 
                                        size=self.n_states)
            
            self.B = np.random.dirichlet(np.ones(self.n_observations),
                                        size=self.n_states)
        else:  # uniform
            # 均匀初始化
            self.pi = np.ones(self.n_states) / self.n_states
            self.A = np.ones((self.n_states, self.n_states)) / self.n_states
            self.B = np.ones((self.n_states, self.n_observations)) / self.n_observations
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        前向算法
        
        计算P(X|λ)和前向变量α。
        
        Args:
            X: 观测序列，shape (T,)
            
        Returns:
            (alpha, log_likelihood): 前向变量和对数似然
        """
        T = len(X)
        alpha = np.zeros((T, self.n_states))
        
        # 初始化
        alpha[0] = self.pi * self.B[:, X[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, X[t]]
        
        # 终止
        likelihood = np.sum(alpha[-1])
        log_likelihood = np.log(likelihood + 1e-10)
        
        return alpha, log_likelihood
    
    def backward(self, X: np.ndarray) -> np.ndarray:
        """
        后向算法
        
        计算后向变量β。
        
        Args:
            X: 观测序列
            
        Returns:
            beta: 后向变量
        """
        T = len(X)
        beta = np.zeros((T, self.n_states))
        
        # 初始化
        beta[-1] = 1.0
        
        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, X[t+1]] * beta[t+1])
        
        return beta
    
    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi算法
        
        找到最可能的状态序列。
        
        Args:
            X: 观测序列
            
        Returns:
            (path, log_prob): 最优路径和对数概率
        """
        T = len(X)
        
        # δ[t,i] = max P(z_1,...,z_{t-1}, z_t=i, x_1,...,x_t)
        delta = np.zeros((T, self.n_states))
        # ψ[t,i] = argmax_{z_{t-1}} P(z_1,...,z_{t-1}, z_t=i, x_1,...,x_t)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化（对数空间避免下溢）
        delta[0] = np.log(self.pi + 1e-10) + np.log(self.B[:, X[0]] + 1e-10)
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                # 找最大的前驱状态
                temp = delta[t-1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = temp[psi[t, j]] + np.log(self.B[j, X[t]] + 1e-10)
        
        # 终止
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        log_prob = delta[-1, path[-1]]
        
        # 回溯
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, log_prob
    
    def baum_welch(self, sequences: List[np.ndarray],
                   max_iter: int = 100,
                   tol: float = 1e-3) -> 'HiddenMarkovModel':
        """
        Baum-Welch算法（EM算法）
        
        从观测序列学习HMM参数。
        
        Args:
            sequences: 观测序列列表
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            self
        """
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # E步：计算期望统计量
            gamma_sum = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            gamma_obs_sum = np.zeros((self.n_states, self.n_observations))
            
            total_ll = 0
            
            for X in sequences:
                T = len(X)
                
                # 前向-后向算法
                alpha, ll = self.forward(X)
                beta = self.backward(X)
                total_ll += ll
                
                # γ_t(i) = P(z_t=i|X, λ)
                gamma = alpha * beta
                gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
                
                # ξ_t(i,j) = P(z_t=i, z_{t+1}=j|X, λ)
                xi = np.zeros((T-1, self.n_states, self.n_states))
                for t in range(T-1):
                    denominator = np.sum(alpha[t] * beta[t])
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = alpha[t, i] * self.A[i, j] * \
                                         self.B[j, X[t+1]] * beta[t+1, j] / \
                                         (denominator + 1e-10)
                
                # 累积统计量
                gamma_sum += gamma[0]  # 用于更新π
                xi_sum += np.sum(xi, axis=0)  # 用于更新A
                
                # 用于更新B
                for t in range(T):
                    gamma_obs_sum[:, X[t]] += gamma[t]
            
            # M步：更新参数
            # 更新初始概率
            self.pi = gamma_sum / len(sequences)
            
            # 更新转移矩阵
            for i in range(self.n_states):
                self.A[i] = xi_sum[i] / (np.sum(xi_sum[i]) + 1e-10)
            
            # 更新发射概率
            for j in range(self.n_states):
                self.B[j] = gamma_obs_sum[j] / (np.sum(gamma_obs_sum[j]) + 1e-10)
            
            # 检查收敛
            avg_ll = total_ll / len(sequences)
            if abs(avg_ll - prev_ll) < tol:
                print(f"Baum-Welch收敛于{iteration+1}次迭代")
                break
            prev_ll = avg_ll
        
        return self
    
    def sample(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        从HMM生成序列
        
        Args:
            length: 序列长度
            
        Returns:
            (states, observations): 状态序列和观测序列
        """
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        
        # 初始状态
        states[0] = np.random.choice(self.n_states, p=self.pi)
        observations[0] = np.random.choice(self.n_observations, 
                                         p=self.B[states[0]])
        
        # 生成序列
        for t in range(1, length):
            states[t] = np.random.choice(self.n_states, 
                                        p=self.A[states[t-1]])
            observations[t] = np.random.choice(self.n_observations,
                                             p=self.B[states[t]])
        
        return states, observations
    
    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """
        预测下一个观测
        
        Args:
            X: 观测序列
            
        Returns:
            下一个观测的概率分布
        """
        # 使用前向算法计算当前状态分布
        alpha, _ = self.forward(X)
        
        # 当前状态分布
        state_prob = alpha[-1] / np.sum(alpha[-1])
        
        # 下一个状态分布
        next_state_prob = state_prob @ self.A
        
        # 下一个观测分布
        next_obs_prob = next_state_prob @ self.B
        
        return next_obs_prob


class GaussianHMM:
    """
    高斯HMM
    
    处理连续观测的HMM，每个状态的发射概率是高斯分布。
    """
    
    def __init__(self, n_states: int, n_features: int,
                 covariance_type: str = 'diag',
                 random_state: Optional[int] = None):
        """
        初始化高斯HMM
        
        Args:
            n_states: 隐状态数
            n_features: 特征维度
            covariance_type: 协方差类型 ('spherical', 'diag', 'full')
            random_state: 随机种子
        """
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # 模型参数
        self.pi = None  # 初始状态概率
        self.A = None   # 状态转移矩阵
        self.means = None  # 各状态的均值
        self.covars = None  # 各状态的协方差
        
        # 初始化
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """初始化参数"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始概率和转移矩阵
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        self.A = np.random.dirichlet(np.ones(self.n_states), 
                                    size=self.n_states)
        
        # 高斯参数
        self.means = np.random.randn(self.n_states, self.n_features)
        
        if self.covariance_type == 'spherical':
            self.covars = np.ones(self.n_states)
        elif self.covariance_type == 'diag':
            self.covars = np.ones((self.n_states, self.n_features))
        else:  # full
            self.covars = np.array([np.eye(self.n_features) 
                                   for _ in range(self.n_states)])
    
    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        计算每个状态的对数似然
        
        Args:
            X: 观测序列，shape (T, n_features)
            
        Returns:
            对数似然矩阵，shape (T, n_states)
        """
        T = len(X)
        log_likelihood = np.zeros((T, self.n_states))
        
        for j in range(self.n_states):
            if self.covariance_type == 'spherical':
                cov = self.covars[j] * np.eye(self.n_features)
            elif self.covariance_type == 'diag':
                cov = np.diag(self.covars[j])
            else:
                cov = self.covars[j]
            
            # 计算多元高斯概率密度
            from scipy.stats import multivariate_normal
            log_likelihood[:, j] = multivariate_normal.logpdf(
                X, self.means[j], cov
            )
        
        return log_likelihood
    
    def forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        前向-后向算法（对数空间）
        
        Args:
            X: 观测序列
            
        Returns:
            (log_alpha, log_beta, log_likelihood): 前向、后向变量和对数似然
        """
        T = len(X)
        log_likelihood = self._compute_log_likelihood(X)
        
        # 前向（对数空间）
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = np.log(self.pi + 1e-10) + log_likelihood[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                temp = log_alpha[t-1] + np.log(self.A[:, j] + 1e-10)
                log_alpha[t, j] = self._log_sum_exp(temp) + log_likelihood[t, j]
        
        # 后向（对数空间）
        log_beta = np.zeros((T, self.n_states))
        log_beta[-1] = 0  # log(1) = 0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                temp = np.log(self.A[i] + 1e-10) + log_likelihood[t+1] + log_beta[t+1]
                log_beta[t, i] = self._log_sum_exp(temp)
        
        # 总对数似然
        total_log_likelihood = self._log_sum_exp(log_alpha[-1])
        
        return log_alpha, log_beta, total_log_likelihood
    
    def _log_sum_exp(self, x: np.ndarray) -> float:
        """计算log(sum(exp(x)))，避免数值溢出"""
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))


def demonstrate_discrete_hmm(show_plot: bool = True) -> None:
    """
    演示离散HMM
    """
    print("\n离散HMM演示 - 天气预测")
    print("=" * 60)
    
    # 定义HMM：隐状态是天气，观测是活动
    # 状态：0=晴天, 1=阴天, 2=雨天
    # 观测：0=散步, 1=购物, 2=清洁, 3=读书
    
    hmm = HiddenMarkovModel(n_states=3, n_observations=4, random_state=42)
    
    # 设置合理的参数
    hmm.pi = np.array([0.6, 0.3, 0.1])  # 初始状态概率
    
    # 状态转移矩阵
    hmm.A = np.array([
        [0.7, 0.2, 0.1],  # 晴天 -> 
        [0.3, 0.4, 0.3],  # 阴天 ->
        [0.2, 0.3, 0.5]   # 雨天 ->
    ])
    
    # 发射概率
    hmm.B = np.array([
        [0.4, 0.3, 0.1, 0.2],  # 晴天时的活动概率
        [0.2, 0.4, 0.2, 0.2],  # 阴天时的活动概率
        [0.1, 0.2, 0.3, 0.4]   # 雨天时的活动概率
    ])
    
    # 生成序列
    length = 50
    true_states, observations = hmm.sample(length)
    
    print(f"生成序列长度: {length}")
    
    # Viterbi解码
    decoded_states, log_prob = hmm.viterbi(observations)
    
    # 计算准确率
    accuracy = np.mean(decoded_states == true_states)
    print(f"Viterbi解码准确率: {accuracy:.3f}")
    
    # 前向算法
    alpha, ll = hmm.forward(observations)
    print(f"序列对数似然: {ll:.3f}")
    
    # 预测下一个观测
    next_obs_prob = hmm.predict_next(observations)
    next_obs = np.argmax(next_obs_prob)
    activities = ['散步', '购物', '清洁', '读书']
    print(f"预测下一个活动: {activities[next_obs]} (概率={next_obs_prob[next_obs]:.3f})")
    
    if show_plot:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # 真实状态和观测
        ax1 = axes[0, 0]
        ax1.plot(true_states, 'b-', linewidth=2, label='真实状态')
        ax1.plot(observations, 'r--', linewidth=1, alpha=0.7, label='观测')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('状态/观测')
        ax1.set_title('序列生成')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Viterbi解码
        ax2 = axes[0, 1]
        ax2.plot(true_states, 'b-', linewidth=2, label='真实状态')
        ax2.plot(decoded_states, 'g--', linewidth=2, label='Viterbi解码')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('状态')
        ax2.set_title(f'Viterbi解码 (准确率={accuracy:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 前向变量
        ax3 = axes[1, 0]
        im3 = ax3.imshow(alpha.T, aspect='auto', cmap='YlOrRd')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('状态')
        ax3.set_title('前向变量α')
        plt.colorbar(im3, ax=ax3)
        
        # 后向变量
        ax4 = axes[1, 1]
        beta = hmm.backward(observations)
        im4 = ax4.imshow(beta.T, aspect='auto', cmap='YlGnBu')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('状态')
        ax4.set_title('后向变量β')
        plt.colorbar(im4, ax=ax4)
        
        # 状态后验概率
        ax5 = axes[2, 0]
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        im5 = ax5.imshow(gamma.T, aspect='auto', cmap='viridis')
        ax5.set_xlabel('时间')
        ax5.set_ylabel('状态')
        ax5.set_title('状态后验概率γ')
        plt.colorbar(im5, ax=ax5)
        
        # 模型参数可视化
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # 显示转移矩阵
        table_data = []
        weather = ['晴', '阴', '雨']
        for i in range(3):
            row = [weather[i]]
            for j in range(3):
                row.append(f'{hmm.A[i, j]:.2f}')
            table_data.append(row)
        
        table = ax6.table(cellText=table_data,
                         colLabels=['从\\到', '晴', '阴', '雨'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.5, 1, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        ax6.text(0.5, 0.3, '状态转移矩阵', ha='center', fontsize=12, weight='bold')
        
        plt.suptitle('隐马尔可夫模型(HMM)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. HMM能够从观测推断隐状态")
    print("2. Viterbi找到最可能的状态序列")
    print("3. 前向-后向算法计算状态后验")
    print("4. 可用于序列预测")


def demonstrate_baum_welch(show_plot: bool = True) -> None:
    """
    演示Baum-Welch算法
    """
    print("\nBaum-Welch算法演示")
    print("=" * 60)
    
    # 创建真实HMM
    true_hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=42)
    
    # 设置真实参数
    true_hmm.pi = np.array([0.6, 0.4])
    true_hmm.A = np.array([[0.7, 0.3],
                          [0.4, 0.6]])
    true_hmm.B = np.array([[0.5, 0.3, 0.2],
                          [0.1, 0.4, 0.5]])
    
    # 生成训练序列
    n_sequences = 20
    sequences = []
    for _ in range(n_sequences):
        _, obs = true_hmm.sample(100)
        sequences.append(obs)
    
    print(f"训练序列: {n_sequences}条，每条长度100")
    
    # 创建学习HMM（随机初始化）
    learned_hmm = HiddenMarkovModel(n_states=2, n_observations=3, 
                                   init_method='random', random_state=123)
    
    print("\n初始参数（随机）:")
    print(f"π: {learned_hmm.pi}")
    print(f"A:\n{learned_hmm.A}")
    print(f"B:\n{learned_hmm.B}")
    
    # 学习参数
    learned_hmm.baum_welch(sequences, max_iter=50)
    
    print("\n学习后参数:")
    print(f"π: {learned_hmm.pi}")
    print(f"A:\n{learned_hmm.A}")
    print(f"B:\n{learned_hmm.B}")
    
    print("\n真实参数:")
    print(f"π: {true_hmm.pi}")
    print(f"A:\n{true_hmm.A}")
    print(f"B:\n{true_hmm.B}")
    
    # 计算参数误差
    pi_error = np.mean(np.abs(learned_hmm.pi - true_hmm.pi))
    A_error = np.mean(np.abs(learned_hmm.A - true_hmm.A))
    B_error = np.mean(np.abs(learned_hmm.B - true_hmm.B))
    
    print(f"\n参数误差:")
    print(f"  π误差: {pi_error:.4f}")
    print(f"  A误差: {A_error:.4f}")
    print(f"  B误差: {B_error:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 初始概率比较
        ax1 = axes[0, 0]
        x = np.arange(2)
        width = 0.35
        ax1.bar(x - width/2, true_hmm.pi, width, label='真实', alpha=0.7)
        ax1.bar(x + width/2, learned_hmm.pi, width, label='学习', alpha=0.7)
        ax1.set_xlabel('状态')
        ax1.set_ylabel('概率')
        ax1.set_title('初始概率π')
        ax1.set_xticks(x)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 转移矩阵比较
        ax2 = axes[0, 1]
        im2 = ax2.imshow(true_hmm.A, cmap='Blues', vmin=0, vmax=1)
        ax2.set_title('真实转移矩阵A')
        ax2.set_xlabel('到状态')
        ax2.set_ylabel('从状态')
        plt.colorbar(im2, ax=ax2)
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f'{true_hmm.A[i, j]:.2f}',
                        ha='center', va='center')
        
        ax3 = axes[0, 2]
        im3 = ax3.imshow(learned_hmm.A, cmap='Greens', vmin=0, vmax=1)
        ax3.set_title('学习转移矩阵A')
        ax3.set_xlabel('到状态')
        ax3.set_ylabel('从状态')
        plt.colorbar(im3, ax=ax3)
        
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, f'{learned_hmm.A[i, j]:.2f}',
                        ha='center', va='center')
        
        # 发射概率比较
        ax4 = axes[1, 0]
        im4 = ax4.imshow(true_hmm.B, cmap='Blues', vmin=0, vmax=1)
        ax4.set_title('真实发射概率B')
        ax4.set_xlabel('观测')
        ax4.set_ylabel('状态')
        plt.colorbar(im4, ax=ax4)
        
        for i in range(2):
            for j in range(3):
                ax4.text(j, i, f'{true_hmm.B[i, j]:.2f}',
                        ha='center', va='center')
        
        ax5 = axes[1, 1]
        im5 = ax5.imshow(learned_hmm.B, cmap='Greens', vmin=0, vmax=1)
        ax5.set_title('学习发射概率B')
        ax5.set_xlabel('观测')
        ax5.set_ylabel('状态')
        plt.colorbar(im5, ax=ax5)
        
        for i in range(2):
            for j in range(3):
                ax5.text(j, i, f'{learned_hmm.B[i, j]:.2f}',
                        ha='center', va='center')
        
        # 参数误差
        ax6 = axes[1, 2]
        errors = [pi_error, A_error, B_error]
        labels = ['π', 'A', 'B']
        colors = ['blue', 'green', 'orange']
        ax6.bar(labels, errors, color=colors, alpha=0.7)
        ax6.set_ylabel('平均绝对误差')
        ax6.set_title('参数学习误差')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Baum-Welch算法学习HMM参数', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. Baum-Welch成功学习HMM参数")
    print("2. 参数估计接近真实值")
    print("3. 需要足够的训练数据")
    print("4. 可能收敛到局部最优")