"""
2.4 指数族分布 (The Exponential Family)
========================================

指数族是一个统一的分布框架，包含了机器学习中的大多数常用分布。

一般形式：
p(x|η) = h(x) * g(η) * exp(η^T * u(x))

其中：
- η: 自然参数 (natural parameters)
- u(x): 充分统计量 (sufficient statistics)
- h(x): 基础测度
- g(η): 归一化因子，确保概率积分为1

等价形式（对数形式）：
ln p(x|η) = ln h(x) + ln g(η) + η^T * u(x)

或者使用对数配分函数A(η)：
p(x|η) = h(x) * exp(η^T * u(x) - A(η))
其中 A(η) = -ln g(η) 是对数配分函数

为什么指数族如此重要？
1. 统一框架：许多分布都是指数族成员
2. 充分统计量：数据压缩不损失信息
3. 共轭先验：指数族分布都有共轭先验
4. 凸性质：对数似然是凹函数，保证全局最优
5. 计算便利：许多性质有封闭形式解

指数族成员包括：
- 伯努利分布、二项分布
- 多项分布
- 高斯分布
- 泊松分布
- 伽马分布、贝塔分布
- 狄利克雷分布
等等...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, loggamma, digamma
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')


class ExponentialFamily:
    """
    指数族分布的基类
    
    这个类定义了指数族分布的通用接口和性质。
    具体的分布（如高斯、伯努利等）继承这个类。
    """
    
    def __init__(self, name: str):
        """
        初始化指数族分布
        
        Args:
            name: 分布名称
        """
        self.name = name
    
    def natural_parameters(self, theta: Any) -> np.ndarray:
        """
        从标准参数θ计算自然参数η
        
        标准参数是我们通常使用的参数（如高斯的μ和σ²），
        自然参数是指数族形式中的参数。
        
        Args:
            theta: 标准参数
            
        Returns:
            自然参数η
        """
        raise NotImplementedError
    
    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        计算充分统计量u(x)
        
        充分统计量包含了数据中关于参数的所有信息。
        这是数据降维的关键：无论数据量多大，
        充分统计量的维度是固定的。
        
        Args:
            x: 数据
            
        Returns:
            充分统计量
        """
        raise NotImplementedError
    
    def log_partition(self, eta: np.ndarray) -> float:
        """
        对数配分函数A(η)
        
        这个函数确保概率归一化。
        它的导数给出期望，二阶导数给出方差。
        
        Args:
            eta: 自然参数
            
        Returns:
            对数配分函数值
        """
        raise NotImplementedError
    
    def base_measure(self, x: np.ndarray) -> float:
        """
        基础测度h(x)
        
        不依赖于参数的部分。
        
        Args:
            x: 数据点
            
        Returns:
            基础测度值
        """
        raise NotImplementedError
    
    def moments_from_natural(self, eta: np.ndarray) -> Dict[str, Any]:
        """
        从自然参数计算矩（期望和方差）
        
        利用对数配分函数的性质：
        E[u(x)] = ∇A(η)  （一阶导数）
        Cov[u(x)] = ∇²A(η)  （二阶导数）
        
        Args:
            eta: 自然参数
            
        Returns:
            包含期望和方差的字典
        """
        raise NotImplementedError


class BernoulliExponential(ExponentialFamily):
    """
    伯努利分布的指数族形式
    
    标准形式：p(x|μ) = μ^x * (1-μ)^(1-x)
    
    指数族形式：
    p(x|η) = exp(η*x - ln(1 + exp(η)))
    
    其中：
    - 自然参数：η = ln(μ/(1-μ))  （对数几率）
    - 充分统计量：u(x) = x
    - 对数配分函数：A(η) = ln(1 + exp(η))
    - 基础测度：h(x) = 1
    
    这个形式在逻辑回归中非常重要！
    """
    
    def __init__(self):
        super().__init__("Bernoulli")
    
    def natural_parameters(self, mu: float) -> float:
        """
        从成功概率μ计算自然参数η
        
        η = ln(μ/(1-μ)) 是对数几率(log-odds)
        这就是逻辑回归中的logit函数！
        """
        if not 0 < mu < 1:
            raise ValueError("μ必须在(0,1)之间")
        return np.log(mu / (1 - mu))
    
    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """充分统计量就是x本身"""
        return x
    
    def log_partition(self, eta: float) -> float:
        """
        对数配分函数
        A(η) = ln(1 + exp(η)) = softplus(η)
        """
        # 使用数值稳定的方法
        return np.log(1 + np.exp(eta)) if eta < 10 else eta
    
    def base_measure(self, x: np.ndarray) -> float:
        """基础测度为1"""
        return 1.0
    
    def moments_from_natural(self, eta: float) -> Dict[str, float]:
        """
        从自然参数计算矩
        
        E[x] = dA/dη = exp(η)/(1 + exp(η)) = σ(η)
        这就是sigmoid函数！
        
        Var[x] = d²A/dη² = σ(η)(1 - σ(η))
        """
        sigmoid = 1 / (1 + np.exp(-eta))  # σ(η)
        return {
            'mean': sigmoid,
            'variance': sigmoid * (1 - sigmoid)
        }
    
    def demonstrate(self) -> None:
        """演示伯努利分布的指数族性质"""
        print(f"\n{self.name}分布的指数族形式")
        print("=" * 50)
        
        mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        print("参数转换：")
        print("-" * 40)
        print(f"{'μ':<10} {'η=ln(μ/(1-μ))':<15} {'恢复的μ':<10}")
        print("-" * 40)
        
        for mu in mu_values:
            eta = self.natural_parameters(mu)
            # 从自然参数恢复原参数
            mu_recovered = 1 / (1 + np.exp(-eta))
            print(f"{mu:<10.2f} {eta:<15.3f} {mu_recovered:<10.3f}")
        
        print("\n观察：")
        print("- μ=0.5时，η=0（对称点）")
        print("- μ→0时，η→-∞")
        print("- μ→1时，η→+∞")
        print("- 这就是逻辑回归的基础！")


class GaussianExponential(ExponentialFamily):
    """
    高斯分布的指数族形式
    
    标准形式：N(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    
    指数族形式（已知方差）：
    p(x|η) = (1/√(2πσ²)) * exp(η*x - η²σ²/2)
    
    其中：
    - 自然参数：η = μ/σ²
    - 充分统计量：u(x) = x
    - 对数配分函数：A(η) = η²σ²/2
    
    完整的指数族形式（未知均值和方差）：
    - 自然参数：η = [μ/σ², -1/(2σ²)]^T
    - 充分统计量：u(x) = [x, x²]^T
    - 这是2维的指数族！
    
    高斯分布在指数族中的特殊地位：
    1. 充分统计量是x和x²（一阶和二阶矩）
    2. 自然参数与精度有关（不是方差）
    3. 共轭先验是高斯-逆伽马分布
    """
    
    def __init__(self, known_variance: bool = False, sigma2: float = 1.0):
        """
        初始化高斯分布
        
        Args:
            known_variance: 方差是否已知
            sigma2: 已知的方差值（如果known_variance=True）
        """
        super().__init__("Gaussian")
        self.known_variance = known_variance
        self.sigma2 = sigma2
    
    def natural_parameters(self, theta: Tuple[float, Optional[float]]) -> np.ndarray:
        """
        从标准参数(μ, σ²)计算自然参数
        
        如果方差已知：η = μ/σ²
        如果方差未知：η = [μ/σ², -1/(2σ²)]^T
        """
        if self.known_variance:
            mu = theta[0] if isinstance(theta, tuple) else theta
            return mu / self.sigma2
        else:
            mu, sigma2 = theta
            return np.array([mu / sigma2, -1 / (2 * sigma2)])
    
    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        计算充分统计量
        
        如果方差已知：u(x) = x
        如果方差未知：u(x) = [x, x²]^T
        """
        if self.known_variance:
            return x
        else:
            return np.array([x, x**2])
    
    def demonstrate(self) -> None:
        """演示高斯分布的指数族性质"""
        print(f"\n{self.name}分布的指数族形式")
        print("=" * 50)
        
        # 案例1：方差已知
        print("\n案例1：方差已知(σ²=1)")
        print("-" * 40)
        self.known_variance = True
        self.sigma2 = 1.0
        
        mu_values = [-2, -1, 0, 1, 2]
        print(f"{'μ':<10} {'η=μ/σ²':<10}")
        print("-" * 40)
        for mu in mu_values:
            eta = self.natural_parameters(mu)
            print(f"{mu:<10.1f} {eta:<10.1f}")
        
        # 案例2：方差未知
        print("\n案例2：均值和方差都未知")
        print("-" * 40)
        self.known_variance = False
        
        params = [(0, 1), (0, 2), (1, 1), (2, 0.5)]
        print(f"{'(μ,σ²)':<15} {'η₁=μ/σ²':<12} {'η₂=-1/(2σ²)':<12}")
        print("-" * 40)
        for mu, sigma2 in params:
            eta = self.natural_parameters((mu, sigma2))
            print(f"({mu},{sigma2}){'':<10} {eta[0]:<12.3f} {eta[1]:<12.3f}")
        
        print("\n关键观察：")
        print("1. 自然参数与精度(1/σ²)成正比，而不是方差")
        print("2. 充分统计量包含一阶矩(x)和二阶矩(x²)")
        print("3. 这解释了为什么高斯分布用均值和方差完全确定")


class MultinomialExponential(ExponentialFamily):
    """
    多项分布的指数族形式
    
    这是分类问题的基础分布。
    
    标准形式：p(x|π) = ∏ πₖ^xₖ
    
    指数族形式：
    p(x|η) = exp(η^T x - A(η))
    
    其中：
    - 自然参数：ηₖ = ln πₖ （对数概率）
    - 充分统计量：u(x) = x （计数向量）
    - 对数配分函数：A(η) = ln(∑exp(ηₖ))
    
    这与softmax函数密切相关！
    """
    
    def __init__(self, K: int):
        """
        初始化多项分布
        
        Args:
            K: 类别数
        """
        super().__init__("Multinomial")
        self.K = K
    
    def natural_parameters(self, pi: np.ndarray) -> np.ndarray:
        """
        从概率向量π计算自然参数
        
        注意：由于概率和为1的约束，
        实际上只有K-1个自由参数。
        """
        # 使用前K-1个类别相对于最后一个类别的对数比
        return np.log(pi[:-1] / pi[-1])
    
    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """充分统计量是计数向量"""
        return x[:-1]  # 只需要前K-1个
    
    def softmax(self, eta: np.ndarray) -> np.ndarray:
        """
        从自然参数恢复概率（softmax函数）
        
        这展示了softmax的来源：
        它是多项分布自然参数的逆变换！
        """
        # 添加最后一个类别（参考类别）的自然参数为0
        eta_full = np.concatenate([eta, [0]])
        # 数值稳定的softmax
        eta_max = np.max(eta_full)
        exp_eta = np.exp(eta_full - eta_max)
        return exp_eta / np.sum(exp_eta)
    
    def demonstrate(self) -> None:
        """演示多项分布与softmax的关系"""
        print(f"\n{self.name}分布与Softmax")
        print("=" * 50)
        
        # 3类别的例子
        self.K = 3
        pi_examples = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.8, 0.1, 0.1]),
            np.array([1/3, 1/3, 1/3])
        ]
        
        print("概率向量 → 自然参数 → 恢复的概率")
        print("-" * 50)
        
        for pi in pi_examples:
            eta = self.natural_parameters(pi)
            pi_recovered = self.softmax(eta)
            
            print(f"π = {pi}")
            print(f"η = {eta} (相对于最后一类的对数比)")
            print(f"恢复 = {pi_recovered}")
            print(f"误差 = {np.max(np.abs(pi - pi_recovered)):.6f}")
            print()
        
        print("观察：")
        print("1. 均匀分布时，所有自然参数为0")
        print("2. 这就是softmax函数在神经网络中的理论基础")
        print("3. 交叉熵损失来源于多项分布的负对数似然")


def demonstrate_conjugate_priors() -> None:
    """
    演示指数族分布的共轭先验
    
    指数族的一个重要性质是都有共轭先验。
    共轭先验使贝叶斯推断变得简单。
    
    共轭对：
    - 伯努利/二项 ↔ 贝塔
    - 多项 ↔ 狄利克雷
    - 高斯(已知方差) ↔ 高斯
    - 高斯(未知方差) ↔ 高斯-逆伽马
    - 泊松 ↔ 伽马
    - 指数 ↔ 伽马
    """
    print("\n指数族分布的共轭先验")
    print("=" * 60)
    
    conjugate_pairs = [
        ("伯努利/二项分布", "贝塔分布", "Beta(α, β)"),
        ("多项分布", "狄利克雷分布", "Dir(α)"),
        ("高斯分布(已知σ²)", "高斯分布", "N(μ₀, σ₀²)"),
        ("高斯分布(未知μ,σ²)", "高斯-逆伽马", "NIG(μ₀,λ,α,β)"),
        ("泊松分布", "伽马分布", "Gamma(α, β)"),
        ("指数分布", "伽马分布", "Gamma(α, β)"),
        ("伽马分布", "伽马分布", "Gamma(α, β)")
    ]
    
    print(f"{'似然分布':<20} {'共轭先验':<20} {'参数形式':<15}")
    print("-" * 60)
    
    for likelihood, prior, params in conjugate_pairs:
        print(f"{likelihood:<20} {prior:<20} {params:<15}")
    
    print("\n共轭先验的优势：")
    print("1. 后验分布与先验分布同族")
    print("2. 参数更新有封闭形式")
    print("3. 计算效率高")
    print("4. 便于序贯更新")
    
    # 示例：伯努利-贝塔共轭
    print("\n示例：伯努利-贝塔共轭更新")
    print("-" * 40)
    print("先验：Beta(α, β)")
    print("观测：m个成功，n-m个失败")
    print("后验：Beta(α+m, β+n-m)")
    print("\n这就是参数的简单加法更新！")


def demonstrate_sufficient_statistics() -> None:
    """
    演示充分统计量的概念
    
    充分统计量是数据压缩的极限：
    它包含了数据中关于参数的所有信息。
    
    无论有多少数据，充分统计量的维度是固定的！
    这是指数族分布的关键优势。
    """
    print("\n充分统计量：数据压缩的极限")
    print("=" * 60)
    
    # 高斯分布的例子
    print("\n例子：估计高斯分布的参数")
    print("-" * 40)
    
    # 生成数据
    np.random.seed(42)
    true_mu, true_sigma = 5.0, 2.0
    
    for n_samples in [10, 100, 1000, 10000]:
        data = np.random.normal(true_mu, true_sigma, n_samples)
        
        # 充分统计量
        sufficient_stats = {
            'sum': np.sum(data),
            'sum_squares': np.sum(data**2),
            'n': n_samples
        }
        
        # 从充分统计量估计参数
        est_mu = sufficient_stats['sum'] / sufficient_stats['n']
        est_sigma2 = (sufficient_stats['sum_squares'] / sufficient_stats['n'] 
                     - est_mu**2)
        
        print(f"\n数据点数: {n_samples}")
        print(f"原始数据大小: {n_samples}个数字")
        print(f"充分统计量: 3个数字")
        print(f"  ∑x = {sufficient_stats['sum']:.2f}")
        print(f"  ∑x² = {sufficient_stats['sum_squares']:.2f}")
        print(f"  n = {sufficient_stats['n']}")
        print(f"估计: μ={est_mu:.3f}, σ²={est_sigma2:.3f}")
        print(f"压缩比: {n_samples/3:.1f}:1")
    
    print("\n关键观察：")
    print("1. 无论数据量多大，只需要3个数字")
    print("2. 这3个数字包含了估计参数的所有信息")
    print("3. 这就是为什么在线算法可以高效更新参数")
    print("4. MapReduce等分布式算法利用了这个性质")


def visualize_exponential_family_relationships() -> None:
    """
    可视化指数族分布之间的关系
    
    展示不同分布如何通过极限、特殊情况等相互联系。
    """
    print("\n指数族分布关系图")
    print("=" * 60)
    
    relationships = """
    伯努利分布 
        ↓ (N次试验)
    二项分布
        ↓ (N→∞, p→0, Np=λ)
    泊松分布
    
    贝塔分布
        ↓ (K维推广)
    狄利克雷分布
    
    指数分布
        ↓ (形状参数)
    伽马分布
        ↓ (特殊情况)
    卡方分布
    
    高斯分布
        ↓ (多维)
    多维高斯分布
        ↓ (精度矩阵)
    高斯图模型
    """
    
    print(relationships)
    
    print("\n这些关系说明了：")
    print("1. 许多分布是其他分布的特殊情况或极限")
    print("2. 理解一个分布有助于理解整个家族")
    print("3. 选择合适的分布需要理解它们的关系")