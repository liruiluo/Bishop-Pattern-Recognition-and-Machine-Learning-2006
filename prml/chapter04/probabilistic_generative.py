"""
4.2 概率生成模型 (Probabilistic Generative Models)
==================================================

生成模型通过建模类条件密度p(x|C_k)和先验p(C_k)，
使用贝叶斯定理得到后验概率进行分类。

贝叶斯定理：
p(C_k|x) = p(x|C_k)p(C_k) / p(x)

其中：
p(x) = Σ_j p(x|C_j)p(C_j)

二分类的后验概率：
p(C_1|x) = σ(a) = 1/(1 + exp(-a))

其中a = ln[p(x|C_1)p(C_1) / p(x|C_2)p(C_2)]

关键洞察：
许多分布（指数族）导致后验概率是输入的线性函数的sigmoid/softmax。

连续输入：
1. 高斯类条件密度
   - 共享协方差：线性决策边界
   - 不同协方差：二次决策边界

2. 最大似然估计
   - 先验：类别比例
   - 均值：类内均值
   - 协方差：类内协方差

离散输入：
朴素贝叶斯假设特征条件独立。

优点：
- 可以生成新样本
- 处理缺失数据
- 可解释性好

缺点：
- 需要更多参数
- 假设可能不成立
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')


class GaussianGenerativeClassifier:
    """
    高斯生成分类器
    
    假设每个类的数据服从高斯分布：
    p(x|C_k) = N(x|μ_k, Σ_k)
    
    两种情况：
    1. 共享协方差矩阵：Σ_k = Σ for all k
       → 线性决策边界
    2. 独立协方差矩阵：每个类有自己的Σ_k
       → 二次决策边界
    
    这就是经典的LDA（线性判别分析）和QDA（二次判别分析）。
    """
    
    def __init__(self, shared_covariance: bool = True,
                 reg_covar: float = 1e-6):
        """
        初始化高斯生成分类器
        
        Args:
            shared_covariance: 是否共享协方差矩阵
            reg_covar: 协方差正则化（避免奇异）
        """
        self.shared_covariance = shared_covariance
        self.reg_covar = reg_covar
        
        # 模型参数
        self.priors = None  # 先验概率 p(C_k)
        self.means = None   # 类均值 μ_k
        self.covariances = None  # 协方差矩阵 Σ_k
        self.classes = None
        self.n_classes = 0
        self.n_features = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianGenerativeClassifier':
        """
        最大似然估计参数
        
        MLE估计：
        - 先验：π_k = N_k / N
        - 均值：μ_k = (1/N_k) Σ_{n∈C_k} x_n
        - 协方差：Σ_k = (1/N_k) Σ_{n∈C_k} (x_n - μ_k)(x_n - μ_k)^T
        
        对于共享协方差：
        Σ = Σ_k π_k Σ_k （加权平均）
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            y: 标签，shape (n_samples,)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # 获取类别
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # 初始化参数
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, n_features))
        
        if self.shared_covariance:
            # 共享协方差
            shared_cov = np.zeros((n_features, n_features))
        else:
            # 每个类独立协方差
            self.covariances = np.zeros((self.n_classes, n_features, n_features))
        
        # 估计每个类的参数
        for k, class_label in enumerate(self.classes):
            # 该类的数据
            X_k = X[y == class_label]
            n_k = len(X_k)
            
            # 先验概率
            self.priors[k] = n_k / n_samples
            
            # 均值
            self.means[k] = np.mean(X_k, axis=0)
            
            # 协方差
            X_centered = X_k - self.means[k]
            cov_k = (X_centered.T @ X_centered) / n_k
            
            if self.shared_covariance:
                # 累积加权协方差
                shared_cov += self.priors[k] * cov_k
            else:
                # 存储类协方差
                self.covariances[k] = cov_k + self.reg_covar * np.eye(n_features)
        
        if self.shared_covariance:
            # 使用共享协方差
            shared_cov += self.reg_covar * np.eye(n_features)
            self.covariances = np.array([shared_cov] * self.n_classes)
        
        return self
    
    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        计算对数似然
        
        log p(x|C_k) = -0.5 * [d*log(2π) + log|Σ_k| + (x-μ_k)^T Σ_k^(-1) (x-μ_k)]
        
        Args:
            X: 输入数据，shape (n_samples, n_features)
            
        Returns:
            对数似然，shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        log_likelihoods = np.zeros((n_samples, self.n_classes))
        
        for k in range(self.n_classes):
            # 使用scipy的多元正态分布
            rv = multivariate_normal(mean=self.means[k], 
                                    cov=self.covariances[k],
                                    allow_singular=True)
            log_likelihoods[:, k] = rv.logpdf(X)
        
        return log_likelihoods
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        计算对数后验概率
        
        log p(C_k|x) = log p(x|C_k) + log p(C_k) - log p(x)
        
        使用log-sum-exp技巧避免数值溢出。
        
        Args:
            X: 输入数据
            
        Returns:
            对数后验概率
        """
        # 对数似然
        log_likelihoods = self._log_likelihood(X)
        
        # 加上对数先验
        log_priors = np.log(self.priors)
        log_joint = log_likelihoods + log_priors
        
        # 归一化（log-sum-exp）
        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal
        
        return log_posterior
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 输入数据
            
        Returns:
            类别概率，shape (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        选择后验概率最大的类。
        
        Args:
            X: 输入数据
            
        Returns:
            预测类别
        """
        log_proba = self.predict_log_proba(X)
        predictions = self.classes[np.argmax(log_proba, axis=1)]
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率"""
        return np.mean(self.predict(X) == y)


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器
    
    假设特征条件独立：
    p(x|C_k) = ∏_i p(x_i|C_k)
    
    这大大减少了参数数量：
    - 一般情况：O(d²)参数（协方差矩阵）
    - 朴素贝叶斯：O(d)参数
    
    虽然独立假设通常不成立，但分类性能往往很好。
    
    支持离散和连续特征：
    - 离散：多项分布
    - 连续：高斯分布
    """
    
    def __init__(self, feature_types: Optional[List[str]] = None,
                 smoothing: float = 1.0):
        """
        初始化朴素贝叶斯分类器
        
        Args:
            feature_types: 特征类型列表 ('discrete' 或 'continuous')
            smoothing: 拉普拉斯平滑参数（用于离散特征）
        """
        self.feature_types = feature_types
        self.smoothing = smoothing
        
        self.classes = None
        self.n_classes = 0
        self.n_features = 0
        self.priors = None
        
        # 特征参数
        self.feature_params = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """
        训练朴素贝叶斯分类器
        
        Args:
            X: 训练数据
            y: 标签
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # 自动检测特征类型
        if self.feature_types is None:
            self.feature_types = []
            for i in range(n_features):
                # 简单判断：如果唯一值少于10，认为是离散
                if len(np.unique(X[:, i])) < 10:
                    self.feature_types.append('discrete')
                else:
                    self.feature_types.append('continuous')
        
        # 获取类别
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # 计算先验
        self.priors = np.zeros(self.n_classes)
        for k, class_label in enumerate(self.classes):
            self.priors[k] = np.sum(y == class_label) / n_samples
        
        # 估计每个特征的参数
        for i in range(n_features):
            self.feature_params[i] = {}
            
            if self.feature_types[i] == 'discrete':
                # 离散特征：估计概率表
                unique_values = np.unique(X[:, i])
                
                for k, class_label in enumerate(self.classes):
                    X_k = X[y == class_label, i]
                    
                    # 计算每个值的概率（带平滑）
                    probs = {}
                    for val in unique_values:
                        count = np.sum(X_k == val) + self.smoothing
                        total = len(X_k) + self.smoothing * len(unique_values)
                        probs[val] = count / total
                    
                    self.feature_params[i][class_label] = {
                        'type': 'discrete',
                        'probs': probs,
                        'default': self.smoothing / (len(X_k) + self.smoothing * len(unique_values))
                    }
            
            else:  # continuous
                # 连续特征：估计高斯参数
                for k, class_label in enumerate(self.classes):
                    X_k = X[y == class_label, i]
                    
                    self.feature_params[i][class_label] = {
                        'type': 'continuous',
                        'mean': np.mean(X_k),
                        'std': np.std(X_k) + 1e-6  # 避免零方差
                    }
        
        return self
    
    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        计算对数似然
        
        log p(x|C_k) = Σ_i log p(x_i|C_k)
        
        Args:
            X: 输入数据
            
        Returns:
            对数似然
        """
        n_samples = X.shape[0]
        log_likelihoods = np.zeros((n_samples, self.n_classes))
        
        for k, class_label in enumerate(self.classes):
            log_like = np.zeros(n_samples)
            
            for i in range(self.n_features):
                params = self.feature_params[i][class_label]
                
                if params['type'] == 'discrete':
                    # 离散特征
                    for j in range(n_samples):
                        val = X[j, i]
                        if val in params['probs']:
                            log_like[j] += np.log(params['probs'][val])
                        else:
                            # 未见过的值，使用默认概率
                            log_like[j] += np.log(params['default'])
                
                else:  # continuous
                    # 连续特征：高斯对数似然
                    mean = params['mean']
                    std = params['std']
                    
                    # log N(x|μ,σ²) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²
                    log_like += (-0.5 * np.log(2 * np.pi) - np.log(std) - 
                               0.5 * ((X[:, i] - mean) / std) ** 2)
            
            log_likelihoods[:, k] = log_like
        
        return log_likelihoods
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        # 对数似然
        log_likelihoods = self._log_likelihood(X)
        
        # 加上对数先验
        log_priors = np.log(self.priors)
        log_joint = log_likelihoods + log_priors
        
        # 归一化
        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal
        
        return np.exp(log_posterior)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        predictions = self.classes[np.argmax(proba, axis=1)]
        return predictions


def demonstrate_gaussian_generative(n_samples: int = 300,
                                   show_plot: bool = True) -> None:
    """
    演示高斯生成模型
    
    比较：
    1. 共享协方差（LDA）：线性边界
    2. 独立协方差（QDA）：二次边界
    
    Args:
        n_samples: 样本数
        show_plot: 是否绘图
    """
    print("\n高斯生成模型")
    print("=" * 60)
    
    # 生成二分类数据
    np.random.seed(42)
    
    # 类1：均值[2, 2]，协方差[[1, 0.5], [0.5, 1]]
    mean1 = np.array([2, 2])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
    
    # 类2：均值[-2, -2]，协方差[[1, -0.5], [-0.5, 2]]
    mean2 = np.array([-2, -2])
    cov2 = np.array([[1, -0.5], [-0.5, 2]])
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples // 2)
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # 训练LDA（共享协方差）
    lda = GaussianGenerativeClassifier(shared_covariance=True)
    lda.fit(X, y)
    lda_acc = lda.score(X, y)
    
    # 训练QDA（独立协方差）
    qda = GaussianGenerativeClassifier(shared_covariance=False)
    qda.fit(X, y)
    qda_acc = qda.score(X, y)
    
    print("训练结果：")
    print(f"LDA（线性边界）准确率: {lda_acc:.2%}")
    print(f"QDA（二次边界）准确率: {qda_acc:.2%}")
    
    print("\n模型参数：")
    print("类别先验: ", lda.priors)
    print("类别均值:")
    for k in range(2):
        print(f"  类{k}: {lda.means[k]}")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        
        models = [(lda, "LDA（共享协方差）"), 
                 (qda, "QDA（独立协方差）")]
        
        for idx, (model, title) in enumerate(models):
            ax = axes[idx]
            
            # 预测概率
            proba = model.predict_proba(X_grid)[:, 1]
            proba = proba.reshape(xx.shape)
            
            # 绘制概率等高线
            contour = ax.contourf(xx, yy, proba, levels=20, cmap='RdBu_r', alpha=0.8)
            ax.contour(xx, yy, proba, levels=[0.5], colors='black', linewidths=2)
            
            # 数据点
            ax.scatter(X1[:, 0], X1[:, 1], c='red', s=20, alpha=0.5, label='类0')
            ax.scatter(X2[:, 0], X2[:, 1], c='blue', s=20, alpha=0.5, label='类1')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(f'{title}\n准确率: {model.score(X, y):.2%}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 第三个图：显示生成的样本
        ax3 = axes[2]
        
        # 从学习的分布生成新样本
        n_gen = 50
        for k in range(2):
            # 生成类k的样本
            samples = np.random.multivariate_normal(lda.means[k], 
                                                   lda.covariances[k], 
                                                   n_gen)
            color = 'red' if k == 0 else 'blue'
            ax3.scatter(samples[:, 0], samples[:, 1], c=color, 
                       marker='*', s=50, alpha=0.3,
                       label=f'生成类{k}')
        
        # 原始数据
        ax3.scatter(X1[:, 0], X1[:, 1], c='red', s=20, alpha=0.8)
        ax3.scatter(X2[:, 0], X2[:, 1], c='blue', s=20, alpha=0.8)
        
        ax3.set_xlabel('x₁')
        ax3.set_ylabel('x₂')
        ax3.set_title('生成新样本')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('高斯生成模型', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. LDA假设共享协方差，导致线性决策边界")
    print("2. QDA允许不同协方差，可以有二次决策边界")
    print("3. 生成模型可以生成新的样本")


def demonstrate_naive_bayes(n_samples: int = 500,
                           show_plot: bool = True) -> None:
    """
    演示朴素贝叶斯
    
    展示条件独立假设的影响。
    
    Args:
        n_samples: 样本数
        show_plot: 是否绘图
    """
    print("\n朴素贝叶斯分类器")
    print("=" * 60)
    
    # 生成混合特征数据
    np.random.seed(42)
    
    # 3个连续特征，2个离散特征
    n_continuous = 3
    n_discrete = 2
    n_features = n_continuous + n_discrete
    
    # 两个类
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    
    # 类0
    X0_cont = np.random.randn(n_class0, n_continuous) - 1
    X0_disc = np.random.choice([0, 1, 2], size=(n_class0, n_discrete), 
                              p=[0.6, 0.3, 0.1])
    X0 = np.hstack([X0_cont, X0_disc])
    
    # 类1
    X1_cont = np.random.randn(n_class1, n_continuous) + 1
    X1_disc = np.random.choice([0, 1, 2], size=(n_class1, n_discrete),
                              p=[0.1, 0.3, 0.6])
    X1 = np.hstack([X1_cont, X1_disc])
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
    
    # 打乱
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    # 分割训练集和测试集
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 训练朴素贝叶斯
    feature_types = ['continuous'] * n_continuous + ['discrete'] * n_discrete
    nb = NaiveBayesClassifier(feature_types=feature_types)
    nb.fit(X_train, y_train)
    
    # 评估
    train_acc = np.mean(nb.predict(X_train) == y_train)
    test_acc = np.mean(nb.predict(X_test) == y_test)
    
    print(f"特征类型: {n_continuous}个连续 + {n_discrete}个离散")
    print(f"训练准确率: {train_acc:.2%}")
    print(f"测试准确率: {test_acc:.2%}")
    
    # 与完全高斯模型比较
    gauss = GaussianGenerativeClassifier(shared_covariance=False)
    gauss.fit(X_train, y_train)
    gauss_test_acc = gauss.score(X_test, y_test)
    print(f"高斯模型测试准确率: {gauss_test_acc:.2%}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 显示特征分布
        for i in range(min(3, n_continuous)):
            ax = axes[0, i]
            
            # 连续特征的类条件分布
            for class_label in [0, 1]:
                X_class = X_train[y_train == class_label, i]
                ax.hist(X_class, bins=20, alpha=0.5, density=True,
                       label=f'类{int(class_label)}')
                
                # 拟合的高斯
                params = nb.feature_params[i][class_label]
                if params['type'] == 'continuous':
                    x_range = np.linspace(X_train[:, i].min(), 
                                         X_train[:, i].max(), 100)
                    from scipy.stats import norm
                    pdf = norm.pdf(x_range, params['mean'], params['std'])
                    ax.plot(x_range, pdf, linewidth=2)
            
            ax.set_xlabel(f'特征{i+1}（连续）')
            ax.set_ylabel('密度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 显示离散特征分布
        for i in range(min(3, n_discrete)):
            ax = axes[1, i]
            
            feat_idx = n_continuous + i
            
            # 离散特征的概率
            values = sorted(np.unique(X_train[:, feat_idx]))
            width = 0.35
            
            for k, class_label in enumerate([0, 1]):
                probs = []
                params = nb.feature_params[feat_idx][class_label]
                
                for val in values:
                    if val in params['probs']:
                        probs.append(params['probs'][val])
                    else:
                        probs.append(params['default'])
                
                x_pos = np.arange(len(values)) + k * width
                ax.bar(x_pos, probs, width, label=f'类{int(class_label)}',
                      alpha=0.7)
            
            ax.set_xlabel(f'特征{feat_idx+1}（离散）')
            ax.set_ylabel('概率')
            ax.set_xticks(np.arange(len(values)) + width/2)
            ax.set_xticklabels(values)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 预测概率分布
        ax_last = axes[1, 2]
        proba_test = nb.predict_proba(X_test)[:, 1]
        
        ax_last.hist(proba_test[y_test == 0], bins=20, alpha=0.5,
                    color='blue', label='真实类0')
        ax_last.hist(proba_test[y_test == 1], bins=20, alpha=0.5,
                    color='red', label='真实类1')
        ax_last.axvline(x=0.5, color='black', linestyle='--',
                       label='决策阈值')
        
        ax_last.set_xlabel('预测概率 P(类1|x)')
        ax_last.set_ylabel('频数')
        ax_last.set_title(f'测试集预测（准确率: {test_acc:.2%}）')
        ax_last.legend()
        ax_last.grid(True, alpha=0.3)
        
        plt.suptitle('朴素贝叶斯分类器', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 朴素贝叶斯假设特征条件独立")
    print("2. 可以处理混合类型特征（连续+离散）")
    print("3. 参数少，训练快，但可能欠拟合")
    print("4. 尽管假设简单，实践中效果往往不错")