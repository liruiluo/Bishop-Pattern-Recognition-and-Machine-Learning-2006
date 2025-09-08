"""
14.2 委员会方法 (Committee Methods)
====================================

委员会方法通过组合多个模型的预测来改善性能。

Bootstrap聚合(Bagging)：
1. 从原始数据集D中有放回采样，生成B个bootstrap样本
2. 在每个bootstrap样本上训练一个模型
3. 组合：
   - 回归：平均
   - 分类：投票

理论基础：
假设有B个独立同分布的预测器，每个方差为σ²
- 单个预测器的期望误差：E[ε²] = bias² + σ²
- 平均后的期望误差：E[ε²_avg] = bias² + σ²/B

关键：降低方差，不改变偏差

投票策略：
1. 硬投票：每个分类器投一票
2. 软投票：使用预测概率的平均
3. 加权投票：不同分类器不同权重

多样性来源：
- 数据扰动：bootstrap采样
- 特征扰动：随机子空间
- 参数扰动：不同初始化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Bootstrap聚合分类器
    
    通过bootstrap采样训练多个基学习器，然后投票决定最终预测。
    
    原理：
    1. Bootstrap采样：从n个样本中有放回采样n个
    2. 每个样本被选中概率：1-(1-1/n)^n ≈ 0.632
    3. Out-of-Bag(OOB)样本：约36.8%样本未被选中
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 10,
                 max_samples: float = 1.0, max_features: float = 1.0,
                 bootstrap: bool = True, bootstrap_features: bool = False,
                 oob_score: bool = False, random_state: Optional[int] = None):
        """
        初始化Bagging分类器
        
        Args:
            base_estimator: 基学习器
            n_estimators: 基学习器数量
            max_samples: 采样比例
            max_features: 特征采样比例
            bootstrap: 是否有放回采样
            bootstrap_features: 是否对特征采样
            oob_score: 是否计算OOB分数
            random_state: 随机种子
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        
        # 内部变量
        self.estimators_ = []
        self.estimators_features_ = []
        self.oob_score_ = None
        self.oob_decision_function_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifier':
        """
        训练Bagging分类器
        
        过程：
        1. 生成bootstrap样本
        2. 训练基学习器
        3. 如果需要，计算OOB分数
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 默认基学习器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=5)
        
        # 计算采样数量
        n_samples_bootstrap = int(self.max_samples * n_samples)
        n_features_bootstrap = int(self.max_features * n_features)
        
        # OOB预测初始化
        if self.oob_score:
            self.oob_decision_function_ = np.zeros((n_samples, self.n_classes_))
            n_oob_predictions = np.zeros(n_samples)
        
        # 训练基学习器
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, n_samples_bootstrap, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, n_samples_bootstrap, replace=False)
            
            # 特征采样
            if self.bootstrap_features:
                feature_indices = np.random.choice(n_features, n_features_bootstrap, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            # 获取训练数据
            X_train = X[sample_indices][:, feature_indices]
            y_train = y[sample_indices]
            
            # 训练基学习器
            estimator = clone(self.base_estimator)
            estimator.fit(X_train, y_train)
            
            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indices)
            
            # OOB预测
            if self.oob_score:
                # 找出未被采样的样本（OOB样本）
                oob_indices = np.setdiff1d(np.arange(n_samples), sample_indices)
                
                if len(oob_indices) > 0:
                    X_oob = X[oob_indices][:, feature_indices]
                    
                    # 预测OOB样本
                    if hasattr(estimator, 'predict_proba'):
                        oob_pred = estimator.predict_proba(X_oob)
                    else:
                        # 如果没有predict_proba，使用硬预测
                        oob_pred_hard = estimator.predict(X_oob)
                        oob_pred = np.zeros((len(oob_indices), self.n_classes_))
                        for j, c in enumerate(self.classes_):
                            oob_pred[oob_pred_hard == c, j] = 1
                    
                    self.oob_decision_function_[oob_indices] += oob_pred
                    n_oob_predictions[oob_indices] += 1
        
        # 计算OOB分数
        if self.oob_score:
            # 归一化OOB预测
            mask = n_oob_predictions > 0
            self.oob_decision_function_[mask] /= n_oob_predictions[mask].reshape(-1, 1)
            
            # 计算准确率
            oob_pred = self.classes_[np.argmax(self.oob_decision_function_, axis=1)]
            self.oob_score_ = np.mean(oob_pred[mask] == y[mask])
            
            print(f"OOB准确率: {self.oob_score_:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测：硬投票
        
        每个基学习器投一票，选择得票最多的类别。
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, (estimator, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
            predictions[:, i] = estimator.predict(X[:, features])
        
        # 投票
        final_predictions = np.zeros(n_samples)
        for i in range(n_samples):
            # 统计每个类别的票数
            votes = np.bincount(predictions[i].astype(int), minlength=self.n_classes_)
            final_predictions[i] = np.argmax(votes)
        
        return self.classes_[final_predictions.astype(int)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率：软投票
        
        平均所有基学习器的预测概率。
        """
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            if hasattr(estimator, 'predict_proba'):
                probas += estimator.predict_proba(X[:, features])
            else:
                # 如果没有predict_proba，使用硬预测
                pred = estimator.predict(X[:, features])
                for j, c in enumerate(self.classes_):
                    probas[pred == c, j] += 1
        
        # 归一化
        probas /= self.n_estimators
        return probas


class VotingClassifier(BaseEstimator, ClassifierMixin):
    """
    投票分类器
    
    组合多个不同的分类器，通过投票决定最终预测。
    
    投票策略：
    1. 硬投票：argmax(Σ_i I(h_i(x) = c))
    2. 软投票：argmax(Σ_i P_i(c|x))
    3. 加权投票：argmax(Σ_i w_i * P_i(c|x))
    """
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]],
                 voting: str = 'soft', weights: Optional[np.ndarray] = None):
        """
        初始化投票分类器
        
        Args:
            estimators: [(名称, 分类器)]列表
            voting: 投票方式 ('hard', 'soft')
            weights: 各分类器权重
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        
        self.estimators_ = []
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingClassifier':
        """训练所有基分类器"""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # 训练每个分类器
        self.estimators_ = []
        for name, estimator in self.estimators:
            print(f"训练 {name}...")
            fitted_estimator = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        根据投票策略组合预测结果。
        """
        if self.voting == 'hard':
            return self._predict_hard(X)
        else:
            return self._predict_soft(X)
    
    def _predict_hard(self, X: np.ndarray) -> np.ndarray:
        """硬投票"""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.estimators_)))
        
        for i, (name, estimator) in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)
        
        # 加权投票
        if self.weights is not None:
            weighted_votes = np.zeros((n_samples, self.n_classes_))
            for i in range(len(self.estimators_)):
                for j, c in enumerate(self.classes_):
                    weighted_votes[predictions[:, i] == c, j] += self.weights[i]
            return self.classes_[np.argmax(weighted_votes, axis=1)]
        else:
            # 简单多数投票
            return stats.mode(predictions, axis=1)[0].ravel()
    
    def _predict_soft(self, X: np.ndarray) -> np.ndarray:
        """软投票"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        weights = self.weights if self.weights is not None else np.ones(len(self.estimators_))
        weights = weights / np.sum(weights)
        
        for i, (name, estimator) in enumerate(self.estimators_):
            if hasattr(estimator, 'predict_proba'):
                probas += weights[i] * estimator.predict_proba(X)
            else:
                # 硬预测转换为概率
                pred = estimator.predict(X)
                for j, c in enumerate(self.classes_):
                    probas[pred == c, j] += weights[i]
        
        return probas


class BayesianModelAveraging:
    """
    贝叶斯模型平均
    
    使用贝叶斯方法计算模型后验概率，进行加权平均。
    
    模型后验：P(M_k|D) ∝ P(D|M_k)P(M_k)
    预测分布：P(y|x,D) = Σ_k P(y|x,M_k)P(M_k|D)
    
    模型证据：P(D|M_k) = ∫ P(D|θ,M_k)P(θ|M_k)dθ
    """
    
    def __init__(self, models: List[BaseEstimator],
                 prior_weights: Optional[np.ndarray] = None):
        """
        初始化BMA
        
        Args:
            models: 候选模型列表
            prior_weights: 模型先验概率
        """
        self.models = models
        self.n_models = len(models)
        
        if prior_weights is None:
            # 均匀先验
            self.prior_weights = np.ones(self.n_models) / self.n_models
        else:
            self.prior_weights = prior_weights / np.sum(prior_weights)
        
        self.posterior_weights = None
        self.log_evidences = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianModelAveraging':
        """
        计算模型后验概率
        
        使用近似方法估计模型证据。
        """
        self.log_evidences = np.zeros(self.n_models)
        
        for i, model in enumerate(self.models):
            # 训练模型
            model.fit(X, y)
            
            # 估计模型证据（使用BIC近似）
            n_samples = len(y)
            
            # 计算对数似然
            if hasattr(model, 'score'):
                # 使用模型的score方法（通常是对数似然或R²）
                log_likelihood = model.score(X, y) * n_samples
            else:
                # 使用预测误差估计
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - n_samples / 2
            
            # BIC = log P(D|M) ≈ log P(D|θ_MAP,M) - (k/2)log(n)
            # k是参数数量，这里简化处理
            n_params = self._count_parameters(model)
            bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
            
            self.log_evidences[i] = bic
        
        # 计算后验权重
        # P(M_k|D) ∝ P(D|M_k)P(M_k)
        log_posteriors = self.log_evidences + np.log(self.prior_weights)
        
        # 归一化（在对数空间避免溢出）
        max_log_posterior = np.max(log_posteriors)
        self.posterior_weights = np.exp(log_posteriors - max_log_posterior)
        self.posterior_weights /= np.sum(self.posterior_weights)
        
        print("模型后验权重：")
        for i, weight in enumerate(self.posterior_weights):
            print(f"  模型{i+1}: {weight:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        贝叶斯模型平均预测
        
        加权平均所有模型的预测。
        """
        predictions = np.zeros((len(X), self.n_models))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # 加权平均
        if len(predictions.shape) == 2:
            # 回归问题
            return np.average(predictions, weights=self.posterior_weights, axis=1)
        else:
            # 分类问题（投票）
            weighted_pred = np.zeros(len(X))
            for i in range(len(X)):
                # 加权投票
                values, counts = np.unique(predictions[i], return_counts=True)
                weighted_counts = np.zeros(len(values))
                for j, val in enumerate(values):
                    mask = predictions[i] == val
                    weighted_counts[j] = np.sum(self.posterior_weights[mask])
                weighted_pred[i] = values[np.argmax(weighted_counts)]
            return weighted_pred
    
    def _count_parameters(self, model) -> int:
        """估计模型参数数量"""
        # 简化：根据模型类型估计
        if hasattr(model, 'coef_'):
            return np.prod(model.coef_.shape)
        elif hasattr(model, 'tree_'):
            return model.tree_.node_count
        else:
            # 默认值
            return 10


def demonstrate_bagging(n_estimators: int = 10, show_plot: bool = True) -> None:
    """
    演示Bagging方法
    
    展示Bootstrap聚合如何降低方差。
    """
    print("\nBagging演示")
    print("=" * 60)
    
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # 生成非线性数据
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建Bagging分类器
    bagging = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=n_estimators,
        oob_score=True,
        random_state=42
    )
    
    # 训练
    bagging.fit(X_train, y_train)
    
    # 预测
    y_pred_train = bagging.predict(X_train)
    y_pred_test = bagging.predict(X_test)
    
    # 单个决策树对比
    single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    single_tree.fit(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    
    # 计算准确率
    from sklearn.metrics import accuracy_score
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    single_acc = accuracy_score(y_test, y_pred_single)
    
    print(f"\n性能对比：")
    print(f"  单决策树测试准确率: {single_acc:.3f}")
    print(f"  Bagging训练准确率: {train_acc:.3f}")
    print(f"  Bagging测试准确率: {test_acc:.3f}")
    if bagging.oob_score:
        print(f"  OOB准确率: {bagging.oob_score_:.3f}")
    
    # 分析方差降低
    print(f"\n方差分析：")
    
    # 多次训练获得方差估计
    n_trials = 10
    single_accs = []
    bagging_accs = []
    
    for trial in range(n_trials):
        # 重新采样
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_trial = X_train[idx]
        y_trial = y_train[idx]
        
        # 单树
        tree = DecisionTreeClassifier(max_depth=5, random_state=trial)
        tree.fit(X_trial, y_trial)
        single_accs.append(accuracy_score(y_test, tree.predict(X_test)))
        
        # Bagging
        bag = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=n_estimators,
            random_state=trial
        )
        bag.fit(X_trial, y_trial)
        bagging_accs.append(accuracy_score(y_test, bag.predict(X_test)))
    
    print(f"  单树准确率: {np.mean(single_accs):.3f} ± {np.std(single_accs):.3f}")
    print(f"  Bagging准确率: {np.mean(bagging_accs):.3f} ± {np.std(bagging_accs):.3f}")
    print(f"  方差降低比例: {(1 - np.std(bagging_accs)/np.std(single_accs)):.1%}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 单决策树决策边界
        ax1 = axes[0, 0]
        Z = single_tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax1.set_title(f'单决策树 (准确率={single_acc:.3f})')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        
        # Bagging决策边界
        ax2 = axes[0, 1]
        Z = bagging.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax2.set_title(f'Bagging (准确率={test_acc:.3f})')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        
        # 预测概率
        ax3 = axes[0, 2]
        Z_proba = bagging.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        contour = ax3.contourf(xx, yy, Z_proba, levels=20, cmap=plt.cm.RdYlBu_r)
        plt.colorbar(contour, ax=ax3)
        ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax3.set_title('预测概率')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        
        # OOB样本分析
        ax4 = axes[1, 0]
        if bagging.oob_score:
            oob_predictions = np.argmax(bagging.oob_decision_function_, axis=1)
            correct = oob_predictions == y_train
            ax4.scatter(X_train[correct, 0], X_train[correct, 1], 
                       c='green', alpha=0.5, label='OOB正确', s=20)
            ax4.scatter(X_train[~correct, 0], X_train[~correct, 1],
                       c='red', alpha=0.5, label='OOB错误', s=20)
            ax4.set_title(f'OOB预测 (准确率={bagging.oob_score_:.3f})')
            ax4.set_xlabel('特征1')
            ax4.set_ylabel('特征2')
            ax4.legend()
        
        # 基学习器多样性
        ax5 = axes[1, 1]
        # 显示前3个基学习器的决策边界
        colors = ['red', 'green', 'blue']
        for i in range(min(3, n_estimators)):
            estimator = bagging.estimators_[i]
            features = bagging.estimators_features_[i]
            
            if len(features) == 2:  # 只有选择了2个特征才能可视化
                Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax5.contour(xx, yy, Z, alpha=0.3, colors=[colors[i]], 
                          linewidths=2, linestyles='--')
        
        ax5.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax5.set_title('基学习器多样性（前3个）')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        
        # 准确率分布
        ax6 = axes[1, 2]
        ax6.boxplot([single_accs, bagging_accs], labels=['单树', 'Bagging'])
        ax6.set_ylabel('测试准确率')
        ax6.set_title('多次试验准确率分布')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Bootstrap聚合 (Bagging) - {n_estimators}个基学习器', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. Bagging显著降低方差")
    print("2. 决策边界更平滑")
    print("3. OOB可用于模型评估")
    print("4. 基学习器具有多样性")


def demonstrate_voting(show_plot: bool = True) -> None:
    """
    演示投票分类器
    
    展示不同投票策略的效果。
    """
    print("\n投票分类器演示")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建异质分类器
    estimators = [
        ('决策树', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('朴素贝叶斯', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_neighbors=5))
    ]
    
    # 硬投票
    hard_voting = VotingClassifier(estimators, voting='hard')
    hard_voting.fit(X_train, y_train)
    
    # 软投票
    soft_voting = VotingClassifier(estimators, voting='soft')
    soft_voting.fit(X_train, y_train)
    
    # 加权软投票（根据个体性能设置权重）
    weights = []
    for name, clf in estimators:
        clf_temp = clone(clf).fit(X_train, y_train)
        acc = accuracy_score(y_test, clf_temp.predict(X_test))
        weights.append(acc)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    weighted_voting = VotingClassifier(estimators, voting='soft', weights=weights)
    weighted_voting.fit(X_train, y_train)
    
    # 评估
    from sklearn.metrics import accuracy_score
    
    print("\n个体分类器性能：")
    for name, clf in estimators:
        clf_temp = clone(clf).fit(X_train, y_train)
        acc = accuracy_score(y_test, clf_temp.predict(X_test))
        print(f"  {name}: {acc:.3f}")
    
    print("\n集成性能：")
    hard_acc = accuracy_score(y_test, hard_voting.predict(X_test))
    soft_acc = accuracy_score(y_test, soft_voting.predict(X_test))
    weighted_acc = accuracy_score(y_test, weighted_voting.predict(X_test))
    
    print(f"  硬投票: {hard_acc:.3f}")
    print(f"  软投票: {soft_acc:.3f}")
    print(f"  加权投票: {weighted_acc:.3f} (权重: {weights})")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 绘制各个分类器
        for idx, (name, clf) in enumerate(estimators[:4]):
            ax = axes[idx // 2, idx % 2]
            
            clf_temp = clone(clf).fit(X_train, y_train)
            Z = clf_temp.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                      cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
            
            acc = accuracy_score(y_test, clf_temp.predict(X_test))
            ax.set_title(f'{name} (准确率={acc:.3f})')
            ax.set_xlabel('特征1')
            ax.set_ylabel('特征2')
        
        # 硬投票
        ax = axes[1, 0]
        Z = hard_voting.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax.set_title(f'硬投票 (准确率={hard_acc:.3f})')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        # 软投票
        ax = axes[1, 1]
        Z = soft_voting.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax.set_title(f'软投票 (准确率={soft_acc:.3f})')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        plt.suptitle('投票分类器', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 投票组合多个分类器")
    print("2. 软投票通常优于硬投票")
    print("3. 加权投票可进一步提升")
    print("4. 分类器多样性很重要")