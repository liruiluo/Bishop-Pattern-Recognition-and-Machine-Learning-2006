"""
14.3 Boosting算法
==================

Boosting通过顺序训练弱学习器，每步关注前面错分的样本。

AdaBoost (Adaptive Boosting)：
1. 初始化样本权重：w_i^(1) = 1/N
2. 对于m = 1, ..., M:
   a. 训练弱分类器h_m最小化加权误差
   b. 计算误差率：ε_m = Σ_i w_i^(m) I(h_m(x_i) ≠ y_i)
   c. 计算分类器权重：α_m = 0.5 log((1-ε_m)/ε_m)
   d. 更新样本权重：w_i^(m+1) ∝ w_i^(m) exp(-α_m y_i h_m(x_i))
3. 最终分类器：H(x) = sign(Σ_m α_m h_m(x))

理论解释：
AdaBoost最小化指数损失：L = Σ_i exp(-y_i f(x_i))
其中f(x) = Σ_m α_m h_m(x)

Gradient Boosting：
通用框架，通过梯度下降在函数空间优化。
1. 初始化：f_0(x) = argmin_γ Σ_i L(y_i, γ)
2. 对于m = 1, ..., M:
   a. 计算负梯度：r_im = -[∂L(y_i, f(x_i))/∂f(x_i)]_{f=f_{m-1}}
   b. 拟合弱学习器h_m到负梯度
   c. 线搜索步长：γ_m = argmin_γ Σ_i L(y_i, f_{m-1}(x_i) + γh_m(x_i))
   d. 更新：f_m(x) = f_{m-1}(x) + γ_m h_m(x)

损失函数：
- 回归：平方损失、绝对损失、Huber损失
- 分类：对数损失（逻辑回归）、指数损失（AdaBoost）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    AdaBoost分类器
    
    自适应增强算法，通过加权组合弱分类器构建强分类器。
    
    算法流程：
    1. 初始化均匀权重
    2. 迭代训练弱分类器
    3. 根据误差调整样本权重
    4. 加权组合所有弱分类器
    
    关键思想：关注难分样本
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 50,
                 learning_rate: float = 1.0, algorithm: str = 'SAMME',
                 random_state: Optional[int] = None):
        """
        初始化AdaBoost
        
        Args:
            base_estimator: 基学习器
            n_estimators: 弱学习器数量
            learning_rate: 学习率（收缩因子）
            algorithm: 算法类型 ('SAMME', 'SAMME.R')
            random_state: 随机种子
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
        # 内部变量
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.feature_importances_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """
        训练AdaBoost
        
        SAMME算法：处理多分类问题的AdaBoost扩展
        SAMME.R：使用概率估计的版本
        """
        n_samples, n_features = X.shape
        
        # 初始化
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 1:
            raise ValueError("AdaBoost需要至少两个类别")
        
        # 将标签转换为{-1, 1}（二分类）或{0, 1, ..., K-1}（多分类）
        if self.n_classes_ == 2:
            y_encoded = np.where(y == self.classes_[0], -1, 1)
        else:
            y_encoded = np.searchsorted(self.classes_, y)
        
        # 默认基学习器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        
        # 初始化样本权重
        sample_weight = np.ones(n_samples) / n_samples
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 训练弱学习器
        for iboost in range(self.n_estimators):
            # 复制基学习器
            estimator = clone(self.base_estimator)
            
            # 训练弱学习器
            if hasattr(estimator, 'sample_weight'):
                estimator.fit(X, y, sample_weight=sample_weight)
            else:
                # 如果不支持样本权重，使用重采样
                indices = np.random.choice(n_samples, n_samples, p=sample_weight)
                estimator.fit(X[indices], y[indices])
            
            # 预测
            y_pred = estimator.predict(X)
            
            if self.n_classes_ == 2:
                # 二分类
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                incorrect = y_pred_encoded != y_encoded
            else:
                # 多分类
                incorrect = y_pred != y
            
            # 计算加权误差
            estimator_error = np.average(incorrect, weights=sample_weight)
            
            # 避免除零和完美预测
            if estimator_error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            
            if estimator_error >= 1.0 - 1.0 / self.n_classes_:
                # 如果误差太大，停止
                if len(self.estimators_) == 0:
                    raise ValueError("基学习器太弱，无法继续")
                break
            
            # 计算分类器权重
            if self.algorithm == 'SAMME':
                # SAMME算法
                alpha = self.learning_rate * np.log(
                    (1.0 - estimator_error) / estimator_error
                ) + np.log(self.n_classes_ - 1.0)
            else:
                # SAMME.R算法（使用概率）
                alpha = self.learning_rate * 0.5 * np.log(
                    (1.0 - estimator_error) / estimator_error
                )
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)
            
            # 更新样本权重
            if self.n_classes_ == 2:
                # 二分类：w_i *= exp(-alpha * y_i * h(x_i))
                sample_weight *= np.exp(-alpha * y_encoded * y_pred_encoded)
            else:
                # 多分类
                sample_weight *= np.exp(alpha * incorrect)
            
            # 归一化权重
            sample_weight /= np.sum(sample_weight)
            
            # 更新特征重要性
            if hasattr(estimator, 'feature_importances_'):
                self.feature_importances_ += alpha * estimator.feature_importances_
            
            # 早停条件
            if estimator_error == 0:
                break
        
        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        组合所有弱分类器的加权预测。
        """
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # 二分类
            decision = np.zeros(n_samples)
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                decision += weight * y_pred_encoded
            
            return self.classes_[(decision > 0).astype(int)]
        else:
            # 多分类
            decision = np.zeros((n_samples, self.n_classes_))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                
                for i, c in enumerate(self.classes_):
                    decision[y_pred == c, i] += weight
            
            return self.classes_[np.argmax(decision, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        将决策函数转换为概率。
        """
        n_samples = X.shape[0]
        decision = self.decision_function(X)
        
        if self.n_classes_ == 2:
            # 二分类：使用sigmoid函数
            decision = decision.reshape(-1, 1)
            decision = np.c_[-decision, decision]
            proba = 1.0 / (1.0 + np.exp(-2 * decision))
        else:
            # 多分类：使用softmax
            proba = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值"""
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            decision = np.zeros(n_samples)
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                decision += weight * y_pred_encoded
        else:
            decision = np.zeros((n_samples, self.n_classes_))
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                for i, c in enumerate(self.classes_):
                    decision[y_pred == c, i] += weight
        
        return decision
    
    def staged_predict(self, X: np.ndarray):
        """生成阶段预测（用于观察训练过程）"""
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            decision = np.zeros(n_samples)
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                decision += weight * y_pred_encoded
                yield self.classes_[(decision > 0).astype(int)]
        else:
            decision = np.zeros((n_samples, self.n_classes_))
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X)
                for i, c in enumerate(self.classes_):
                    decision[y_pred == c, i] += weight
                yield self.classes_[np.argmax(decision, axis=1)]


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    梯度提升分类器
    
    通过优化对数损失函数进行分类。
    
    损失函数：L = -Σ_i [y_i log p_i + (1-y_i) log(1-p_i)]
    
    算法：
    1. 初始化为类别先验概率的对数几率
    2. 迭代拟合负梯度（残差）
    3. 使用决策树作为基学习器
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 loss: str = 'deviance', random_state: Optional[int] = None):
        """
        初始化梯度提升分类器
        
        Args:
            n_estimators: 提升轮数
            learning_rate: 学习率（收缩因子）
            max_depth: 树的最大深度
            subsample: 子采样比例
            loss: 损失函数 ('deviance', 'exponential')
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        
        # 内部变量
        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        训练梯度提升分类器
        
        使用前向分步算法优化损失函数。
        """
        n_samples, n_features = X.shape
        
        # 初始化
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ > 2:
            raise NotImplementedError("当前仅支持二分类")
        
        # 转换标签为{0, 1}
        y_encoded = (y == self.classes_[1]).astype(int)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化预测值（类别先验的对数几率）
        p = np.mean(y_encoded)
        self.init_ = np.log(p / (1 - p)) if p > 0 and p < 1 else 0
        f = np.full(n_samples, self.init_)
        
        # 训练基学习器
        for m in range(self.n_estimators):
            # 计算概率
            p = 1.0 / (1.0 + np.exp(-f))
            
            # 计算负梯度（伪残差）
            if self.loss == 'deviance':
                # 对数损失的负梯度
                negative_gradient = y_encoded - p
            elif self.loss == 'exponential':
                # 指数损失的负梯度（类似AdaBoost）
                negative_gradient = y_encoded * np.exp(-y_encoded * f)
            else:
                raise ValueError(f"不支持的损失函数: {self.loss}")
            
            # 子采样
            if self.subsample < 1.0:
                subsample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, subsample_size, replace=False)
            else:
                indices = np.arange(n_samples)
            
            # 拟合回归树到负梯度
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(X[indices], negative_gradient[indices])
            
            # 预测
            predictions = tree.predict(X)
            
            # 线搜索最优步长（这里简化为固定学习率）
            # 实际实现中，每个叶节点应该有自己的最优值
            f += self.learning_rate * predictions
            
            # 保存树
            self.estimators_.append(tree)
            
            # 记录训练分数
            train_pred = (f > 0).astype(int)
            train_score = accuracy_score(y_encoded, train_pred)
            self.train_score_.append(train_score)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        decision = self.decision_function(X)
        return self.classes_[(decision > 0).astype(int)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        decision = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-decision))
        return np.c_[1 - p, p]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值"""
        n_samples = X.shape[0]
        f = np.full(n_samples, self.init_)
        
        for estimator in self.estimators_:
            f += self.learning_rate * estimator.predict(X)
        
        return f
    
    def staged_predict(self, X: np.ndarray):
        """生成阶段预测"""
        n_samples = X.shape[0]
        f = np.full(n_samples, self.init_)
        
        for estimator in self.estimators_:
            f += self.learning_rate * estimator.predict(X)
            yield self.classes_[(f > 0).astype(int)]


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    梯度提升回归器
    
    通过最小化任意可微损失函数进行回归。
    
    支持的损失函数：
    - 平方损失：L = 0.5 * (y - f)²
    - 绝对损失：L = |y - f|
    - Huber损失：结合平方和绝对损失的优点
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 loss: str = 'ls', alpha: float = 0.9,
                 random_state: Optional[int] = None):
        """
        初始化梯度提升回归器
        
        Args:
            n_estimators: 提升轮数
            learning_rate: 学习率
            max_depth: 树的最大深度
            subsample: 子采样比例
            loss: 损失函数 ('ls', 'lad', 'huber')
            alpha: Huber损失的分位点
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.loss = loss
        self.alpha = alpha
        self.random_state = random_state
        
        # 内部变量
        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        """训练梯度提升回归器"""
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化预测值
        if self.loss == 'ls':
            # 平方损失：使用均值
            self.init_ = np.mean(y)
        elif self.loss == 'lad':
            # 绝对损失：使用中位数
            self.init_ = np.median(y)
        elif self.loss == 'huber':
            # Huber损失：使用均值
            self.init_ = np.mean(y)
        else:
            raise ValueError(f"不支持的损失函数: {self.loss}")
        
        f = np.full(n_samples, self.init_)
        
        # 训练基学习器
        for m in range(self.n_estimators):
            # 计算负梯度
            residuals = y - f
            
            if self.loss == 'ls':
                # 平方损失的负梯度就是残差
                negative_gradient = residuals
            elif self.loss == 'lad':
                # 绝对损失的负梯度是符号函数
                negative_gradient = np.sign(residuals)
            elif self.loss == 'huber':
                # Huber损失的负梯度
                delta = np.percentile(np.abs(residuals), self.alpha * 100)
                negative_gradient = np.where(
                    np.abs(residuals) <= delta,
                    residuals,  # 平方损失部分
                    delta * np.sign(residuals)  # 绝对损失部分
                )
            
            # 子采样
            if self.subsample < 1.0:
                subsample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, subsample_size, replace=False)
            else:
                indices = np.arange(n_samples)
            
            # 拟合回归树
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(X[indices], negative_gradient[indices])
            
            # 更新预测
            predictions = tree.predict(X)
            f += self.learning_rate * predictions
            
            # 保存树
            self.estimators_.append(tree)
            
            # 记录训练分数
            if self.loss == 'ls':
                train_score = -mean_squared_error(y, f)
            elif self.loss == 'lad':
                train_score = -np.mean(np.abs(y - f))
            else:
                train_score = -np.mean(np.abs(residuals))
            
            self.train_score_.append(train_score)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        n_samples = X.shape[0]
        f = np.full(n_samples, self.init_)
        
        for estimator in self.estimators_:
            f += self.learning_rate * estimator.predict(X)
        
        return f
    
    def staged_predict(self, X: np.ndarray):
        """生成阶段预测"""
        n_samples = X.shape[0]
        f = np.full(n_samples, self.init_)
        
        for estimator in self.estimators_:
            f += self.learning_rate * estimator.predict(X)
            yield f.copy()


def demonstrate_adaboost(n_estimators: int = 50, show_plot: bool = True) -> None:
    """
    演示AdaBoost算法
    
    展示样本权重的自适应调整过程。
    """
    print("\nAdaBoost演示")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=2,
                              flip_y=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练AdaBoost
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # 决策树桩
        n_estimators=n_estimators,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(X_train, y_train)
    
    # 评估
    train_scores = []
    test_scores = []
    
    for y_pred_train, y_pred_test in zip(
        ada.staged_predict(X_train),
        ada.staged_predict(X_test)
    ):
        train_scores.append(accuracy_score(y_train, y_pred_train))
        test_scores.append(accuracy_score(y_test, y_pred_test))
    
    print(f"训练集准确率: {train_scores[-1]:.3f}")
    print(f"测试集准确率: {test_scores[-1]:.3f}")
    print(f"使用的弱学习器数: {len(ada.estimators_)}")
    
    # 单个弱学习器对比
    weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    weak_learner.fit(X_train, y_train)
    weak_acc = accuracy_score(y_test, weak_learner.predict(X_test))
    print(f"单个弱学习器准确率: {weak_acc:.3f}")
    
    # 分析错误率和权重
    print("\n前5个弱学习器：")
    for i in range(min(5, len(ada.estimators_))):
        print(f"  学习器{i+1}: 误差={ada.estimator_errors_[i]:.3f}, "
              f"权重={ada.estimator_weights_[i]:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 弱学习器决策边界
        ax1 = axes[0, 0]
        Z = weak_learner.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax1.set_title(f'单个弱学习器 (准确率={weak_acc:.3f})')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        
        # AdaBoost决策边界
        ax2 = axes[0, 1]
        Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax2.set_title(f'AdaBoost (准确率={test_scores[-1]:.3f})')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        
        # 决策函数
        ax3 = axes[0, 2]
        Z = ada.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        contour = ax3.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdYlBu_r)
        plt.colorbar(contour, ax=ax3)
        ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax3.set_title('决策函数值')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        
        # 训练过程
        ax4 = axes[1, 0]
        ax4.plot(range(1, len(train_scores) + 1), train_scores, 
                'b-', linewidth=2, label='训练集')
        ax4.plot(range(1, len(test_scores) + 1), test_scores,
                'r--', linewidth=2, label='测试集')
        ax4.set_xlabel('提升轮数')
        ax4.set_ylabel('准确率')
        ax4.set_title('训练过程')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 弱学习器权重
        ax5 = axes[1, 1]
        ax5.bar(range(1, len(ada.estimator_weights_) + 1),
               ada.estimator_weights_, alpha=0.7)
        ax5.set_xlabel('弱学习器')
        ax5.set_ylabel('权重')
        ax5.set_title('弱学习器权重')
        ax5.grid(True, alpha=0.3)
        
        # 弱学习器误差
        ax6 = axes[1, 2]
        ax6.plot(range(1, len(ada.estimator_errors_) + 1),
                ada.estimator_errors_, 'g-', linewidth=2)
        ax6.axhline(y=0.5, color='r', linestyle='--', label='随机猜测')
        ax6.set_xlabel('弱学习器')
        ax6.set_ylabel('加权误差率')
        ax6.set_title('弱学习器误差')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'AdaBoost算法 - {n_estimators}个弱学习器', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. AdaBoost将多个弱学习器组合成强学习器")
    print("2. 关注被错分的样本")
    print("3. 早期学习器权重通常较大")
    print("4. 可能对噪声敏感")


def demonstrate_gradient_boosting(n_estimators: int = 100,
                                 learning_rate: float = 0.1,
                                 show_plot: bool = True) -> None:
    """
    演示梯度提升算法
    
    展示通过拟合残差逐步改进预测。
    """
    print("\n梯度提升演示")
    print("=" * 60)
    
    # 回归问题演示
    print("\n1. 回归问题")
    print("-" * 40)
    
    # 生成非线性数据
    np.random.seed(42)
    n_samples = 200
    X_reg = np.random.uniform(-3, 3, (n_samples, 1))
    y_true = np.sin(2 * X_reg).ravel() + 0.1 * X_reg.ravel()**2
    y_reg = y_true + 0.3 * np.random.randn(n_samples)
    
    # 分割数据
    n_train = 140
    X_train_reg = X_reg[:n_train]
    y_train_reg = y_reg[:n_train]
    X_test_reg = X_reg[n_train:]
    y_test_reg = y_reg[n_train:]
    
    # 训练梯度提升回归器
    gb_reg = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        loss='ls',
        random_state=42
    )
    gb_reg.fit(X_train_reg, y_train_reg)
    
    # 评估
    train_mse = mean_squared_error(y_train_reg, gb_reg.predict(X_train_reg))
    test_mse = mean_squared_error(y_test_reg, gb_reg.predict(X_test_reg))
    
    print(f"训练MSE: {train_mse:.4f}")
    print(f"测试MSE: {test_mse:.4f}")
    
    # 分类问题演示
    print("\n2. 分类问题")
    print("-" * 40)
    
    from sklearn.datasets import make_circles
    
    # 生成非线性可分数据
    X_clf, y_clf = make_circles(n_samples=500, noise=0.15, factor=0.5, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )
    
    # 训练梯度提升分类器
    gb_clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        loss='deviance',
        random_state=42
    )
    gb_clf.fit(X_train_clf, y_train_clf)
    
    # 评估
    train_acc = accuracy_score(y_train_clf, gb_clf.predict(X_train_clf))
    test_acc = accuracy_score(y_test_clf, gb_clf.predict(X_test_clf))
    
    print(f"训练准确率: {train_acc:.3f}")
    print(f"测试准确率: {test_acc:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 回归：真实函数和预测
        ax1 = axes[0, 0]
        X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        y_plot_true = np.sin(2 * X_plot).ravel() + 0.1 * X_plot.ravel()**2
        y_plot_pred = gb_reg.predict(X_plot)
        
        ax1.plot(X_plot, y_plot_true, 'b-', linewidth=2, label='真实函数')
        ax1.scatter(X_train_reg, y_train_reg, alpha=0.3, s=20, label='训练数据')
        ax1.plot(X_plot, y_plot_pred, 'r--', linewidth=2, label='GB预测')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'梯度提升回归 (MSE={test_mse:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 回归：阶段预测
        ax2 = axes[0, 1]
        staged_preds = list(gb_reg.staged_predict(X_plot))
        n_stages = [1, 5, 10, 20, 50, n_estimators]
        
        for stage in n_stages[:5]:
            if stage <= len(staged_preds):
                ax2.plot(X_plot, staged_preds[stage-1], alpha=0.5,
                        label=f'{stage}轮')
        
        ax2.plot(X_plot, y_plot_true, 'b-', linewidth=2, label='真实')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('阶段预测（回归）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 回归：训练曲线
        ax3 = axes[0, 2]
        ax3.plot(-np.array(gb_reg.train_score_), 'b-', linewidth=2)
        ax3.set_xlabel('提升轮数')
        ax3.set_ylabel('训练损失')
        ax3.set_title('训练损失曲线')
        ax3.grid(True, alpha=0.3)
        
        # 分类：决策边界
        ax4 = axes[1, 0]
        h = 0.02
        x_min, x_max = X_clf[:, 0].min() - 0.5, X_clf[:, 0].max() + 0.5
        y_min, y_max = X_clf[:, 1].min() - 0.5, X_clf[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = gb_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax4.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax4.scatter(X_test_clf[:, 0], X_test_clf[:, 1], c=y_test_clf,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax4.set_title(f'梯度提升分类 (准确率={test_acc:.3f})')
        ax4.set_xlabel('特征1')
        ax4.set_ylabel('特征2')
        
        # 分类：概率预测
        ax5 = axes[1, 1]
        Z_proba = gb_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        contour = ax5.contourf(xx, yy, Z_proba, levels=20, cmap=plt.cm.RdYlBu_r)
        plt.colorbar(contour, ax=ax5)
        ax5.scatter(X_test_clf[:, 0], X_test_clf[:, 1], c=y_test_clf,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax5.set_title('预测概率')
        ax5.set_xlabel('特征1')
        ax5.set_ylabel('特征2')
        
        # 分类：训练曲线
        ax6 = axes[1, 2]
        ax6.plot(gb_clf.train_score_, 'g-', linewidth=2)
        ax6.set_xlabel('提升轮数')
        ax6.set_ylabel('训练准确率')
        ax6.set_title('训练准确率曲线')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'梯度提升 - {n_estimators}轮，学习率={learning_rate}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 梯度提升逐步拟合残差")
    print("2. 学习率控制过拟合")
    print("3. 可处理各种损失函数")
    print("4. 通常性能优异但训练慢")