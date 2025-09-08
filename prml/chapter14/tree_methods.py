"""
14.4-14.5 决策树与随机森林
===========================

决策树：
通过递归分割特征空间构建树结构模型。

分裂准则：
1. 分类：
   - 信息增益：IG = H(S) - Σ_v |S_v|/|S| * H(S_v)
   - 信息增益比：IGR = IG / H(A)
   - Gini不纯度：Gini = 1 - Σ_k p_k²

2. 回归：
   - 方差减少：Var = Var(S) - Σ_v |S_v|/|S| * Var(S_v)
   - 平均绝对误差

随机森林：
结合Bagging和随机特征选择的集成方法。

关键创新：
1. Bootstrap采样训练集
2. 每次分裂随机选择m个特征（m << p）
3. 不剪枝，充分生长
4. 投票/平均进行预测

优势：
- 减少过拟合
- 提供特征重要性
- OOB误差估计
- 并行训练

特征重要性：
基于不纯度减少或排列重要性计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Dict
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TreeNode:
    """决策树节点"""
    
    def __init__(self, feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None,
                 value: Optional[Union[int, float]] = None,
                 n_samples: int = 0,
                 impurity: float = 0.0):
        """
        初始化树节点
        
        Args:
            feature: 分裂特征索引
            threshold: 分裂阈值
            left: 左子节点
            right: 右子节点
            value: 叶节点的预测值
            n_samples: 节点样本数
            impurity: 节点不纯度
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity
        
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return self.value is not None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    决策树分类器
    
    使用CART算法构建二叉决策树。
    
    算法流程：
    1. 选择最优分裂点（最大化信息增益）
    2. 递归分裂直到满足停止条件
    3. 叶节点预测为多数类
    
    剪枝策略：
    - 预剪枝：限制深度、最小样本数
    - 后剪枝：代价复杂度剪枝
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = 'gini',
                 max_features: Optional[Union[int, str]] = None,
                 random_state: Optional[int] = None):
        """
        初始化决策树分类器
        
        Args:
            max_depth: 最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            criterion: 分裂准则 ('gini', 'entropy')
            max_features: 每次分裂考虑的特征数
            random_state: 随机种子
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        # 内部变量
        self.tree_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.feature_importances_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """训练决策树"""
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 确定每次分裂考虑的特征数
        if self.max_features is None:
            self.max_features_ = self.n_features_
        elif self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            self.max_features_ = int(np.log2(self.n_features_))
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(self.n_features_)
        
        # 构建树
        self.tree_ = self._build_tree(X, y, depth=0)
        
        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                   depth: int) -> TreeNode:
        """递归构建决策树"""
        n_samples = len(y)
        
        # 计算当前节点的不纯度
        if self.criterion == 'gini':
            impurity = self._gini(y)
        else:  # entropy
            impurity = self._entropy(y)
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            # 创建叶节点
            value = self._most_common_label(y)
            return TreeNode(value=value, n_samples=n_samples, impurity=impurity)
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y, impurity)
        
        if best_feature is None:
            # 无法分裂，创建叶节点
            value = self._most_common_label(y)
            return TreeNode(value=value, n_samples=n_samples, impurity=impurity)
        
        # 更新特征重要性
        self.feature_importances_[best_feature] += best_gain * n_samples
        
        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 检查最小叶节点样本数
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            value = self._most_common_label(y)
            return TreeNode(value=value, n_samples=n_samples, impurity=impurity)
        
        # 递归构建子树
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            n_samples=n_samples,
            impurity=impurity
        )
    
    def _best_split(self, X: np.ndarray, y: np.ndarray,
                   parent_impurity: float) -> Tuple[Optional[int], Optional[float], float]:
        """寻找最佳分裂点"""
        n_samples = len(y)
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # 随机选择特征
        features = np.random.choice(self.n_features_, 
                                  self.max_features_,
                                  replace=False)
        
        for feature in features:
            # 获取特征的唯一值作为候选分裂点
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # 分裂
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left == 0 or n_right == 0:
                    continue
                
                # 计算子节点不纯度
                if self.criterion == 'gini':
                    left_impurity = self._gini(y[left_mask])
                    right_impurity = self._gini(y[right_mask])
                else:
                    left_impurity = self._entropy(y[left_mask])
                    right_impurity = self._entropy(y[right_mask])
                
                # 计算信息增益
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _gini(self, y: np.ndarray) -> float:
        """计算Gini不纯度"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """计算信息熵"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """返回最常见的标签"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return self.classes_[most_common] if hasattr(self, 'classes_') else most_common
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return np.array([self._predict_sample(sample) for sample in X])
    
    def _predict_sample(self, x: np.ndarray) -> int:
        """预测单个样本"""
        node = self.tree_
        
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（简化版：叶节点的类别分布）"""
        # 这里简化处理，实际应该存储叶节点的类别分布
        predictions = self.predict(X)
        n_samples = len(X)
        probas = np.zeros((n_samples, self.n_classes_))
        
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            probas[i, class_idx] = 1.0
        
        return probas


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    随机森林分类器
    
    组合多个决策树，每棵树在bootstrap样本上训练，
    每次分裂只考虑随机子集特征。
    
    核心要素：
    1. Bootstrap采样
    2. 特征随机性
    3. 投票集成
    4. OOB评估
    """
    
    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 max_features: Union[str, int] = 'sqrt',
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: Optional[int] = None):
        """
        初始化随机森林
        
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            max_features: 每次分裂考虑的特征数
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            bootstrap: 是否使用bootstrap采样
            oob_score: 是否计算OOB分数
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        # 内部变量
        self.estimators_ = []
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self.feature_importances_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """训练随机森林"""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # OOB预测初始化
        if self.oob_score:
            self.oob_decision_function_ = np.zeros((n_samples, self.n_classes_))
            n_oob_predictions = np.zeros(n_samples)
        
        # 特征重要性初始化
        self.feature_importances_ = np.zeros(n_features)
        
        # 训练每棵树
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 创建决策树
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=i  # 每棵树不同的随机种子
            )
            
            # 训练树
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # 累积特征重要性
            self.feature_importances_ += tree.feature_importances_
            
            # OOB预测
            if self.oob_score and self.bootstrap:
                # 找出OOB样本
                oob_indices = np.setdiff1d(np.arange(n_samples), indices)
                
                if len(oob_indices) > 0:
                    oob_pred = tree.predict_proba(X[oob_indices])
                    
                    # 累积OOB预测
                    for j, idx in enumerate(oob_indices):
                        self.oob_decision_function_[idx] += oob_pred[j]
                        n_oob_predictions[idx] += 1
        
        # 平均特征重要性
        self.feature_importances_ /= self.n_estimators
        
        # 计算OOB分数
        if self.oob_score and self.bootstrap:
            # 归一化OOB预测
            mask = n_oob_predictions > 0
            self.oob_decision_function_[mask] /= n_oob_predictions[mask].reshape(-1, 1)
            
            # 计算准确率
            oob_pred = self.classes_[np.argmax(self.oob_decision_function_, axis=1)]
            self.oob_score_ = np.mean(oob_pred[mask] == y[mask])
            
            print(f"OOB准确率: {self.oob_score_:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测：多数投票"""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        # 投票
        final_predictions = []
        for i in range(n_samples):
            counter = Counter(predictions[i])
            final_predictions.append(counter.most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率：平均所有树的概率"""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        for tree in self.estimators_:
            probas += tree.predict_proba(X)
        
        probas /= self.n_estimators
        return probas


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    随机森林回归器
    
    预测值为所有树预测的平均。
    """
    
    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 max_features: Union[str, int] = 'sqrt',
                 random_state: Optional[int] = None):
        """初始化随机森林回归器"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        
        self.estimators_ = []
        self.feature_importances_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """训练"""
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 特征重要性初始化
        self.feature_importances_ = np.zeros(n_features)
        
        # 训练每棵树
        for i in range(self.n_estimators):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 创建回归树（这里简化使用sklearn）
            from sklearn.tree import DecisionTreeRegressor
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=i
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # 累积特征重要性
            self.feature_importances_ += tree.feature_importances_
        
        # 平均特征重要性
        self.feature_importances_ /= self.n_estimators
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测：平均所有树的预测"""
        predictions = np.zeros((len(X), self.n_estimators))
        
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        return np.mean(predictions, axis=1)


def demonstrate_decision_tree(max_depth: int = 5, show_plot: bool = True) -> None:
    """
    演示决策树
    
    展示决策树的分裂过程和决策边界。
    """
    print("\n决策树演示")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 生成数据
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练不同深度的决策树
    depths = [1, 3, 5, 10]
    trees = []
    scores = []
    
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        trees.append(tree)
        
        score = accuracy_score(y_test, tree.predict(X_test))
        scores.append(score)
        print(f"深度{depth:2d}的树: 测试准确率={score:.3f}")
    
    # 特征重要性
    print(f"\n特征重要性（深度={max_depth}）:")
    tree_main = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_main.fit(X_train, y_train)
    
    for i, importance in enumerate(tree_main.feature_importances_):
        print(f"  特征{i}: {importance:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 绘制不同深度的决策边界
        for idx, (depth, tree, score) in enumerate(zip(depths, trees, scores)):
            ax = axes[idx]
            
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                      cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
            
            ax.set_title(f'深度={depth} (准确率={score:.3f})')
            ax.set_xlabel('特征1')
            ax.set_ylabel('特征2')
        
        # 过拟合分析
        ax = axes[4]
        
        train_scores = []
        test_scores = []
        max_depths = range(1, 20)
        
        for depth in max_depths:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(X_train, y_train)
            
            train_scores.append(accuracy_score(y_train, tree.predict(X_train)))
            test_scores.append(accuracy_score(y_test, tree.predict(X_test)))
        
        ax.plot(max_depths, train_scores, 'b-', linewidth=2, label='训练集')
        ax.plot(max_depths, test_scores, 'r--', linewidth=2, label='测试集')
        ax.set_xlabel('树深度')
        ax.set_ylabel('准确率')
        ax.set_title('过拟合分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 树结构可视化（简化）
        ax = axes[5]
        ax.axis('off')
        
        # 显示决策路径示例
        def plot_tree_structure(node, x=0.5, y=1, width=1, ax=ax, depth=0, max_depth=3):
            if depth >= max_depth or node.is_leaf():
                # 叶节点
                ax.text(x, y, f'类{node.value}\nn={node.n_samples}',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen'))
            else:
                # 内部节点
                ax.text(x, y, f'X{node.feature} ≤ {node.threshold:.2f}\nn={node.n_samples}',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue'))
                
                # 绘制子树
                if node.left and depth < max_depth - 1:
                    ax.plot([x, x - width/4], [y - 0.1, y - 0.3], 'k-')
                    plot_tree_structure(node.left, x - width/4, y - 0.4,
                                      width/2, ax, depth + 1, max_depth)
                
                if node.right and depth < max_depth - 1:
                    ax.plot([x, x + width/4], [y - 0.1, y - 0.3], 'k-')
                    plot_tree_structure(node.right, x + width/4, y - 0.4,
                                      width/2, ax, depth + 1, max_depth)
        
        # 绘制树结构
        plot_tree_structure(tree_main.tree_)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.2)
        ax.set_title('决策树结构（部分）')
        
        plt.suptitle('决策树分类', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 深度增加，模型复杂度增加")
    print("2. 过深容易过拟合")
    print("3. 决策边界呈矩形")
    print("4. 可解释性强")


def demonstrate_random_forest(n_estimators: int = 100,
                            max_features: str = 'sqrt',
                            show_plot: bool = True) -> None:
    """
    演示随机森林
    
    展示随机森林的集成效果和特征重要性。
    """
    print("\n随机森林演示")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 生成更复杂的数据
    X, y = make_classification(n_samples=500, n_features=20,
                              n_informative=10, n_redundant=5,
                              n_clusters_per_class=2,
                              flip_y=0.05, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        max_features=max_features,
        oob_score=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # 评估
    train_score = accuracy_score(y_train, rf.predict(X_train))
    test_score = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"训练准确率: {train_score:.3f}")
    print(f"测试准确率: {test_score:.3f}")
    if rf.oob_score:
        print(f"OOB准确率: {rf.oob_score_:.3f}")
    
    # 单棵树对比
    single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    single_score = accuracy_score(y_test, single_tree.predict(X_test))
    print(f"单棵树准确率: {single_score:.3f}")
    
    # 特征重要性
    print("\n前10个重要特征:")
    importance_indices = np.argsort(rf.feature_importances_)[::-1][:10]
    for i, idx in enumerate(importance_indices):
        print(f"  {i+1}. 特征{idx}: {rf.feature_importances_[idx]:.3f}")
    
    if show_plot:
        # 使用前两个最重要的特征进行可视化
        top_features = importance_indices[:2]
        X_vis = X[:, top_features]
        X_train_vis = X_train[:, top_features]
        X_test_vis = X_test[:, top_features]
        
        # 重新训练用于可视化
        rf_vis = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        rf_vis.fit(X_train_vis, y_train)
        
        tree_vis = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_vis.fit(X_train_vis, y_train)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
        y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 单棵树决策边界
        ax1 = axes[0, 0]
        Z = tree_vis.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax1.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        tree_acc = accuracy_score(y_test, tree_vis.predict(X_test_vis))
        ax1.set_title(f'单棵树 (准确率={tree_acc:.3f})')
        ax1.set_xlabel(f'特征{top_features[0]}')
        ax1.set_ylabel(f'特征{top_features[1]}')
        
        # 随机森林决策边界
        ax2 = axes[0, 1]
        Z = rf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax2.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        rf_acc = accuracy_score(y_test, rf_vis.predict(X_test_vis))
        ax2.set_title(f'随机森林 (准确率={rf_acc:.3f})')
        ax2.set_xlabel(f'特征{top_features[0]}')
        ax2.set_ylabel(f'特征{top_features[1]}')
        
        # 预测概率
        ax3 = axes[0, 2]
        Z_proba = rf_vis.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        contour = ax3.contourf(xx, yy, Z_proba, levels=20, cmap=plt.cm.RdYlBu_r)
        plt.colorbar(contour, ax=ax3)
        ax3.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test,
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
        ax3.set_title('预测概率')
        ax3.set_xlabel(f'特征{top_features[0]}')
        ax3.set_ylabel(f'特征{top_features[1]}')
        
        # 特征重要性
        ax4 = axes[1, 0]
        indices = np.argsort(rf.feature_importances_)[::-1][:10]
        ax4.bar(range(10), rf.feature_importances_[indices], alpha=0.7)
        ax4.set_xticks(range(10))
        ax4.set_xticklabels([f'F{i}' for i in indices])
        ax4.set_xlabel('特征')
        ax4.set_ylabel('重要性')
        ax4.set_title('特征重要性（前10）')
        ax4.grid(True, alpha=0.3)
        
        # 树的数量影响
        ax5 = axes[1, 1]
        n_trees = [1, 5, 10, 20, 50, 100]
        scores_by_trees = []
        
        for n in n_trees:
            rf_temp = RandomForestClassifier(
                n_estimators=n,
                max_depth=10,
                random_state=42
            )
            rf_temp.fit(X_train, y_train)
            scores_by_trees.append(accuracy_score(y_test, rf_temp.predict(X_test)))
        
        ax5.plot(n_trees, scores_by_trees, 'bo-', linewidth=2)
        ax5.set_xlabel('树的数量')
        ax5.set_ylabel('测试准确率')
        ax5.set_title('性能vs树数量')
        ax5.grid(True, alpha=0.3)
        
        # OOB误差曲线
        ax6 = axes[1, 2]
        
        # 累积OOB误差
        oob_errors = []
        for n in range(1, min(len(rf.estimators_) + 1, 101)):
            # 使用前n棵树预测
            predictions = np.zeros((len(X_train), n))
            for i in range(n):
                predictions[:, i] = rf.estimators_[i].predict(X_train)
            
            # 投票
            from scipy import stats
            ensemble_pred = stats.mode(predictions, axis=1)[0].ravel()
            oob_error = 1 - accuracy_score(y_train, ensemble_pred)
            oob_errors.append(oob_error)
        
        ax6.plot(range(1, len(oob_errors) + 1), oob_errors, 'g-', linewidth=2)
        ax6.set_xlabel('树的数量')
        ax6.set_ylabel('OOB误差率')
        ax6.set_title('OOB误差收敛')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'随机森林 - {n_estimators}棵树', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 随机森林优于单棵树")
    print("2. OOB提供无偏估计")
    print("3. 特征重要性有助于特征选择")
    print("4. 树数量增加性能趋于稳定")