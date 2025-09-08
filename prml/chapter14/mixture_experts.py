"""
14.6 混合专家模型 (Mixture of Experts)
=======================================

混合专家模型通过门控网络动态选择局部专家。

模型结构：
- 专家网络：E_i(x), i=1,...,K
- 门控网络：g(x) = softmax(W_g^T x)
- 输出：y = Σ_i g_i(x) E_i(x)

训练：
EM算法或梯度下降联合优化专家和门控网络。

E步：计算专家的后验责任
h_ij = g_j(x_i) P(y_i|x_i, θ_j) / Σ_k g_k(x_i) P(y_i|x_i, θ_k)

M步：更新专家和门控参数
- 专家：加权最大似然
- 门控：多类逻辑回归

优势：
1. 自动分工：不同专家处理不同区域
2. 软划分：平滑过渡
3. 可解释：门控权重显示专家分工

扩展：
- 层次混合专家(HME)：树状结构
- 无限混合专家：非参数贝叶斯
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


class MixtureOfExperts(BaseEstimator, RegressorMixin):
    """
    混合专家模型（回归）
    
    通过门控网络组合多个专家的预测。
    每个专家负责输入空间的一个区域。
    
    算法：
    1. 门控网络计算每个专家的权重
    2. 专家网络做出预测
    3. 加权组合得到最终输出
    
    训练使用EM算法或端到端梯度下降。
    """
    
    def __init__(self, n_experts: int = 4,
                 expert_type: str = 'linear',
                 gating_type: str = 'softmax',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        初始化混合专家模型
        
        Args:
            n_experts: 专家数量
            expert_type: 专家类型 ('linear', 'mlp', 'rbf')
            gating_type: 门控类型 ('softmax', 'hierarchical')
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_experts = n_experts
        self.expert_type = expert_type
        self.gating_type = gating_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 内部变量
        self.experts_ = []
        self.gating_network_ = None
        self.responsibilities_ = None
        self.training_log_likelihood_ = []
        
    def _create_expert(self) -> BaseEstimator:
        """创建专家网络"""
        if self.expert_type == 'linear':
            return LinearRegression()
        elif self.expert_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(10,),
                max_iter=200,
                random_state=self.random_state
            )
        elif self.expert_type == 'rbf':
            # 简化：使用带RBF核的岭回归
            from sklearn.kernel_ridge import KernelRidge
            return KernelRidge(kernel='rbf', gamma=0.1)
        else:
            raise ValueError(f"不支持的专家类型: {self.expert_type}")
    
    def _create_gating_network(self, n_features: int) -> BaseEstimator:
        """创建门控网络"""
        if self.gating_type == 'softmax':
            # 多类逻辑回归作为门控网络
            return LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=200,
                random_state=self.random_state
            )
        elif self.gating_type == 'hierarchical':
            # 层次门控（简化版）
            return LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=200,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的门控类型: {self.gating_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MixtureOfExperts':
        """
        训练混合专家模型
        
        使用EM算法迭代优化：
        E步：计算专家的责任（后验概率）
        M步：更新专家和门控网络
        """
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化专家
        self.experts_ = [self._create_expert() for _ in range(self.n_experts)]
        
        # 初始化门控网络
        self.gating_network_ = self._create_gating_network(n_features)
        
        # 初始化责任矩阵（随机）
        self.responsibilities_ = np.random.dirichlet(
            np.ones(self.n_experts), size=n_samples
        )
        
        # 为门控网络创建初始标签
        initial_labels = np.argmax(self.responsibilities_, axis=1)
        
        # EM算法
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            # M步：更新模型参数
            
            # 更新专家
            for k in range(self.n_experts):
                # 使用加权最小二乘
                weights = self.responsibilities_[:, k]
                
                # 只使用有足够权重的样本
                mask = weights > 1e-10
                if np.sum(mask) > 1:
                    # 对于线性回归，可以直接使用样本权重
                    if hasattr(self.experts_[k], 'sample_weight'):
                        self.experts_[k].fit(X[mask], y[mask], 
                                           sample_weight=weights[mask])
                    else:
                        # 否则使用重采样近似
                        n_samples_expert = max(10, int(np.sum(weights) * 2))
                        indices = np.random.choice(
                            np.where(mask)[0],
                            size=n_samples_expert,
                            p=weights[mask] / np.sum(weights[mask]),
                            replace=True
                        )
                        self.experts_[k].fit(X[indices], y[indices])
            
            # 更新门控网络
            # 使用责任作为软标签
            expert_labels = np.argmax(self.responsibilities_, axis=1)
            self.gating_network_.fit(X, expert_labels)
            
            # E步：计算责任
            
            # 获取门控权重
            gating_weights = self.gating_network_.predict_proba(X)
            
            # 计算每个专家的似然
            expert_likelihoods = np.zeros((n_samples, self.n_experts))
            
            for k in range(self.n_experts):
                # 预测
                y_pred = self.experts_[k].predict(X)
                
                # 假设高斯噪声，计算似然
                # 简化：使用固定方差
                variance = 1.0
                likelihood = np.exp(-0.5 * (y - y_pred)**2 / variance)
                likelihood /= np.sqrt(2 * np.pi * variance)
                
                expert_likelihoods[:, k] = likelihood
            
            # 计算后验责任
            weighted_likelihoods = gating_weights * expert_likelihoods
            self.responsibilities_ = weighted_likelihoods / (
                np.sum(weighted_likelihoods, axis=1, keepdims=True) + 1e-10
            )
            
            # 计算对数似然
            log_likelihood = np.sum(np.log(
                np.sum(weighted_likelihoods, axis=1) + 1e-10
            ))
            self.training_log_likelihood_.append(log_likelihood)
            
            # 检查收敛
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"EM算法收敛于{iteration+1}次迭代")
                break
            
            prev_ll = log_likelihood
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        组合所有专家的加权预测。
        """
        n_samples = X.shape[0]
        
        # 获取门控权重
        gating_weights = self.gating_network_.predict_proba(X)
        
        # 获取专家预测
        expert_predictions = np.zeros((n_samples, self.n_experts))
        for k in range(self.n_experts):
            expert_predictions[:, k] = self.experts_[k].predict(X)
        
        # 加权组合
        predictions = np.sum(gating_weights * expert_predictions, axis=1)
        
        return predictions
    
    def predict_expert_weights(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的专家权重"""
        return self.gating_network_.predict_proba(X)


class HierarchicalMixtureOfExperts(BaseEstimator, RegressorMixin):
    """
    层次混合专家模型
    
    树状结构的混合专家，每个节点都是一个门控网络。
    
    结构：
    - 根节点：顶层门控
    - 中间节点：子门控
    - 叶节点：专家网络
    
    优势：
    - 更复杂的划分
    - 层次化决策
    - 可解释的树结构
    """
    
    def __init__(self, n_levels: int = 2,
                 n_experts_per_node: int = 2,
                 expert_type: str = 'linear',
                 random_state: Optional[int] = None):
        """
        初始化层次混合专家
        
        Args:
            n_levels: 层数
            n_experts_per_node: 每个节点的分支数
            expert_type: 专家类型
            random_state: 随机种子
        """
        self.n_levels = n_levels
        self.n_experts_per_node = n_experts_per_node
        self.expert_type = expert_type
        self.random_state = random_state
        
        # 计算总专家数
        self.n_experts = n_experts_per_node ** n_levels
        
        # 内部变量
        self.tree_ = None
        
    class TreeNode:
        """树节点"""
        def __init__(self, level: int, is_leaf: bool = False):
            self.level = level
            self.is_leaf = is_leaf
            self.gating_network = None
            self.children = []
            self.expert = None
    
    def _build_tree(self, level: int) -> 'TreeNode':
        """递归构建树结构"""
        if level >= self.n_levels:
            # 叶节点：专家
            node = self.TreeNode(level, is_leaf=True)
            if self.expert_type == 'linear':
                node.expert = LinearRegression()
            else:
                node.expert = MLPRegressor(
                    hidden_layer_sizes=(10,),
                    random_state=self.random_state
                )
            return node
        else:
            # 内部节点：门控
            node = self.TreeNode(level)
            node.gating_network = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                random_state=self.random_state
            )
            
            # 创建子节点
            for _ in range(self.n_experts_per_node):
                child = self._build_tree(level + 1)
                node.children.append(child)
            
            return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HierarchicalMixtureOfExperts':
        """训练层次混合专家"""
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 构建树
        self.tree_ = self._build_tree(0)
        
        # 简化训练：使用k-means初始化划分
        from sklearn.cluster import KMeans
        
        # 递归训练
        def train_node(node, X_node, y_node, depth=0):
            if node.is_leaf:
                # 训练专家
                if len(X_node) > 0:
                    node.expert.fit(X_node, y_node)
            else:
                # 使用聚类初始化划分
                if len(X_node) > self.n_experts_per_node:
                    kmeans = KMeans(
                        n_clusters=self.n_experts_per_node,
                        random_state=self.random_state
                    )
                    labels = kmeans.fit_predict(X_node)
                else:
                    labels = np.random.choice(
                        self.n_experts_per_node,
                        size=len(X_node)
                    )
                
                # 训练门控网络
                if len(np.unique(labels)) > 1:
                    node.gating_network.fit(X_node, labels)
                else:
                    # 如果只有一个类，创建默认门控
                    node.gating_network.fit(
                        np.vstack([X_node, X_node[:1]]),
                        np.array([0] * len(X_node) + [1])
                    )
                
                # 递归训练子节点
                for i, child in enumerate(node.children):
                    mask = labels == i
                    if np.sum(mask) > 0:
                        train_node(child, X_node[mask], y_node[mask], depth + 1)
        
        train_node(self.tree_, X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            predictions[i] = self._predict_sample(X[i:i+1])
        
        return predictions
    
    def _predict_sample(self, x: np.ndarray) -> float:
        """预测单个样本"""
        def predict_recursive(node, weight=1.0):
            if node.is_leaf:
                # 专家预测
                return weight * node.expert.predict(x)[0]
            else:
                # 获取门控权重
                gating_weights = node.gating_network.predict_proba(x)[0]
                
                # 递归预测
                prediction = 0
                for i, child in enumerate(node.children):
                    prediction += predict_recursive(
                        child, weight * gating_weights[i]
                    )
                
                return prediction
        
        return predict_recursive(self.tree_)


def demonstrate_mixture_of_experts(n_experts: int = 4,
                                  show_plot: bool = True) -> None:
    """
    演示混合专家模型
    
    展示专家的自动分工和软划分。
    """
    print("\n混合专家模型演示")
    print("=" * 60)
    
    # 生成具有不同区域特性的数据
    np.random.seed(42)
    n_samples = 400
    
    # 生成四个不同的区域
    X = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        x1, x2 = X[i]
        
        if x1 < 0 and x2 < 0:
            # 区域1：线性
            y[i] = 2 * x1 + x2 + np.random.randn() * 0.1
        elif x1 >= 0 and x2 < 0:
            # 区域2：二次
            y[i] = x1**2 - x2 + np.random.randn() * 0.1
        elif x1 < 0 and x2 >= 0:
            # 区域3：正弦
            y[i] = np.sin(2 * x1) + x2 + np.random.randn() * 0.1
        else:
            # 区域4：指数
            y[i] = np.exp(0.5 * x1) - x2 + np.random.randn() * 0.1
    
    # 分割数据
    n_train = 300
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    print("数据特性：")
    print("  区域1（左下）：线性关系")
    print("  区域2（右下）：二次关系")
    print("  区域3（左上）：正弦关系")
    print("  区域4（右上）：指数关系")
    
    # 训练混合专家模型
    moe = MixtureOfExperts(
        n_experts=n_experts,
        expert_type='linear',
        max_iter=100,
        random_state=42
    )
    moe.fit(X_train, y_train)
    
    # 预测
    y_pred_train = moe.predict(X_train)
    y_pred_test = moe.predict(X_test)
    
    # 评估
    from sklearn.metrics import mean_squared_error, r2_score
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n混合专家模型性能：")
    print(f"  训练MSE: {train_mse:.4f}, R²: {train_r2:.3f}")
    print(f"  测试MSE: {test_mse:.4f}, R²: {test_r2:.3f}")
    
    # 单个全局模型对比
    global_model = LinearRegression()
    global_model.fit(X_train, y_train)
    y_pred_global = global_model.predict(X_test)
    global_mse = mean_squared_error(y_test, y_pred_global)
    global_r2 = r2_score(y_test, y_pred_global)
    
    print(f"\n全局线性模型性能：")
    print(f"  测试MSE: {global_mse:.4f}, R²: {global_r2:.3f}")
    
    # 分析专家分工
    expert_weights = moe.predict_expert_weights(X)
    dominant_expert = np.argmax(expert_weights, axis=1)
    
    print(f"\n专家使用统计：")
    for k in range(n_experts):
        count = np.sum(dominant_expert == k)
        print(f"  专家{k+1}: {count}个样本 ({count/n_samples:.1%})")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 真实函数
        ax1 = axes[0, 0]
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20)
        plt.colorbar(scatter, ax=ax1)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('真实函数')
        
        # 添加区域分割线
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # MoE预测
        ax2 = axes[0, 1]
        y_pred_all = moe.predict(X)
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=y_pred_all, cmap='viridis', s=20)
        plt.colorbar(scatter, ax=ax2)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_title(f'MoE预测 (R²={test_r2:.3f})')
        
        # 专家分配
        ax3 = axes[0, 2]
        scatter = ax3.scatter(X[:, 0], X[:, 1], c=dominant_expert, 
                            cmap='tab10', s=20)
        plt.colorbar(scatter, ax=ax3)
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_title('主导专家分配')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 各专家权重
        for k in range(min(3, n_experts)):
            ax = axes[1, k]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=expert_weights[:, k],
                               cmap='YlOrRd', s=20, vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'专家{k+1}权重')
            ax.axvline(x=0, color='b', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='b', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'混合专家模型 - {n_experts}个专家', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # 第二个图：训练过程和性能分析
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        
        # 训练曲线
        ax1 = axes2[0]
        if len(moe.training_log_likelihood_) > 0:
            ax1.plot(moe.training_log_likelihood_, 'b-', linewidth=2)
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('对数似然')
            ax1.set_title('训练过程')
            ax1.grid(True, alpha=0.3)
        
        # 预测vs真实
        ax2 = axes2[1]
        ax2.scatter(y_test, y_pred_test, alpha=0.5, s=20)
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        ax2.set_title(f'预测准确性 (MSE={test_mse:.4f})')
        ax2.grid(True, alpha=0.3)
        
        # 残差分布
        ax3 = axes2[2]
        residuals = y_test - y_pred_test
        ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('残差')
        ax3.set_ylabel('频数')
        ax3.set_title(f'残差分布 (均值={np.mean(residuals):.4f})')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('混合专家模型分析', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 演示层次混合专家
    print("\n层次混合专家模型演示")
    print("=" * 60)
    
    hme = HierarchicalMixtureOfExperts(
        n_levels=2,
        n_experts_per_node=2,
        expert_type='linear',
        random_state=42
    )
    
    hme.fit(X_train, y_train)
    y_pred_hme = hme.predict(X_test)
    
    hme_mse = mean_squared_error(y_test, y_pred_hme)
    hme_r2 = r2_score(y_test, y_pred_hme)
    
    print(f"层次MoE性能：")
    print(f"  测试MSE: {hme_mse:.4f}, R²: {hme_r2:.3f}")
    
    print("\n模型比较：")
    print(f"  全局线性: MSE={global_mse:.4f}")
    print(f"  标准MoE: MSE={test_mse:.4f}")
    print(f"  层次MoE: MSE={hme_mse:.4f}")
    
    print("\n观察：")
    print("1. MoE自动发现数据的局部结构")
    print("2. 不同专家负责不同区域")
    print("3. 软划分提供平滑过渡")
    print("4. 性能优于全局模型")