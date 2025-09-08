"""
9.1 K均值聚类 (K-means Clustering)
==================================

K均值是最经典的聚类算法之一。
它通过迭代优化将数据分配到K个簇中。

算法步骤：
1. 初始化K个簇中心
2. E步：将每个数据点分配到最近的簇中心
3. M步：更新簇中心为簇内数据点的均值
4. 重复直到收敛

目标函数（失真度量）：
J = Σᵢ Σₙ rₙₖ ||xₙ - μₖ||²

其中：
- rₙₖ是指示变量（xₙ属于簇k时为1）
- μₖ是簇k的中心

K均值的特点：
1. 简单高效
2. 假设簇是球形的
3. 对初始化敏感
4. 需要预先指定K值
5. 对离群点敏感

与GMM的关系：
K均值可以看作是具有相同球形协方差的GMM的特殊情况。

改进方法：
1. K-means++：改进初始化
2. Mini-batch K-means：处理大数据
3. Kernel K-means：处理非线性可分数据
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class KMeans:
    """
    K均值聚类算法
    
    实现标准K均值算法及其变体。
    """
    
    def __init__(self, n_clusters: int = 3,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 init: str = 'k-means++',
                 n_init: int = 10,
                 random_state: Optional[int] = None):
        """
        初始化K均值
        
        Args:
            n_clusters: 簇的数量K
            max_iter: 最大迭代次数
            tol: 收敛容差
            init: 初始化方法 ('random', 'k-means++')
            n_init: 运行次数（选择最好的结果）
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        
        # 结果
        self.cluster_centers_ = None  # 簇中心
        self.labels_ = None  # 数据点的簇标签
        self.inertia_ = None  # 失真度量（within-cluster sum of squares）
        self.n_iter_ = None  # 实际迭代次数
        
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        初始化簇中心
        
        Args:
            X: 数据，shape (n_samples, n_features)
            
        Returns:
            初始簇中心，shape (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # 随机选择K个数据点作为初始中心
            if self.random_state is not None:
                np.random.seed(self.random_state)
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices].copy()
            
        elif self.init == 'k-means++':
            # K-means++初始化
            # 选择距离已有中心远的点作为新中心
            centroids = []
            
            # 随机选择第一个中心
            if self.random_state is not None:
                np.random.seed(self.random_state)
            first_idx = np.random.randint(n_samples)
            centroids.append(X[first_idx])
            
            # 选择剩余的中心
            for _ in range(1, self.n_clusters):
                # 计算每个点到最近中心的距离
                distances = cdist(X, centroids, metric='euclidean')
                min_distances = np.min(distances, axis=1)
                
                # 根据距离的平方概率选择下一个中心
                probabilities = min_distances ** 2
                probabilities /= probabilities.sum()
                
                next_idx = np.random.choice(n_samples, p=probabilities)
                centroids.append(X[next_idx])
            
            centroids = np.array(centroids)
        else:
            raise ValueError(f"未知的初始化方法: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, 
                        centroids: np.ndarray) -> np.ndarray:
        """
        E步：将数据点分配到最近的簇
        
        Args:
            X: 数据，shape (n_samples, n_features)
            centroids: 簇中心，shape (n_clusters, n_features)
            
        Returns:
            簇标签，shape (n_samples,)
        """
        # 计算每个点到所有中心的距离
        distances = cdist(X, centroids, metric='euclidean')
        
        # 分配到最近的簇
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, 
                         labels: np.ndarray) -> np.ndarray:
        """
        M步：更新簇中心
        
        Args:
            X: 数据，shape (n_samples, n_features)
            labels: 簇标签，shape (n_samples,)
            
        Returns:
            新的簇中心，shape (n_clusters, n_features)
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # 找出属于簇k的点
            mask = labels == k
            if np.any(mask):
                # 计算均值
                centroids[k] = X[mask].mean(axis=0)
            else:
                # 如果簇为空，随机重新初始化
                centroids[k] = X[np.random.randint(len(X))]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                        centroids: np.ndarray) -> float:
        """
        计算失真度量（within-cluster sum of squares）
        
        Args:
            X: 数据
            labels: 簇标签
            centroids: 簇中心
            
        Returns:
            失真度量
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                # 计算簇内点到中心的距离平方和
                cluster_points = X[mask]
                distances = np.linalg.norm(cluster_points - centroids[k], axis=1)
                inertia += np.sum(distances ** 2)
        
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        拟合K均值模型
        
        Args:
            X: 训练数据，shape (n_samples, n_features)
            
        Returns:
            self
        """
        best_inertia = np.inf
        best_result = None
        
        # 运行多次，选择最好的结果
        for run in range(self.n_init):
            # 初始化
            centroids = self._init_centroids(X)
            
            # 迭代优化
            for iteration in range(self.max_iter):
                # E步：分配簇
                labels = self._assign_clusters(X, centroids)
                
                # M步：更新中心
                new_centroids = self._update_centroids(X, labels)
                
                # 检查收敛
                if np.allclose(centroids, new_centroids, atol=self.tol):
                    break
                
                centroids = new_centroids
            
            # 计算失真度量
            inertia = self._compute_inertia(X, labels, centroids)
            
            # 保存最好的结果
            if inertia < best_inertia:
                best_inertia = inertia
                best_result = {
                    'centroids': centroids,
                    'labels': labels,
                    'inertia': inertia,
                    'n_iter': iteration + 1
                }
        
        # 设置最好的结果
        self.cluster_centers_ = best_result['centroids']
        self.labels_ = best_result['labels']
        self.inertia_ = best_result['inertia']
        self.n_iter_ = best_result['n_iter']
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的簇标签
        
        Args:
            X: 数据，shape (n_samples, n_features)
            
        Returns:
            簇标签，shape (n_samples,)
        """
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并预测
        
        Args:
            X: 数据
            
        Returns:
            簇标签
        """
        self.fit(X)
        return self.labels_


class FuzzyKMeans:
    """
    模糊K均值（软聚类）
    
    每个点可以属于多个簇，用隶属度表示。
    
    目标函数：
    J = Σᵢ Σₖ uᵢₖᵐ ||xᵢ - cₖ||²
    
    其中：
    - uᵢₖ是点i对簇k的隶属度
    - m是模糊指数（通常为2）
    """
    
    def __init__(self, n_clusters: int = 3,
                 m: float = 2.0,
                 max_iter: int = 100,
                 tol: float = 1e-4):
        """
        初始化模糊K均值
        
        Args:
            n_clusters: 簇数量
            m: 模糊指数（m>1，越大越模糊）
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        
        # 结果
        self.cluster_centers_ = None
        self.membership_ = None  # 隶属度矩阵
        self.n_iter_ = None
        
    def _init_membership(self, n_samples: int) -> np.ndarray:
        """
        初始化隶属度矩阵
        
        Args:
            n_samples: 样本数
            
        Returns:
            隶属度矩阵，shape (n_samples, n_clusters)
        """
        # 随机初始化，确保每行和为1
        membership = np.random.rand(n_samples, self.n_clusters)
        membership = membership / membership.sum(axis=1, keepdims=True)
        return membership
    
    def _update_centers(self, X: np.ndarray, 
                       membership: np.ndarray) -> np.ndarray:
        """
        更新簇中心
        
        cₖ = Σᵢ uᵢₖᵐ xᵢ / Σᵢ uᵢₖᵐ
        
        Args:
            X: 数据
            membership: 隶属度矩阵
            
        Returns:
            新的簇中心
        """
        # 计算加权中心
        um = membership ** self.m  # shape (n_samples, n_clusters)
        centers = (um.T @ X) / um.sum(axis=0, keepdims=True).T
        return centers
    
    def _update_membership(self, X: np.ndarray, 
                          centers: np.ndarray) -> np.ndarray:
        """
        更新隶属度
        
        uᵢₖ = 1 / Σⱼ (||xᵢ - cₖ|| / ||xᵢ - cⱼ||)^(2/(m-1))
        
        Args:
            X: 数据
            centers: 簇中心
            
        Returns:
            新的隶属度矩阵
        """
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        # 计算距离
        distances = cdist(X, centers, metric='euclidean')
        
        # 更新隶属度
        power = 2 / (self.m - 1)
        for i in range(n_samples):
            for k in range(self.n_clusters):
                if distances[i, k] == 0:
                    # 点恰好在中心上
                    membership[i, :] = 0
                    membership[i, k] = 1
                    break
                else:
                    membership[i, k] = 1 / np.sum((distances[i, k] / distances[i, :]) ** power)
        
        return membership
    
    def fit(self, X: np.ndarray) -> 'FuzzyKMeans':
        """
        拟合模糊K均值
        
        Args:
            X: 数据
            
        Returns:
            self
        """
        n_samples = X.shape[0]
        
        # 初始化隶属度
        membership = self._init_membership(n_samples)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 更新中心
            centers = self._update_centers(X, membership)
            
            # 更新隶属度
            new_membership = self._update_membership(X, centers)
            
            # 检查收敛
            if np.allclose(membership, new_membership, atol=self.tol):
                break
            
            membership = new_membership
        
        self.cluster_centers_ = centers
        self.membership_ = membership
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（返回最大隶属度的簇）
        
        Args:
            X: 数据
            
        Returns:
            簇标签
        """
        membership = self._update_membership(X, self.cluster_centers_)
        return np.argmax(membership, axis=1)


def elbow_method(X: np.ndarray, k_range: range, 
                 show_plot: bool = True) -> List[float]:
    """
    肘部法则选择K值
    
    绘制不同K值的失真度量，寻找"肘部"。
    
    Args:
        X: 数据
        k_range: K值范围
        show_plot: 是否显示图形
        
    Returns:
        每个K的失真度量
    """
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=5)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('簇数量 K', fontsize=12)
        plt.ylabel('失真度量 (Within-cluster sum of squares)', fontsize=12)
        plt.title('肘部法则', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 标记可能的肘部
        if len(inertias) > 2:
            # 计算二阶差分
            diff1 = np.diff(inertias)
            diff2 = np.diff(diff1)
            elbow_idx = np.argmax(diff2) + 1  # +1因为diff减少了长度
            plt.axvline(x=k_range[elbow_idx], color='r', linestyle='--', 
                       label=f'可能的肘部 (K={k_range[elbow_idx]})')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return inertias


def silhouette_analysis(X: np.ndarray, labels: np.ndarray) -> float:
    """
    轮廓系数分析
    
    轮廓系数衡量聚类的紧密度和分离度。
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    其中：
    - a(i)：点i到同簇其他点的平均距离
    - b(i)：点i到最近其他簇的平均距离
    
    Args:
        X: 数据
        labels: 簇标签
        
    Returns:
        平均轮廓系数
    """
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        # 当前点的簇
        current_cluster = labels[i]
        
        # a(i)：到同簇点的平均距离
        same_cluster_mask = labels == current_cluster
        if np.sum(same_cluster_mask) > 1:
            same_cluster_points = X[same_cluster_mask]
            distances_same = np.linalg.norm(same_cluster_points - X[i], axis=1)
            a_i = np.mean(distances_same[distances_same > 0])
        else:
            a_i = 0
        
        # b(i)：到最近其他簇的平均距离
        b_i = np.inf
        for k in range(n_clusters):
            if k != current_cluster:
                other_cluster_mask = labels == k
                if np.any(other_cluster_mask):
                    other_cluster_points = X[other_cluster_mask]
                    distances_other = np.linalg.norm(other_cluster_points - X[i], axis=1)
                    mean_distance = np.mean(distances_other)
                    b_i = min(b_i, mean_distance)
        
        # 轮廓系数
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)


def demonstrate_kmeans(show_plot: bool = True) -> None:
    """
    演示K均值聚类
    """
    print("\nK均值聚类演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    
    # 创建三个簇
    cluster1 = np.random.randn(100, 2) + [2, 2]
    cluster2 = np.random.randn(100, 2) + [-2, 2]
    cluster3 = np.random.randn(100, 2) + [0, -2]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"数据集：{X.shape[0]}个样本，{X.shape[1]}维")
    
    # 标准K均值
    print("\n标准K均值：")
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
    kmeans.fit(X)
    
    print(f"  迭代次数: {kmeans.n_iter_}")
    print(f"  失真度量: {kmeans.inertia_:.2f}")
    
    # 计算轮廓系数
    silhouette_score = silhouette_analysis(X, kmeans.labels_)
    print(f"  轮廓系数: {silhouette_score:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始数据
        ax1 = axes[0, 0]
        ax1.scatter(X[:, 0], X[:, 1], c='gray', s=30, alpha=0.6)
        ax1.set_title('原始数据')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.grid(True, alpha=0.3)
        
        # K均值结果
        ax2 = axes[0, 1]
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, 
                            cmap='viridis', s=30, alpha=0.6)
        ax2.scatter(kmeans.cluster_centers_[:, 0], 
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='*', s=300, edgecolors='black',
                   linewidths=2, label='簇中心')
        ax2.set_title(f'K均值聚类 (K=3)')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2)
        
        # 不同初始化方法比较
        ax3 = axes[1, 0]
        
        methods = ['random', 'k-means++']
        inertias_by_method = {method: [] for method in methods}
        
        for method in methods:
            for _ in range(20):
                km = KMeans(n_clusters=3, init=method, n_init=1)
                km.fit(X)
                inertias_by_method[method].append(km.inertia_)
        
        positions = [1, 2]
        bp = ax3.boxplot([inertias_by_method[m] for m in methods],
                         positions=positions, widths=0.6)
        ax3.set_xticklabels(methods)
        ax3.set_ylabel('失真度量')
        ax3.set_title('初始化方法比较')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 肘部法则
        ax4 = axes[1, 1]
        k_range = range(1, 8)
        inertias = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=5)
            km.fit(X)
            inertias.append(km.inertia_)
        
        ax4.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax4.set_xlabel('簇数量 K')
        ax4.set_ylabel('失真度量')
        ax4.set_title('肘部法则')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=3, color='r', linestyle='--', 
                   label='真实K=3')
        ax4.legend()
        
        plt.suptitle('K均值聚类分析', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. K-means++初始化比随机初始化更稳定")
    print("2. 肘部法则可以帮助选择K值")
    print("3. K均值假设簇是球形的")
    print("4. 对初始化和离群点敏感")


def demonstrate_fuzzy_kmeans(show_plot: bool = True) -> None:
    """
    演示模糊K均值
    """
    print("\n模糊K均值演示")
    print("=" * 60)
    
    # 生成有重叠的数据
    np.random.seed(42)
    
    cluster1 = np.random.randn(100, 2) + [1, 1]
    cluster2 = np.random.randn(100, 2) + [-1, -1]
    X = np.vstack([cluster1, cluster2])
    
    print(f"数据集：{X.shape[0]}个样本，2个重叠的簇")
    
    # 标准K均值
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    
    # 模糊K均值
    fuzzy_kmeans = FuzzyKMeans(n_clusters=2, m=2.0)
    fuzzy_kmeans.fit(X)
    
    print(f"\n模糊K均值：")
    print(f"  迭代次数: {fuzzy_kmeans.n_iter_}")
    print(f"  模糊指数m: 2.0")
    
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 标准K均值
        ax1 = axes[0]
        ax1.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, 
                   cmap='coolwarm', s=30, alpha=0.6)
        ax1.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   c='black', marker='*', s=300, 
                   edgecolors='white', linewidths=2)
        ax1.set_title('标准K均值（硬聚类）')
        ax1.set_xlabel('特征1')
        ax1.set_ylabel('特征2')
        ax1.grid(True, alpha=0.3)
        
        # 模糊K均值 - 簇1隶属度
        ax2 = axes[1]
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], 
                             c=fuzzy_kmeans.membership_[:, 0],
                             cmap='RdBu_r', s=30, vmin=0, vmax=1)
        ax2.scatter(fuzzy_kmeans.cluster_centers_[:, 0],
                   fuzzy_kmeans.cluster_centers_[:, 1],
                   c='black', marker='*', s=300,
                   edgecolors='white', linewidths=2)
        ax2.set_title('模糊K均值 - 簇1隶属度')
        ax2.set_xlabel('特征1')
        ax2.set_ylabel('特征2')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2)
        
        # 不确定性（熵）
        ax3 = axes[2]
        # 计算隶属度的熵
        membership = fuzzy_kmeans.membership_
        entropy = -np.sum(membership * np.log(membership + 1e-10), axis=1)
        entropy = entropy / np.log(2)  # 归一化到[0,1]
        
        scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=entropy,
                             cmap='viridis', s=30)
        ax3.set_title('不确定性（熵）')
        ax3.set_xlabel('特征1')
        ax3.set_ylabel('特征2')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3)
        
        plt.suptitle('硬聚类 vs 软聚类', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 显示一些点的隶属度
    print("\n部分点的隶属度：")
    print("-" * 40)
    print("点索引 | 簇1隶属度 | 簇2隶属度")
    print("-" * 40)
    for i in [0, 50, 100, 150]:
        u1 = fuzzy_kmeans.membership_[i, 0]
        u2 = fuzzy_kmeans.membership_[i, 1]
        print(f"{i:6d} | {u1:9.3f} | {u2:9.3f}")
    
    print("\n观察：")
    print("1. 模糊K均值提供软分配")
    print("2. 边界点有中等隶属度")
    print("3. 可以识别不确定的分配")
    print("4. 适合重叠的簇")