"""
8.3 马尔可夫随机场 (Markov Random Fields)
=========================================

马尔可夫随机场(MRF)是无向图模型，用于表示变量之间的对称依赖关系。

核心概念：
1. 无向图：边表示直接依赖，没有方向性
2. 团(Clique)：完全连接的子图
3. 势函数(Potential Function)：定义在团上的非负函数

联合概率分布：
p(x) = (1/Z) Π ψ_c(x_c)

其中：
- ψ_c是团c上的势函数
- Z是配分函数(归一化常数)
- x_c是团c中的变量

马尔可夫性质：
给定节点的邻居，该节点与图中其他节点条件独立。

Hammersley-Clifford定理：
正的概率分布满足马尔可夫性质当且仅当它可以表示为团势函数的乘积。

常见模型：
1. Ising模型：二值变量的格子模型
2. Potts模型：多状态Ising模型
3. 条件随机场(CRF)：判别式模型

与贝叶斯网络的区别：
- MRF使用无向图，BN使用有向图
- MRF不能直接表示因果关系
- MRF的归一化更困难（需要计算配分函数）
- MRF更适合表示局部相互作用
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Callable
from itertools import product, combinations
import warnings
warnings.filterwarnings('ignore')


class MarkovRandomField:
    """
    马尔可夫随机场
    
    使用无向图和势函数定义联合概率分布。
    """
    
    def __init__(self):
        """初始化MRF"""
        self.graph = nx.Graph()  # 无向图
        self.potentials = {}  # 势函数
        self.variable_states = {}  # 变量状态空间
        
    def add_node(self, node: str, states: List[int]) -> None:
        """
        添加节点
        
        Args:
            node: 节点名称
            states: 节点的状态空间
        """
        self.graph.add_node(node)
        self.variable_states[node] = states
        
    def add_edge(self, node1: str, node2: str) -> None:
        """
        添加无向边
        
        Args:
            node1: 第一个节点
            node2: 第二个节点
        """
        self.graph.add_edge(node1, node2)
        
    def add_potential(self, clique: Tuple[str, ...], 
                     potential: np.ndarray) -> None:
        """
        添加团势函数
        
        Args:
            clique: 团（节点元组）
            potential: 势函数表
        """
        # 确保团中的节点已排序（避免重复）
        clique = tuple(sorted(clique))
        self.potentials[clique] = potential
        
    def get_cliques(self) -> List[Set[str]]:
        """
        获取图中的所有极大团
        
        Returns:
            极大团列表
        """
        return list(nx.find_cliques(self.graph))
    
    def is_clique(self, nodes: Set[str]) -> bool:
        """
        检查节点集合是否构成团
        
        Args:
            nodes: 节点集合
            
        Returns:
            是否为团
        """
        # 检查所有节点对是否相连
        for node1, node2 in combinations(nodes, 2):
            if not self.graph.has_edge(node1, node2):
                return False
        return True
    
    def compute_energy(self, assignment: Dict[str, int]) -> float:
        """
        计算给定赋值的能量
        
        能量 E(x) = -Σ log ψ_c(x_c)
        
        Args:
            assignment: 变量赋值
            
        Returns:
            能量值
        """
        energy = 0.0
        
        for clique, potential in self.potentials.items():
            # 获取团中变量的赋值
            indices = []
            for var in clique:
                state = assignment[var]
                idx = self.variable_states[var].index(state)
                indices.append(idx)
            
            # 累加势函数的对数
            pot_value = potential[tuple(indices)]
            if pot_value > 0:
                energy -= np.log(pot_value)
            else:
                energy = np.inf
        
        return energy
    
    def compute_unnormalized_probability(self, 
                                        assignment: Dict[str, int]) -> float:
        """
        计算未归一化的概率
        
        P_unnorm(x) = Π ψ_c(x_c)
        
        Args:
            assignment: 变量赋值
            
        Returns:
            未归一化的概率
        """
        prob = 1.0
        
        for clique, potential in self.potentials.items():
            # 获取团中变量的赋值
            indices = []
            for var in clique:
                state = assignment[var]
                idx = self.variable_states[var].index(state)
                indices.append(idx)
            
            prob *= potential[tuple(indices)]
        
        return prob
    
    def compute_partition_function(self) -> float:
        """
        计算配分函数
        
        Z = Σ_x Π ψ_c(x_c)
        
        警告：对大规模问题计算复杂度为指数级！
        
        Returns:
            配分函数Z
        """
        Z = 0.0
        
        # 生成所有可能的赋值
        variables = list(self.graph.nodes())
        state_lists = [self.variable_states[var] for var in variables]
        
        for states in product(*state_lists):
            assignment = dict(zip(variables, states))
            Z += self.compute_unnormalized_probability(assignment)
        
        return Z
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        获取节点的马尔可夫毯
        
        在MRF中，马尔可夫毯就是节点的邻居。
        
        Args:
            node: 节点名称
            
        Returns:
            马尔可夫毯
        """
        return set(self.graph.neighbors(node))
    
    def visualize(self) -> None:
        """可视化MRF结构"""
        plt.figure(figsize=(10, 8))
        
        # 使用spring布局
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos,
                              node_color='lightcoral',
                              node_size=2000,
                              edgecolors='black',
                              linewidths=2)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='gray',
                              width=2)
        
        # 绘制标签
        nx.draw_networkx_labels(self.graph, pos,
                               font_size=12,
                               font_weight='bold')
        
        # 标记团
        cliques = self.get_cliques()
        for i, clique in enumerate(cliques):
            if len(clique) > 2:  # 只标记大于2的团
                # 计算团中心
                clique_pos = np.mean([pos[node] for node in clique], axis=0)
                plt.text(clique_pos[0], clique_pos[1] + 0.1,
                        f"C{i+1}", fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="yellow", alpha=0.5))
        
        plt.title("马尔可夫随机场结构", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class IsingModel(MarkovRandomField):
    """
    Ising模型
    
    二值变量的格子模型，常用于统计物理和图像处理。
    
    能量函数：
    E(x) = -J Σ_{<i,j>} x_i x_j - h Σ_i x_i
    
    其中：
    - J是耦合强度（正值偏好相同状态）
    - h是外场强度
    - x_i ∈ {-1, +1}
    """
    
    def __init__(self, width: int, height: int,
                 J: float = 1.0, h: float = 0.0):
        """
        初始化Ising模型
        
        Args:
            width: 格子宽度
            height: 格子高度
            J: 耦合强度
            h: 外场强度
        """
        super().__init__()
        self.width = width
        self.height = height
        self.J = J
        self.h = h
        
        # 创建格子
        self._create_lattice()
        
    def _create_lattice(self) -> None:
        """创建二维格子"""
        # 添加节点
        for i in range(self.height):
            for j in range(self.width):
                node = f"({i},{j})"
                self.add_node(node, [-1, 1])
        
        # 添加边（4邻域）
        for i in range(self.height):
            for j in range(self.width):
                node = f"({i},{j})"
                
                # 右邻居
                if j < self.width - 1:
                    neighbor = f"({i},{j+1})"
                    self.add_edge(node, neighbor)
                    # 添加成对势函数
                    potential = np.array([[np.exp(self.J), np.exp(-self.J)],
                                        [np.exp(-self.J), np.exp(self.J)]])
                    self.add_potential((node, neighbor), potential)
                
                # 下邻居
                if i < self.height - 1:
                    neighbor = f"({i+1},{j})"
                    self.add_edge(node, neighbor)
                    # 添加成对势函数
                    potential = np.array([[np.exp(self.J), np.exp(-self.J)],
                                        [np.exp(-self.J), np.exp(self.J)]])
                    self.add_potential((node, neighbor), potential)
                
                # 添加单节点势函数（外场）
                node_potential = np.array([np.exp(-self.h), np.exp(self.h)])
                self.add_potential((node,), node_potential)
    
    def gibbs_sampling(self, n_iterations: int = 1000,
                       initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用Gibbs采样生成样本
        
        Args:
            n_iterations: 迭代次数
            initial_state: 初始状态
            
        Returns:
            最终状态
        """
        # 初始化状态
        if initial_state is None:
            state = np.random.choice([-1, 1], 
                                   size=(self.height, self.width))
        else:
            state = initial_state.copy()
        
        for _ in range(n_iterations):
            # 随机选择一个节点
            i = np.random.randint(self.height)
            j = np.random.randint(self.width)
            
            # 计算条件概率
            # P(x_ij = 1 | neighbors) ∝ exp(J Σ neighbors + h)
            neighbor_sum = 0
            if i > 0:
                neighbor_sum += state[i-1, j]
            if i < self.height - 1:
                neighbor_sum += state[i+1, j]
            if j > 0:
                neighbor_sum += state[i, j-1]
            if j < self.width - 1:
                neighbor_sum += state[i, j+1]
            
            # 计算翻转概率
            delta_E = 2 * (self.J * neighbor_sum + self.h)
            prob_positive = 1 / (1 + np.exp(-delta_E))
            
            # 采样新状态
            if np.random.rand() < prob_positive:
                state[i, j] = 1
            else:
                state[i, j] = -1
        
        return state
    
    def visualize_state(self, state: np.ndarray, title: str = "Ising模型状态") -> None:
        """
        可视化Ising模型状态
        
        Args:
            state: 状态矩阵
            title: 图标题
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(state, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()


def demonstrate_mrf() -> None:
    """演示马尔可夫随机场"""
    print("\n马尔可夫随机场演示")
    print("=" * 60)
    
    # 创建简单的MRF
    mrf = MarkovRandomField()
    
    # 添加节点
    for node in ['A', 'B', 'C', 'D']:
        mrf.add_node(node, [0, 1])
    
    # 添加边（形成环）
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    for edge in edges:
        mrf.add_edge(*edge)
    
    # 添加成对势函数（偏好相同状态）
    potential_same = np.array([[2.0, 1.0],
                              [1.0, 2.0]])
    for edge in edges:
        mrf.add_potential(edge, potential_same)
    
    # 添加单节点势函数
    potential_node = np.array([1.0, 1.5])
    for node in ['A', 'B', 'C', 'D']:
        mrf.add_potential((node,), potential_node)
    
    # 可视化
    print("\nMRF结构：")
    mrf.visualize()
    
    # 计算一些配置的概率
    print("\n未归一化概率：")
    print("-" * 40)
    
    configs = [
        {'A': 0, 'B': 0, 'C': 0, 'D': 0},
        {'A': 1, 'B': 1, 'C': 1, 'D': 1},
        {'A': 0, 'B': 1, 'C': 0, 'D': 1},
    ]
    
    for config in configs:
        prob = mrf.compute_unnormalized_probability(config)
        energy = mrf.compute_energy(config)
        print(f"配置 {config}")
        print(f"  未归一化概率: {prob:.4f}")
        print(f"  能量: {energy:.4f}")
    
    # 计算配分函数
    Z = mrf.compute_partition_function()
    print(f"\n配分函数 Z = {Z:.4f}")
    
    # 马尔可夫毯
    print("\n马尔可夫毯：")
    print("-" * 40)
    for node in mrf.graph.nodes():
        mb = mrf.get_markov_blanket(node)
        print(f"{node}: {mb}")


def demonstrate_ising_model() -> None:
    """演示Ising模型"""
    print("\nIsing模型演示")
    print("=" * 60)
    
    # 创建Ising模型
    width, height = 20, 20
    
    # 不同温度（1/J）下的模型
    temperatures = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, temp in enumerate(temperatures):
        J = 1.0 / temp  # 耦合强度与温度成反比
        ising = IsingModel(width, height, J=J, h=0.0)
        
        print(f"\n温度 T = {temp} (J = {J:.2f}):")
        
        # Gibbs采样
        initial_state = np.random.choice([-1, 1], size=(height, width))
        final_state = ising.gibbs_sampling(n_iterations=5000,
                                          initial_state=initial_state)
        
        # 计算磁化强度
        magnetization = np.mean(final_state)
        print(f"  磁化强度: {magnetization:.3f}")
        
        # 可视化
        ax = axes[idx]
        im = ax.imshow(final_state, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"T = {temp}\nM = {magnetization:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('off')
        
        if idx == 2:
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Ising模型：不同温度下的相变", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n观察：")
    print("1. 低温：系统有序，自旋倾向于对齐")
    print("2. 高温：系统无序，自旋随机分布")
    print("3. 临界温度附近：出现相变")


def compare_bn_mrf() -> None:
    """比较贝叶斯网络和马尔可夫随机场"""
    print("\n贝叶斯网络 vs 马尔可夫随机场")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 贝叶斯网络
    ax1 = axes[0]
    bn = nx.DiGraph()
    bn.add_edges_from([('A', 'C'), ('B', 'C'), ('C', 'D')])
    pos_bn = {'A': (0, 1), 'B': (2, 1), 'C': (1, 0), 'D': (1, -1)}
    
    nx.draw_networkx_nodes(bn, pos_bn, ax=ax1,
                          node_color='lightblue',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(bn, pos_bn, ax=ax1,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          width=2)
    nx.draw_networkx_labels(bn, pos_bn, ax=ax1,
                           font_size=14,
                           font_weight='bold')
    ax1.set_title("贝叶斯网络\nP(A,B,C,D) = P(A)P(B)P(C|A,B)P(D|C)")
    ax1.axis('off')
    
    # 马尔可夫随机场（道德化）
    ax2 = axes[1]
    mrf = nx.Graph()
    mrf.add_edges_from([('A', 'C'), ('B', 'C'), ('C', 'D'), 
                       ('A', 'B')])  # 添加道德边
    pos_mrf = pos_bn
    
    nx.draw_networkx_nodes(mrf, pos_mrf, ax=ax2,
                          node_color='lightcoral',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(mrf, pos_mrf, ax=ax2,
                          edge_color='gray',
                          width=2)
    # 高亮道德边
    nx.draw_networkx_edges(mrf, pos_mrf, ax=ax2,
                          edgelist=[('A', 'B')],
                          edge_color='red',
                          width=3,
                          style='dashed')
    nx.draw_networkx_labels(mrf, pos_mrf, ax=ax2,
                           font_size=14,
                           font_weight='bold')
    ax2.set_title("马尔可夫随机场（道德化）\nP(A,B,C,D) ∝ ψ(A,B,C)ψ(C,D)")
    ax2.axis('off')
    
    plt.suptitle("图模型比较", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n主要区别：")
    print("1. 方向性：BN有向，MRF无向")
    print("2. 参数化：BN用条件概率，MRF用势函数")
    print("3. 归一化：BN自动归一化，MRF需要配分函数")
    print("4. 条件独立：BN用d-分离，MRF用图分离")
    print("5. 道德化：BN转MRF需要添加道德边")