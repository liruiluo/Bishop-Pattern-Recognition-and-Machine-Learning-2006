"""
8.1-8.2 贝叶斯网络 (Bayesian Networks)
======================================

贝叶斯网络是有向无环图(DAG)，用于表示随机变量之间的条件依赖关系。

核心概念：
1. 节点：表示随机变量
2. 有向边：表示条件依赖（因果关系）
3. 条件概率表(CPT)：每个节点都有一个CPT

联合概率分解：
p(x₁,...,xₙ) = Π p(xᵢ|parents(xᵢ))

其中parents(xᵢ)是节点xᵢ的父节点集合。

条件独立性：
贝叶斯网络编码了变量之间的条件独立性。
通过d-分离准则可以判断条件独立性。

三种基本结构：
1. 链式(head-to-tail)：A→B→C，给定B时A和C条件独立
2. 分叉(tail-to-tail)：A←B→C，给定B时A和C条件独立  
3. 汇聚(head-to-head)：A→B←C，B未观测时A和C独立

马尔可夫毯：
节点的马尔可夫毯包括：
- 父节点
- 子节点
- 子节点的其他父节点

给定马尔可夫毯，节点与网络中其他节点条件独立。

应用：
- 医疗诊断
- 故障检测
- 风险评估
- 因果推理
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class BayesianNetwork:
    """
    贝叶斯网络
    
    使用有向无环图表示变量之间的条件依赖关系。
    """
    
    def __init__(self):
        """初始化贝叶斯网络"""
        self.graph = nx.DiGraph()  # 有向图
        self.cpds = {}  # 条件概率分布
        self.variable_states = {}  # 每个变量的状态空间
        
    def add_node(self, node: str, states: List[Any]) -> None:
        """
        添加节点
        
        Args:
            node: 节点名称
            states: 节点的可能状态
        """
        self.graph.add_node(node)
        self.variable_states[node] = states
        
    def add_edge(self, parent: str, child: str) -> None:
        """
        添加有向边
        
        Args:
            parent: 父节点
            child: 子节点
        """
        self.graph.add_edge(parent, child)
        
    def add_cpd(self, node: str, cpd: np.ndarray, 
                evidence: Optional[List[str]] = None) -> None:
        """
        添加条件概率分布
        
        CPD表示P(node|evidence)。
        
        Args:
            node: 节点名称
            cpd: 条件概率表，shape根据父节点数量确定
            evidence: 父节点列表
        """
        if evidence is None:
            evidence = list(self.graph.predecessors(node))
        
        self.cpds[node] = {
            'table': cpd,
            'evidence': evidence,
            'variable': node
        }
        
    def is_valid(self) -> bool:
        """
        检查网络是否有效
        
        Returns:
            是否为有效的贝叶斯网络
        """
        # 检查是否为DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            print("错误：图包含环")
            return False
        
        # 检查每个节点是否有CPD
        for node in self.graph.nodes():
            if node not in self.cpds:
                print(f"错误：节点{node}缺少CPD")
                return False
        
        # 检查CPD维度是否正确
        for node, cpd_info in self.cpds.items():
            cpd = cpd_info['table']
            evidence = cpd_info['evidence']
            
            # 计算期望的形状
            expected_shape = [len(self.variable_states[node])]
            for parent in evidence:
                expected_shape.append(len(self.variable_states[parent]))
            
            if cpd.shape != tuple(expected_shape):
                print(f"错误：节点{node}的CPD维度不匹配")
                print(f"  期望: {expected_shape}, 实际: {cpd.shape}")
                return False
            
            # 检查概率和是否为1
            if len(evidence) > 0:
                # 对每个父节点配置，概率和应为1
                sum_axes = tuple(range(1, len(cpd.shape)))
                if not np.allclose(cpd.sum(axis=0), 1.0):
                    print(f"错误：节点{node}的CPD概率和不为1")
                    return False
            else:
                # 根节点
                if not np.allclose(cpd.sum(), 1.0):
                    print(f"错误：节点{node}的CPD概率和不为1")
                    return False
        
        return True
    
    def get_joint_probability(self, assignment: Dict[str, Any]) -> float:
        """
        计算给定赋值的联合概率
        
        P(x₁,...,xₙ) = Π P(xᵢ|parents(xᵢ))
        
        Args:
            assignment: 变量赋值字典
            
        Returns:
            联合概率
        """
        probability = 1.0
        
        for node in self.graph.nodes():
            cpd_info = self.cpds[node]
            cpd = cpd_info['table']
            evidence = cpd_info['evidence']
            
            # 获取节点状态的索引
            node_state = assignment[node]
            node_idx = self.variable_states[node].index(node_state)
            
            if len(evidence) == 0:
                # 根节点
                probability *= cpd[node_idx]
            else:
                # 有父节点
                # 构建索引 - 注意CPD表的索引顺序
                if len(evidence) == 1:
                    # 单个父节点
                    parent_state = assignment[evidence[0]]
                    parent_idx = self.variable_states[evidence[0]].index(parent_state)
                    probability *= cpd[node_idx, parent_idx]
                else:
                    # 多个父节点
                    parent_indices = []
                    for parent in evidence:
                        parent_state = assignment[parent]
                        parent_idx = self.variable_states[parent].index(parent_state)
                        parent_indices.append(parent_idx)
                    
                    # 使用多维索引
                    probability *= cpd[(node_idx,) + tuple(parent_indices)]
        
        return probability
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        获取节点的马尔可夫毯
        
        马尔可夫毯包括：
        - 父节点
        - 子节点  
        - 子节点的其他父节点
        
        Args:
            node: 节点名称
            
        Returns:
            马尔可夫毯中的节点集合
        """
        markov_blanket = set()
        
        # 添加父节点
        markov_blanket.update(self.graph.predecessors(node))
        
        # 添加子节点
        children = list(self.graph.successors(node))
        markov_blanket.update(children)
        
        # 添加子节点的其他父节点
        for child in children:
            markov_blanket.update(self.graph.predecessors(child))
        
        # 移除节点自身
        markov_blanket.discard(node)
        
        return markov_blanket
    
    def is_d_separated(self, X: Set[str], Y: Set[str], 
                       Z: Set[str]) -> bool:
        """
        判断X和Y是否在给定Z的条件下d-分离
        
        使用贝叶斯球算法(Bayes Ball Algorithm)。
        
        Args:
            X: 节点集合X
            Y: 节点集合Y
            Z: 观测节点集合Z
            
        Returns:
            是否d-分离
        """
        # 找出X到Y的所有路径
        paths = []
        for x in X:
            for y in Y:
                try:
                    # 获取所有简单路径
                    for path in nx.all_simple_paths(
                        self.graph.to_undirected(), x, y
                    ):
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # 检查每条路径是否被阻塞
        for path in paths:
            if not self._is_path_blocked(path, Z):
                return False  # 存在未阻塞的路径
        
        return True  # 所有路径都被阻塞
    
    def _is_path_blocked(self, path: List[str], Z: Set[str]) -> bool:
        """
        检查路径是否被阻塞
        
        Args:
            path: 路径（节点列表）
            Z: 观测节点集合
            
        Returns:
            路径是否被阻塞
        """
        if len(path) < 3:
            # 直接连接的节点
            return False
        
        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            curr_node = path[i]
            next_node = path[i + 1]
            
            # 判断三元组的类型
            prev_to_curr = self.graph.has_edge(prev_node, curr_node)
            curr_to_next = self.graph.has_edge(curr_node, next_node)
            
            if prev_to_curr and curr_to_next:
                # 链式：prev → curr → next
                if curr_node in Z:
                    return True  # 被阻塞
                    
            elif not prev_to_curr and not curr_to_next:
                # 汇聚：prev → curr ← next
                # 检查curr及其后代是否在Z中
                descendants = nx.descendants(self.graph, curr_node)
                if curr_node not in Z and not any(d in Z for d in descendants):
                    return True  # 被阻塞
                    
            else:
                # 分叉：prev ← curr → next 或 prev → curr ← next
                if curr_node in Z:
                    return True  # 被阻塞
        
        return False  # 路径未被阻塞
    
    def visualize(self, show_cpds: bool = False) -> None:
        """
        可视化贝叶斯网络
        
        Args:
            show_cpds: 是否显示CPD表
        """
        plt.figure(figsize=(12, 8))
        
        # 使用分层布局
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color='lightblue',
                              node_size=2000,
                              edgecolors='black',
                              linewidths=2)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              width=2,
                              arrowstyle='->')
        
        # 绘制标签
        nx.draw_networkx_labels(self.graph, pos,
                               font_size=12,
                               font_weight='bold')
        
        plt.title("贝叶斯网络结构", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        if show_cpds:
            print("\n条件概率分布：")
            print("=" * 60)
            for node, cpd_info in self.cpds.items():
                print(f"\n节点: {node}")
                if cpd_info['evidence']:
                    print(f"父节点: {cpd_info['evidence']}")
                print(f"CPD表:\n{cpd_info['table']}")


def create_alarm_network() -> BayesianNetwork:
    """
    创建经典的Alarm网络
    
    这是一个简化版的防盗报警系统：
    - Burglary(入室盗窃)和Earthquake(地震)可能触发Alarm(警报)
    - John和Mary会根据警报打电话
    
    Returns:
        Alarm贝叶斯网络
    """
    bn = BayesianNetwork()
    
    # 添加节点
    bn.add_node('Burglary', [False, True])
    bn.add_node('Earthquake', [False, True])
    bn.add_node('Alarm', [False, True])
    bn.add_node('JohnCalls', [False, True])
    bn.add_node('MaryCalls', [False, True])
    
    # 添加边
    bn.add_edge('Burglary', 'Alarm')
    bn.add_edge('Earthquake', 'Alarm')
    bn.add_edge('Alarm', 'JohnCalls')
    bn.add_edge('Alarm', 'MaryCalls')
    
    # 添加CPD
    # P(Burglary)
    cpd_burglary = np.array([0.999, 0.001])
    bn.add_cpd('Burglary', cpd_burglary)
    
    # P(Earthquake)
    cpd_earthquake = np.array([0.998, 0.002])
    bn.add_cpd('Earthquake', cpd_earthquake)
    
    # P(Alarm|Burglary, Earthquake)
    #                    B=F,E=F  B=F,E=T  B=T,E=F  B=T,E=T
    cpd_alarm = np.array([[0.999,   0.71,    0.06,    0.05],   # Alarm=F
                         [0.001,   0.29,    0.94,    0.95]])  # Alarm=T
    bn.add_cpd('Alarm', cpd_alarm, ['Burglary', 'Earthquake'])
    
    # P(JohnCalls|Alarm)
    #                      A=F    A=T
    cpd_john = np.array([[0.95,  0.10],   # John=F
                        [0.05,  0.90]])   # John=T
    bn.add_cpd('JohnCalls', cpd_john, ['Alarm'])
    
    # P(MaryCalls|Alarm)
    #                      A=F    A=T
    cpd_mary = np.array([[0.99,  0.30],   # Mary=F
                        [0.01,  0.70]])   # Mary=T
    bn.add_cpd('MaryCalls', cpd_mary, ['Alarm'])
    
    return bn


def demonstrate_bayesian_network() -> None:
    """
    演示贝叶斯网络的基本功能
    """
    print("\n贝叶斯网络演示：Alarm网络")
    print("=" * 60)
    
    # 创建Alarm网络
    bn = create_alarm_network()
    
    # 验证网络
    print("\n网络验证:")
    if bn.is_valid():
        print("  ✓ 网络结构有效")
    
    # 可视化
    print("\n网络结构:")
    bn.visualize(show_cpds=False)
    
    # 计算联合概率
    print("\n联合概率计算:")
    print("-" * 40)
    
    # 场景1：没有入室盗窃和地震，警报响了，John打电话了
    assignment1 = {
        'Burglary': False,
        'Earthquake': False,
        'Alarm': True,
        'JohnCalls': True,
        'MaryCalls': False
    }
    prob1 = bn.get_joint_probability(assignment1)
    print(f"P(B=F, E=F, A=T, J=T, M=F) = {prob1:.6f}")
    
    # 场景2：有入室盗窃，警报响了，两人都打电话了
    assignment2 = {
        'Burglary': True,
        'Earthquake': False,
        'Alarm': True,
        'JohnCalls': True,
        'MaryCalls': True
    }
    prob2 = bn.get_joint_probability(assignment2)
    print(f"P(B=T, E=F, A=T, J=T, M=T) = {prob2:.6f}")
    
    # 马尔可夫毯
    print("\n马尔可夫毯:")
    print("-" * 40)
    for node in bn.graph.nodes():
        mb = bn.get_markov_blanket(node)
        print(f"{node}: {mb}")
    
    # 条件独立性
    print("\n条件独立性测试:")
    print("-" * 40)
    
    # Burglary和Earthquake是否独立？
    if bn.is_d_separated({'Burglary'}, {'Earthquake'}, set()):
        print("Burglary ⊥ Earthquake: 是")
    else:
        print("Burglary ⊥ Earthquake: 否")
    
    # 给定Alarm，JohnCalls和MaryCalls是否条件独立？
    if bn.is_d_separated({'JohnCalls'}, {'MaryCalls'}, {'Alarm'}):
        print("JohnCalls ⊥ MaryCalls | Alarm: 是")
    else:
        print("JohnCalls ⊥ MaryCalls | Alarm: 否")
    
    # 给定Alarm，Burglary和JohnCalls是否条件独立？
    if bn.is_d_separated({'Burglary'}, {'JohnCalls'}, {'Alarm'}):
        print("Burglary ⊥ JohnCalls | Alarm: 是")
    else:
        print("Burglary ⊥ JohnCalls | Alarm: 否")


def demonstrate_three_patterns() -> None:
    """
    演示三种基本的条件独立模式
    """
    print("\n三种基本条件独立模式")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 模式1：链式（Head-to-Tail）
    ax1 = axes[0]
    G1 = nx.DiGraph()
    G1.add_edges_from([('A', 'B'), ('B', 'C')])
    pos1 = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0)}
    
    nx.draw_networkx_nodes(G1, pos1, ax=ax1,
                          node_color='lightblue',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(G1, pos1, ax=ax1,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          width=2)
    nx.draw_networkx_labels(G1, pos1, ax=ax1,
                           font_size=14,
                           font_weight='bold')
    ax1.set_title("链式：A→B→C\nA⊥C|B", fontsize=12)
    ax1.axis('off')
    
    # 模式2：分叉（Tail-to-Tail）
    ax2 = axes[1]
    G2 = nx.DiGraph()
    G2.add_edges_from([('B', 'A'), ('B', 'C')])
    pos2 = {'A': (0, 0), 'B': (1, 0.5), 'C': (2, 0)}
    
    nx.draw_networkx_nodes(G2, pos2, ax=ax2,
                          node_color='lightblue',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(G2, pos2, ax=ax2,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          width=2)
    nx.draw_networkx_labels(G2, pos2, ax=ax2,
                           font_size=14,
                           font_weight='bold')
    ax2.set_title("分叉：A←B→C\nA⊥C|B", fontsize=12)
    ax2.axis('off')
    
    # 模式3：汇聚（Head-to-Head）
    ax3 = axes[2]
    G3 = nx.DiGraph()
    G3.add_edges_from([('A', 'B'), ('C', 'B')])
    pos3 = {'A': (0, 0), 'B': (1, -0.5), 'C': (2, 0)}
    
    nx.draw_networkx_nodes(G3, pos3, ax=ax3,
                          node_color='lightblue',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(G3, pos3, ax=ax3,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          width=2)
    nx.draw_networkx_labels(G3, pos3, ax=ax3,
                           font_size=14,
                           font_weight='bold')
    ax3.set_title("汇聚：A→B←C\nA⊥C (B未观测)\nA⊥̸C|B", fontsize=12)
    ax3.axis('off')
    
    plt.suptitle("贝叶斯网络中的条件独立模式", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n说明：")
    print("1. 链式和分叉：观测中间节点阻塞路径")
    print("2. 汇聚（V结构）：观测汇聚节点打开路径")
    print("3. 这些模式是d-分离的基础")