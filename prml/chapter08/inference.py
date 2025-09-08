"""
8.4 图模型中的推断 (Inference in Graphical Models)
==================================================

推断是图模型的核心任务之一，主要包括：
1. 边际推断：计算P(X_i)
2. 条件推断：计算P(X_i|E=e)
3. MAP推断：找到最可能的配置

精确推断算法：
1. 变量消除(Variable Elimination)
2. 信念传播(Belief Propagation)
3. 联结树算法(Junction Tree)

近似推断算法：
1. 循环信念传播(Loopy BP)
2. 蒙特卡洛方法(MCMC)
3. 变分推断(Variational Inference)

复杂度：
精确推断通常是NP-hard的，复杂度取决于图的树宽。
对于树结构，推断是多项式时间的。

消息传递：
许多推断算法可以统一在消息传递框架下：
- Sum-Product：计算边际概率
- Max-Product：计算最大后验配置
- Min-Sum：在对数域的Max-Product
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import networkx as nx
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class VariableElimination:
    """
    变量消除算法
    
    通过逐个消除变量来计算边际概率。
    消除顺序影响计算复杂度。
    """
    
    def __init__(self, factors: List[Dict]):
        """
        初始化
        
        Args:
            factors: 因子列表，每个因子包含：
                    - 'vars': 变量列表
                    - 'table': 概率表(numpy数组)
                    - 'states': 每个变量的状态数
        """
        self.factors = factors
        
    def eliminate_variable(self, var: str, 
                          factors: List[Dict]) -> List[Dict]:
        """
        消除一个变量
        
        步骤：
        1. 找出包含该变量的所有因子
        2. 将这些因子相乘
        3. 对该变量求和（边际化）
        
        Args:
            var: 要消除的变量
            factors: 当前因子列表
            
        Returns:
            更新后的因子列表
        """
        # 分离包含和不包含该变量的因子
        containing = []
        not_containing = []
        
        for factor in factors:
            if var in factor['vars']:
                containing.append(factor)
            else:
                not_containing.append(factor)
        
        if not containing:
            return factors
        
        # 将包含该变量的因子相乘
        product_factor = self.multiply_factors(containing)
        
        # 边际化（对该变量求和）
        marginalized = self.marginalize(product_factor, var)
        
        # 返回更新的因子列表
        if marginalized is not None:
            not_containing.append(marginalized)
        
        return not_containing
    
    def multiply_factors(self, factors: List[Dict]) -> Dict:
        """
        将多个因子相乘
        
        Args:
            factors: 因子列表
            
        Returns:
            乘积因子
        """
        if not factors:
            return None
        
        if len(factors) == 1:
            return factors[0].copy()
        
        # 获取所有变量的并集
        all_vars = []
        seen_vars = set()
        for factor in factors:
            for var in factor['vars']:
                if var not in seen_vars:
                    all_vars.append(var)
                    seen_vars.add(var)
        
        # 确定结果表的形状
        shape = []
        states = {}
        for var in all_vars:
            for factor in factors:
                if var in factor['vars']:
                    var_states = factor['states'][var]
                    shape.append(var_states)
                    states[var] = var_states
                    break
        
        # 初始化结果表
        result_table = np.ones(shape)
        
        # 计算乘积
        for idx in np.ndindex(*shape):
            assignment = dict(zip(all_vars, idx))
            
            for factor in factors:
                # 获取因子中相关变量的索引
                factor_idx = []
                for var in factor['vars']:
                    factor_idx.append(assignment[var])
                
                result_table[idx] *= factor['table'][tuple(factor_idx)]
        
        return {
            'vars': all_vars,
            'table': result_table,
            'states': states
        }
    
    def marginalize(self, factor: Dict, var: str) -> Optional[Dict]:
        """
        对变量进行边际化（求和）
        
        Args:
            factor: 因子
            var: 要边际化的变量
            
        Returns:
            边际化后的因子
        """
        if var not in factor['vars']:
            return factor
        
        # 找到变量的轴
        var_idx = factor['vars'].index(var)
        
        # 对该轴求和
        new_table = np.sum(factor['table'], axis=var_idx)
        
        # 更新变量列表和状态
        new_vars = [v for v in factor['vars'] if v != var]
        new_states = {v: s for v, s in factor['states'].items() if v != var}
        
        if not new_vars:
            # 所有变量都被消除，返回标量
            return None
        
        return {
            'vars': new_vars,
            'table': new_table,
            'states': new_states
        }
    
    def query(self, query_vars: List[str], 
             evidence: Optional[Dict[str, int]] = None,
             elimination_order: Optional[List[str]] = None) -> np.ndarray:
        """
        查询变量的边际概率
        
        Args:
            query_vars: 查询变量
            evidence: 证据变量及其值
            elimination_order: 消除顺序
            
        Returns:
            查询变量的边际概率表
        """
        # 复制因子
        factors = [f.copy() for f in self.factors]
        
        # 处理证据
        if evidence:
            factors = self.condition_on_evidence(factors, evidence)
        
        # 确定消除顺序
        all_vars = set()
        for factor in factors:
            all_vars.update(factor['vars'])
        
        vars_to_eliminate = all_vars - set(query_vars)
        if evidence:
            vars_to_eliminate -= set(evidence.keys())
        
        if elimination_order is None:
            elimination_order = list(vars_to_eliminate)
        
        # 逐个消除变量
        for var in elimination_order:
            if var in vars_to_eliminate:
                factors = self.eliminate_variable(var, factors)
        
        # 将剩余因子相乘
        result = self.multiply_factors(factors)
        
        # 归一化
        if result is not None:
            result['table'] = result['table'] / np.sum(result['table'])
        
        return result
    
    def condition_on_evidence(self, factors: List[Dict], 
                            evidence: Dict[str, int]) -> List[Dict]:
        """
        根据证据条件化因子
        
        Args:
            factors: 因子列表
            evidence: 证据
            
        Returns:
            条件化后的因子
        """
        conditioned_factors = []
        
        for factor in factors:
            # 检查因子是否包含证据变量
            evidence_in_factor = {var: val for var, val in evidence.items() 
                                 if var in factor['vars']}
            
            if not evidence_in_factor:
                conditioned_factors.append(factor)
                continue
            
            # 创建条件化的因子
            new_vars = [v for v in factor['vars'] 
                       if v not in evidence_in_factor]
            
            if not new_vars:
                # 因子变成常数
                continue
            
            # 提取相关的概率值
            slices = []
            new_states = {}
            for var in factor['vars']:
                if var in evidence_in_factor:
                    slices.append(evidence_in_factor[var])
                else:
                    slices.append(slice(None))
                    new_states[var] = factor['states'][var]
            
            new_table = factor['table'][tuple(slices)]
            
            conditioned_factors.append({
                'vars': new_vars,
                'table': new_table,
                'states': new_states
            })
        
        return conditioned_factors


class BeliefPropagation:
    """
    信念传播算法（Sum-Product算法）
    
    在树结构上是精确的，在有环图上是近似的。
    """
    
    def __init__(self, graph: nx.Graph, node_potentials: Dict,
                 edge_potentials: Dict):
        """
        初始化
        
        Args:
            graph: 图结构
            node_potentials: 节点势函数
            edge_potentials: 边势函数
        """
        self.graph = graph
        self.node_potentials = node_potentials
        self.edge_potentials = edge_potentials
        
        # 消息
        self.messages = defaultdict(lambda: defaultdict(lambda: None))
        
    def send_message(self, from_node: str, to_node: str) -> np.ndarray:
        """
        从from_node发送消息到to_node
        
        消息计算：
        m_{i→j}(x_j) = Σ_{x_i} ψ_i(x_i) ψ_{ij}(x_i, x_j) Π_{k∈N(i)\j} m_{k→i}(x_i)
        
        Args:
            from_node: 发送节点
            to_node: 接收节点
            
        Returns:
            消息（概率分布）
        """
        # 获取节点势
        node_pot = self.node_potentials[from_node]
        n_states_from = len(node_pot)
        
        # 获取边势
        if (from_node, to_node) in self.edge_potentials:
            edge_pot = self.edge_potentials[(from_node, to_node)]
        else:
            edge_pot = self.edge_potentials[(to_node, from_node)].T
        
        n_states_to = edge_pot.shape[1]
        
        # 收集来自其他邻居的消息
        incoming_messages = np.ones(n_states_from)
        for neighbor in self.graph.neighbors(from_node):
            if neighbor != to_node:
                msg = self.messages[neighbor][from_node]
                if msg is not None:
                    incoming_messages *= msg
        
        # 计算消息
        message = np.zeros(n_states_to)
        for j in range(n_states_to):
            for i in range(n_states_from):
                message[j] += (node_pot[i] * edge_pot[i, j] * 
                              incoming_messages[i])
        
        # 归一化（避免数值问题）
        message = message / (np.sum(message) + 1e-10)
        
        return message
    
    def run(self, max_iterations: int = 100, 
           tolerance: float = 1e-6) -> Dict[str, np.ndarray]:
        """
        运行信念传播算法
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            每个节点的边际概率
        """
        nodes = list(self.graph.nodes())
        
        # 初始化消息为均匀分布
        for edge in self.graph.edges():
            i, j = edge
            n_states_i = len(self.node_potentials[i])
            n_states_j = len(self.node_potentials[j])
            self.messages[i][j] = np.ones(n_states_j) / n_states_j
            self.messages[j][i] = np.ones(n_states_i) / n_states_i
        
        # 迭代更新消息
        for iteration in range(max_iterations):
            old_messages = {(i, j): self.messages[i][j].copy() 
                          for i, j in self.graph.edges() 
                          for _ in range(2)}  # 双向
            
            # 更新所有消息
            for edge in self.graph.edges():
                i, j = edge
                # i → j
                self.messages[i][j] = self.send_message(i, j)
                # j → i
                self.messages[j][i] = self.send_message(j, i)
            
            # 检查收敛
            converged = True
            for edge in self.graph.edges():
                i, j = edge
                if (np.max(np.abs(self.messages[i][j] - old_messages[(i, j)])) > tolerance or
                    np.max(np.abs(self.messages[j][i] - old_messages[(j, i)])) > tolerance):
                    converged = False
                    break
            
            if converged:
                print(f"信念传播在{iteration}次迭代后收敛")
                break
        
        # 计算边际概率（信念）
        beliefs = {}
        for node in nodes:
            belief = self.node_potentials[node].copy()
            
            # 乘以所有传入消息
            for neighbor in self.graph.neighbors(node):
                msg = self.messages[neighbor][node]
                if msg is not None:
                    belief *= msg
            
            # 归一化
            belief = belief / np.sum(belief)
            beliefs[node] = belief
        
        return beliefs


def demonstrate_variable_elimination() -> None:
    """演示变量消除算法"""
    print("\n变量消除算法演示")
    print("=" * 60)
    
    # 创建简单的贝叶斯网络：A → B → C
    print("\n网络结构：A → B → C")
    
    # 定义因子
    # P(A)
    factor_A = {
        'vars': ['A'],
        'table': np.array([0.3, 0.7]),  # P(A=0), P(A=1)
        'states': {'A': 2}
    }
    
    # P(B|A)
    factor_B = {
        'vars': ['B', 'A'],
        'table': np.array([[0.8, 0.3],   # P(B=0|A=0), P(B=0|A=1)
                          [0.2, 0.7]]),  # P(B=1|A=0), P(B=1|A=1)
        'states': {'B': 2, 'A': 2}
    }
    
    # P(C|B)
    factor_C = {
        'vars': ['C', 'B'],
        'table': np.array([[0.9, 0.4],   # P(C=0|B=0), P(C=0|B=1)
                          [0.1, 0.6]]),  # P(C=1|B=0), P(C=1|B=1)
        'states': {'C': 2, 'B': 2}
    }
    
    # 创建变量消除对象
    ve = VariableElimination([factor_A, factor_B, factor_C])
    
    # 查询P(C)
    print("\n查询P(C):")
    result = ve.query(['C'])
    print(f"P(C=0) = {result['table'][0]:.4f}")
    print(f"P(C=1) = {result['table'][1]:.4f}")
    
    # 查询P(C|A=1)
    print("\n查询P(C|A=1):")
    result = ve.query(['C'], evidence={'A': 1})
    print(f"P(C=0|A=1) = {result['table'][0]:.4f}")
    print(f"P(C=1|A=1) = {result['table'][1]:.4f}")
    
    # 查询P(A|C=1)（反向推理）
    print("\n查询P(A|C=1):")
    result = ve.query(['A'], evidence={'C': 1})
    print(f"P(A=0|C=1) = {result['table'][0]:.4f}")
    print(f"P(A=1|C=1) = {result['table'][1]:.4f}")
    
    print("\n观察：")
    print("1. 变量消除可以处理任意查询")
    print("2. 消除顺序影响效率但不影响结果")
    print("3. 可以进行反向推理（诊断）")


def demonstrate_belief_propagation() -> None:
    """演示信念传播算法"""
    print("\n信念传播算法演示")
    print("=" * 60)
    
    # 创建简单的马尔可夫链：A - B - C - D
    graph = nx.Graph()
    graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    
    print("\n网络结构：A — B — C — D")
    
    # 定义节点势函数（先验）
    node_potentials = {
        'A': np.array([0.3, 0.7]),
        'B': np.array([0.5, 0.5]),
        'C': np.array([0.6, 0.4]),
        'D': np.array([0.2, 0.8])
    }
    
    # 定义边势函数（偏好相同状态）
    same_state_preference = np.array([[2.0, 0.5],
                                     [0.5, 2.0]])
    edge_potentials = {
        ('A', 'B'): same_state_preference,
        ('B', 'C'): same_state_preference,
        ('C', 'D'): same_state_preference
    }
    
    # 运行信念传播
    bp = BeliefPropagation(graph, node_potentials, edge_potentials)
    beliefs = bp.run(max_iterations=100)
    
    print("\n边际概率（信念）：")
    print("-" * 40)
    for node in sorted(beliefs.keys()):
        belief = beliefs[node]
        print(f"{node}: P(0)={belief[0]:.4f}, P(1)={belief[1]:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 网络结构
    ax1 = axes[0]
    pos = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0)}
    nx.draw_networkx_nodes(graph, pos, ax=ax1,
                          node_color='lightblue',
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    nx.draw_networkx_edges(graph, pos, ax=ax1,
                          edge_color='gray',
                          width=2)
    nx.draw_networkx_labels(graph, pos, ax=ax1,
                           font_size=14,
                           font_weight='bold')
    ax1.set_title("马尔可夫链结构")
    ax1.axis('off')
    
    # 信念
    ax2 = axes[1]
    nodes = sorted(beliefs.keys())
    x = np.arange(len(nodes))
    width = 0.35
    
    probs_0 = [beliefs[node][0] for node in nodes]
    probs_1 = [beliefs[node][1] for node in nodes]
    
    ax2.bar(x - width/2, probs_0, width, label='State 0', color='blue', alpha=0.6)
    ax2.bar(x + width/2, probs_1, width, label='State 1', color='red', alpha=0.6)
    
    ax2.set_xlabel('节点')
    ax2.set_ylabel('概率')
    ax2.set_title('边际概率（信念）')
    ax2.set_xticks(x)
    ax2.set_xticklabels(nodes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n观察：")
    print("1. 信念传播在树结构上收敛到精确解")
    print("2. 相邻节点的信念相互影响")
    print("3. 边势函数偏好相同状态导致相关性")