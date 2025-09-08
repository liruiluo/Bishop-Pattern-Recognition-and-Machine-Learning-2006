"""
Chapter 8: Graphical Models (图模型)
====================================

本章介绍概率图模型，这是表示和推理复杂概率分布的强大框架。

主要内容：
1. 贝叶斯网络 (8.1-8.2)
   - 有向无环图
   - 条件独立性
   - d-分离准则

2. 马尔可夫随机场 (8.3)
   - 无向图模型
   - 团和势函数
   - Ising模型

3. 推断算法 (8.4)
   - 变量消除
   - 信念传播
   - 联结树算法

核心概念：
图模型提供了一种直观的方式来表示变量之间的依赖关系，
并利用条件独立性进行高效推断。

应用领域：
- 计算机视觉
- 自然语言处理
- 生物信息学
- 专家系统
- 因果推理
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 导入各节的实现
from .bayesian_networks import (
    BayesianNetwork,
    create_alarm_network,
    demonstrate_bayesian_network,
    demonstrate_three_patterns
)

from .markov_random_fields import (
    MarkovRandomField,
    IsingModel,
    demonstrate_mrf,
    demonstrate_ising_model,
    compare_bn_mrf
)

from .inference import (
    VariableElimination,
    BeliefPropagation,
    demonstrate_variable_elimination,
    demonstrate_belief_propagation
)


def run_chapter08(cfg: DictConfig) -> None:
    """
    运行第8章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第8章：图模型 (Graphical Models)")
    print("="*80)
    
    # 8.1-8.2 贝叶斯网络
    print("\n" + "-"*60)
    print("8.1-8.2 贝叶斯网络")
    print("-"*60)
    
    # 基本演示
    demonstrate_bayesian_network()
    
    # 三种条件独立模式
    demonstrate_three_patterns()
    
    # 医疗诊断示例
    demonstrate_medical_diagnosis()
    
    # 8.3 马尔可夫随机场
    print("\n" + "-"*60)
    print("8.3 马尔可夫随机场")
    print("-"*60)
    
    # MRF演示
    demonstrate_mrf()
    
    # Ising模型
    demonstrate_ising_model()
    
    # 比较BN和MRF
    compare_bn_mrf()
    
    # 8.4 推断算法
    print("\n" + "-"*60)
    print("8.4 推断算法")
    print("-"*60)
    
    # 变量消除
    demonstrate_variable_elimination()
    
    # 信念传播
    demonstrate_belief_propagation()
    
    # 推断复杂度分析
    analyze_inference_complexity()
    
    print("\n" + "="*80)
    print("第8章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 图模型用图结构表示概率分布")
    print("2. 条件独立性是图模型的核心")
    print("3. BN适合因果建模，MRF适合对称依赖")
    print("4. 精确推断通常是困难的")
    print("5. 消息传递是统一的推断框架")
    print("6. 近似推断对大规模问题是必要的")


def demonstrate_medical_diagnosis() -> None:
    """
    演示医疗诊断的贝叶斯网络
    
    一个简化的医疗诊断系统：
    - 疾病影响症状
    - 根据症状推断疾病
    """
    print("\n医疗诊断贝叶斯网络")
    print("=" * 60)
    
    # 创建网络
    bn = BayesianNetwork()
    
    # 添加节点
    # 疾病
    bn.add_node('Flu', [False, True])  # 流感
    bn.add_node('Cold', [False, True])  # 感冒
    
    # 症状
    bn.add_node('Fever', [False, True])  # 发烧
    bn.add_node('Cough', [False, True])  # 咳嗽
    bn.add_node('Headache', [False, True])  # 头痛
    
    # 添加边（疾病→症状）
    bn.add_edge('Flu', 'Fever')
    bn.add_edge('Flu', 'Cough')
    bn.add_edge('Flu', 'Headache')
    bn.add_edge('Cold', 'Cough')
    bn.add_edge('Cold', 'Headache')
    
    # 添加CPD
    # 先验概率
    bn.add_cpd('Flu', np.array([0.95, 0.05]))  # 5%的人有流感
    bn.add_cpd('Cold', np.array([0.8, 0.2]))   # 20%的人感冒
    
    # P(Fever|Flu)
    cpd_fever = np.array([[0.99, 0.1],   # P(Fever=F|Flu=F), P(Fever=F|Flu=T)
                         [0.01, 0.9]])   # P(Fever=T|Flu=F), P(Fever=T|Flu=T)
    bn.add_cpd('Fever', cpd_fever, ['Flu'])
    
    # P(Cough|Flu, Cold)
    cpd_cough = np.array([[0.9, 0.3, 0.2, 0.05],   # P(Cough=F|...)
                         [0.1, 0.7, 0.8, 0.95]])   # P(Cough=T|...)
    bn.add_cpd('Cough', cpd_cough, ['Flu', 'Cold'])
    
    # P(Headache|Flu, Cold)
    cpd_headache = np.array([[0.8, 0.4, 0.3, 0.1],   # P(Headache=F|...)
                            [0.2, 0.6, 0.7, 0.9]])   # P(Headache=T|...)
    bn.add_cpd('Headache', cpd_headache, ['Flu', 'Cold'])
    
    # 可视化网络
    print("\n网络结构：")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 分层布局
    pos = {
        'Flu': (0, 1),
        'Cold': (2, 1),
        'Fever': (-1, 0),
        'Cough': (1, 0),
        'Headache': (3, 0)
    }
    
    # 绘制节点（不同类型用不同颜色）
    disease_nodes = ['Flu', 'Cold']
    symptom_nodes = ['Fever', 'Cough', 'Headache']
    
    nx.draw_networkx_nodes(bn.graph, pos, nodelist=disease_nodes,
                          node_color='lightblue', node_size=2000,
                          edgecolors='black', linewidths=2, label='疾病')
    nx.draw_networkx_nodes(bn.graph, pos, nodelist=symptom_nodes,
                          node_color='lightgreen', node_size=2000,
                          edgecolors='black', linewidths=2, label='症状')
    
    nx.draw_networkx_edges(bn.graph, pos, edge_color='gray',
                          arrows=True, arrowsize=20, width=2)
    nx.draw_networkx_labels(bn.graph, pos, font_size=12, font_weight='bold')
    
    plt.title("医疗诊断贝叶斯网络", fontsize=14)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 推断示例
    print("\n诊断推理：")
    print("-" * 40)
    
    # 场景1：有发烧和咳嗽，推断疾病
    print("\n场景1：患者有发烧和咳嗽")
    
    # 使用变量消除进行推断
    factors = [
        {'vars': ['Flu'], 'table': np.array([0.95, 0.05]), 'states': {'Flu': 2}},
        {'vars': ['Cold'], 'table': np.array([0.8, 0.2]), 'states': {'Cold': 2}},
        {'vars': ['Fever', 'Flu'], 
         'table': cpd_fever, 
         'states': {'Fever': 2, 'Flu': 2}},
        {'vars': ['Cough', 'Flu', 'Cold'], 
         'table': cpd_cough.reshape(2, 2, 2), 
         'states': {'Cough': 2, 'Flu': 2, 'Cold': 2}},
        {'vars': ['Headache', 'Flu', 'Cold'], 
         'table': cpd_headache.reshape(2, 2, 2), 
         'states': {'Headache': 2, 'Flu': 2, 'Cold': 2}}
    ]
    
    ve = VariableElimination(factors)
    
    # P(Flu|Fever=T, Cough=T)
    result = ve.query(['Flu'], evidence={'Fever': 1, 'Cough': 1})
    print(f"P(Flu=T|Fever=T, Cough=T) = {result['table'][1]:.3f}")
    
    # P(Cold|Fever=T, Cough=T)
    result = ve.query(['Cold'], evidence={'Fever': 1, 'Cough': 1})
    print(f"P(Cold=T|Fever=T, Cough=T) = {result['table'][1]:.3f}")
    
    print("\n诊断：流感的可能性很高")
    
    # 场景2：只有咳嗽，没有发烧
    print("\n场景2：患者只有咳嗽，没有发烧")
    
    # P(Flu|Fever=F, Cough=T)
    result = ve.query(['Flu'], evidence={'Fever': 0, 'Cough': 1})
    print(f"P(Flu=T|Fever=F, Cough=T) = {result['table'][1]:.3f}")
    
    # P(Cold|Fever=F, Cough=T)
    result = ve.query(['Cold'], evidence={'Fever': 0, 'Cough': 1})
    print(f"P(Cold=T|Fever=F, Cough=T) = {result['table'][1]:.3f}")
    
    print("\n诊断：更可能是感冒而不是流感")
    
    print("\n观察：")
    print("1. 贝叶斯网络能进行诊断推理")
    print("2. 多个症状提供更准确的诊断")
    print("3. 缺失的症状也提供信息")
    print("4. 可以处理不确定性和噪声")


def analyze_inference_complexity() -> None:
    """
    分析推断算法的复杂度
    """
    print("\n推断复杂度分析")
    print("=" * 60)
    
    # 创建不同结构的图
    structures = {
        '链': nx.path_graph(5),
        '树': nx.balanced_tree(2, 3),
        '环': nx.cycle_graph(6),
        '完全图': nx.complete_graph(4),
        '格子': nx.grid_2d_graph(3, 3)
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, (name, graph) in enumerate(structures.items()):
        ax = axes[idx]
        
        # 计算树宽（近似）
        if name == '链' or name == '树':
            treewidth = 1 if name == '链' else 2
        elif name == '环':
            treewidth = 2
        elif name == '完全图':
            treewidth = len(graph) - 1
        else:  # 格子
            treewidth = min(3, 3)  # min(width, height)
        
        # 可视化
        if name == '格子':
            pos = dict(zip(graph.nodes(), 
                         [(i, j) for i in range(3) for j in range(3)]))
        else:
            pos = nx.spring_layout(graph)
        
        nx.draw(graph, pos, ax=ax, with_labels=True,
               node_color='lightblue', node_size=500,
               edge_color='gray', width=2)
        
        ax.set_title(f"{name}\n树宽≈{treewidth}")
        ax.axis('off')
    
    plt.suptitle("不同图结构的推断复杂度", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n复杂度分析：")
    print("-" * 40)
    print("结构      | 树宽 | 推断复杂度")
    print("-" * 40)
    print("链        |  1   | O(n)")
    print("树        |  1   | O(n)")
    print("环        |  2   | O(n)")
    print("完全图    | n-1  | O(2^n)")
    print("k×k格子   |  k   | O(n·2^k)")
    
    print("\n关键见解：")
    print("1. 树宽决定精确推断的复杂度")
    print("2. 树结构允许线性时间推断")
    print("3. 稠密图的推断是指数级的")
    print("4. 许多实际问题有较小的树宽")
    print("5. 近似推断对高树宽图是必要的")