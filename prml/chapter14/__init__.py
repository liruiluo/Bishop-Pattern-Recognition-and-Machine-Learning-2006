"""
Chapter 14: Combining Models (组合模型)
========================================

本章介绍通过组合多个模型来提升性能的方法。

主要内容：
1. 贝叶斯模型平均 (14.1)
   - 模型后验概率
   - 预测分布

2. 委员会方法 (14.2)
   - Bootstrap聚合(Bagging)
   - 子空间方法
   - 错误相关性

3. Boosting (14.3)
   - AdaBoost算法
   - 指数损失函数
   - Gradient Boosting

4. 决策树 (14.4)
   - 分裂准则
   - 剪枝策略

5. 随机森林 (14.5)
   - 特征随机性
   - OOB估计

6. 混合专家模型 (14.6)
   - 门控网络
   - 局部专家

核心思想：
集成学习通过组合多个弱学习器构建强学习器。

偏差-方差分解：
误差 = 偏差² + 方差 + 噪声

组合策略：
1. 平均：降低方差
2. Boosting：降低偏差
3. Stacking：学习组合

多样性来源：
- 数据扰动（Bagging）
- 特征扰动（随机子空间）
- 参数扰动（不同初始化）
- 算法扰动（异质集成）

应用：
- 分类和回归
- 特征重要性评估
- 异常检测
- 推荐系统
"""

from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple

# 导入各节的实现
from .ensemble_methods import (
    BaggingClassifier,
    VotingClassifier,
    BayesianModelAveraging,
    demonstrate_bagging,
    demonstrate_voting
)

from .boosting import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    demonstrate_adaboost,
    demonstrate_gradient_boosting
)

from .tree_methods import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    demonstrate_decision_tree,
    demonstrate_random_forest
)

from .mixture_experts import (
    MixtureOfExperts,
    HierarchicalMixtureOfExperts,
    demonstrate_mixture_of_experts
)


def run_chapter14(cfg: DictConfig) -> None:
    """
    运行第14章的所有演示代码
    
    Args:
        cfg: Hydra配置对象
    """
    print("\n" + "="*80)
    print("第14章：组合模型 (Combining Models)")
    print("="*80)
    
    # 14.1 贝叶斯模型平均
    print("\n" + "-"*60)
    print("14.1 贝叶斯模型平均")
    print("-"*60)
    demonstrate_bayesian_averaging(
        n_models=cfg.bayesian_averaging.n_models,
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.2 委员会方法
    print("\n" + "-"*60)
    print("14.2 委员会方法 (Bagging & Voting)")
    print("-"*60)
    
    # Bagging演示
    demonstrate_bagging(
        n_estimators=cfg.committee.n_members,
        show_plot=cfg.visualization.show_plots
    )
    
    # 投票演示
    demonstrate_voting(
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.3 Boosting
    print("\n" + "-"*60)
    print("14.3 Boosting算法")
    print("-"*60)
    
    # AdaBoost演示
    demonstrate_adaboost(
        n_estimators=cfg.boosting.adaboost.n_estimators,
        show_plot=cfg.visualization.show_plots
    )
    
    # Gradient Boosting演示
    demonstrate_gradient_boosting(
        n_estimators=cfg.boosting.gradient_boosting.n_estimators,
        learning_rate=cfg.boosting.gradient_boosting.learning_rate,
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.4 决策树
    print("\n" + "-"*60)
    print("14.4 决策树")
    print("-"*60)
    demonstrate_decision_tree(
        max_depth=cfg.decision_tree.max_depth,
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.5 随机森林
    print("\n" + "-"*60)
    print("14.5 随机森林")
    print("-"*60)
    demonstrate_random_forest(
        n_estimators=cfg.random_forest.n_estimators,
        max_features=cfg.random_forest.max_features,
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.6 混合专家模型
    print("\n" + "-"*60)
    print("14.6 混合专家模型")
    print("-"*60)
    demonstrate_mixture_of_experts(
        n_experts=cfg.mixture_of_experts.n_experts,
        show_plot=cfg.visualization.show_plots
    )
    
    # 14.7 堆叠泛化
    print("\n" + "-"*60)
    print("14.7 堆叠泛化 (Stacking)")
    print("-"*60)
    demonstrate_stacking(
        n_folds=cfg.stacking.n_folds,
        show_plot=cfg.visualization.show_plots
    )
    
    # 比较不同组合方法
    compare_ensemble_methods(
        show_plot=cfg.visualization.show_plots
    )
    
    print("\n" + "="*80)
    print("第14章演示完成！")
    print("="*80)
    print("\n关键要点：")
    print("1. 集成学习提升泛化性能")
    print("2. Bagging降低方差")
    print("3. Boosting降低偏差")
    print("4. 多样性是关键")
    print("5. 组合策略很重要")
    print("6. 计算代价需考虑")


def demonstrate_bayesian_averaging(n_models: int = 5, 
                                 show_plot: bool = True) -> None:
    """
    演示贝叶斯模型平均
    
    通过多个模型的后验加权平均进行预测。
    """
    print("\n贝叶斯模型平均演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 200
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y_true = np.sin(X).ravel() + 0.1 * X.ravel()**2
    y = y_true + 0.1 * np.random.randn(n_samples)
    
    # 创建多个不同复杂度的模型
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    
    models = []
    degrees = [1, 2, 3, 5, 7]  # 不同的多项式度数
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = BayesianRidge()
        model.fit(X_poly, y)
        models.append((poly, model))
    
    # 计算模型证据（边际似然）
    log_evidences = []
    for poly, model in models:
        X_poly = poly.transform(X)
        # 使用模型的对数边际似然作为证据
        log_evidence = model.score(X_poly, y)
        log_evidences.append(log_evidence)
    
    # 计算模型后验概率（使用均匀先验）
    log_evidences = np.array(log_evidences)
    # 防止数值溢出的技巧
    max_log_evidence = np.max(log_evidences)
    model_posteriors = np.exp(log_evidences - max_log_evidence)
    model_posteriors /= np.sum(model_posteriors)
    
    print("模型后验概率：")
    for i, (degree, prob) in enumerate(zip(degrees, model_posteriors)):
        print(f"  度数{degree}模型: {prob:.3f}")
    
    # 预测
    X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_test_true = np.sin(X_test).ravel() + 0.1 * X_test.ravel()**2
    
    # 各模型预测
    predictions = []
    for poly, model in models:
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)
        predictions.append(y_pred)
    
    # 贝叶斯模型平均
    y_bma = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, model_posteriors):
        y_bma += weight * pred
    
    # 计算误差
    individual_errors = []
    for pred in predictions:
        error = np.mean((pred - y_test_true)**2)
        individual_errors.append(error)
    
    bma_error = np.mean((y_bma - y_test_true)**2)
    
    print(f"\n预测误差（MSE）：")
    for degree, error in zip(degrees, individual_errors):
        print(f"  度数{degree}模型: {error:.4f}")
    print(f"  BMA: {bma_error:.4f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # 绘制各个模型
        for idx, (degree, (poly, model), pred) in enumerate(zip(degrees, models, predictions)):
            ax = axes[idx]
            ax.scatter(X, y, alpha=0.3, s=10, label='数据')
            ax.plot(X_test, y_test_true, 'b-', linewidth=2, label='真实函数')
            ax.plot(X_test, pred, 'r--', linewidth=2, 
                   label=f'度数{degree} (权重={model_posteriors[idx]:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'多项式度数 {degree}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制BMA结果
        ax = axes[5]
        ax.scatter(X, y, alpha=0.3, s=10, label='数据')
        ax.plot(X_test, y_test_true, 'b-', linewidth=2, label='真实函数')
        ax.plot(X_test, y_bma, 'g-', linewidth=3, label='BMA')
        
        # 绘制置信区间
        y_std = np.zeros_like(y_bma)
        for pred, weight in zip(predictions, model_posteriors):
            y_std += weight * (pred - y_bma)**2
        y_std = np.sqrt(y_std)
        
        ax.fill_between(X_test.ravel(), 
                        y_bma - 2*y_std, 
                        y_bma + 2*y_std,
                        alpha=0.2, color='green', label='±2σ')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('贝叶斯模型平均')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('贝叶斯模型平均', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. BMA自动选择合适的模型复杂度")
    print("2. 避免过拟合和欠拟合")
    print("3. 提供不确定性估计")
    print("4. 性能通常优于单一模型")


def demonstrate_stacking(n_folds: int = 5, show_plot: bool = True) -> None:
    """
    演示堆叠泛化
    
    使用元学习器组合多个基学习器。
    """
    print("\n堆叠泛化演示")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=20, 
                              n_informative=15, n_redundant=5,
                              n_classes=3, random_state=42)
    
    # 分割训练集和测试集
    n_train = 350
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 基学习器
    base_learners = [
        ('决策树', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('朴素贝叶斯', GaussianNB()),
        ('逻辑回归', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    # 第一层：训练基学习器并生成元特征
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储元特征
    meta_features_train = np.zeros((len(X_train), len(base_learners) * 3))  # 3个类别
    meta_features_test = np.zeros((len(X_test), len(base_learners) * 3))
    
    print("第一层：训练基学习器")
    for i, (name, clf) in enumerate(base_learners):
        print(f"  训练 {name}...")
        
        # 对训练集使用交叉验证生成元特征
        meta_feature = np.zeros((len(X_train), 3))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            
            # 训练基学习器
            clf_fold = clf.__class__(**clf.get_params())
            clf_fold.fit(X_fold_train, y_fold_train)
            
            # 预测验证集（概率）
            proba = clf_fold.predict_proba(X_fold_val)
            meta_feature[val_idx] = proba
        
        meta_features_train[:, i*3:(i+1)*3] = meta_feature
        
        # 在完整训练集上训练，预测测试集
        clf.fit(X_train, y_train)
        test_proba = clf.predict_proba(X_test)
        meta_features_test[:, i*3:(i+1)*3] = test_proba
    
    # 第二层：训练元学习器
    print("\n第二层：训练元学习器")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(meta_features_train, y_train)
    
    # 预测
    stacking_pred = meta_learner.predict(meta_features_test)
    
    # 计算各模型性能
    from sklearn.metrics import accuracy_score
    
    print("\n模型性能（测试集准确率）：")
    for name, clf in base_learners:
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: {acc:.3f}")
    
    stacking_acc = accuracy_score(y_test, stacking_pred)
    print(f"  Stacking: {stacking_acc:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 可视化元特征
        ax1 = axes[0, 0]
        im1 = ax1.imshow(meta_features_train[:50], aspect='auto', cmap='viridis')
        ax1.set_xlabel('元特征')
        ax1.set_ylabel('样本')
        ax1.set_title('训练集元特征（前50个样本）')
        plt.colorbar(im1, ax=ax1)
        
        # 基学习器权重
        ax2 = axes[0, 1]
        # 元学习器的系数
        weights = meta_learner.coef_.mean(axis=0)  # 平均各类别的权重
        weights_per_model = []
        for i in range(len(base_learners)):
            weights_per_model.append(np.abs(weights[i*3:(i+1)*3]).mean())
        
        x_pos = np.arange(len(base_learners))
        ax2.bar(x_pos, weights_per_model, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name for name, _ in base_learners], rotation=45)
        ax2.set_ylabel('平均权重')
        ax2.set_title('基学习器重要性')
        ax2.grid(True, alpha=0.3)
        
        # 预测对比
        ax3 = axes[1, 0]
        n_show = 50
        x_axis = np.arange(n_show)
        
        # 绘制各模型预测
        for i, (name, clf) in enumerate(base_learners[:2]):  # 只显示前两个
            pred = clf.predict(X_test[:n_show])
            ax3.scatter(x_axis, pred + i*0.1, alpha=0.5, s=20, label=name)
        
        ax3.scatter(x_axis, stacking_pred[:n_show] - 0.2, 
                   alpha=0.7, s=30, marker='^', label='Stacking')
        ax3.scatter(x_axis, y_test[:n_show] + 0.3, 
                   alpha=0.3, s=40, marker='s', label='真实')
        
        ax3.set_xlabel('样本')
        ax3.set_ylabel('类别')
        ax3.set_title('预测对比（前50个测试样本）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 性能对比
        ax4 = axes[1, 1]
        model_names = [name for name, _ in base_learners] + ['Stacking']
        accuracies = []
        for name, clf in base_learners:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, pred))
        accuracies.append(stacking_acc)
        
        colors = ['blue'] * len(base_learners) + ['red']
        bars = ax4.bar(model_names, accuracies, color=colors, alpha=0.7)
        ax4.set_ylabel('准确率')
        ax4.set_title('模型性能对比')
        ax4.set_ylim([min(accuracies) * 0.9, max(accuracies) * 1.05])
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.suptitle('堆叠泛化 (Stacking)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. Stacking学习最优组合方式")
    print("2. 通常优于简单平均或投票")
    print("3. 避免过拟合需要交叉验证")
    print("4. 可以使用任意元学习器")


def compare_ensemble_methods(show_plot: bool = True) -> None:
    """
    比较不同的集成方法
    """
    print("\n集成方法综合比较")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        BaggingClassifier as SkBagging,
        RandomForestClassifier as SkRF,
        AdaBoostClassifier as SkAdaBoost,
        GradientBoostingClassifier as SkGB
    )
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              n_classes=2, random_state=42)
    
    # 定义模型
    models = {
        '单决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Bagging': SkBagging(n_estimators=50, random_state=42),
        '随机森林': SkRF(n_estimators=50, random_state=42),
        'AdaBoost': SkAdaBoost(n_estimators=50, random_state=42),
        'Gradient Boosting': SkGB(n_estimators=50, random_state=42)
    }
    
    # 评估性能
    results = {}
    print("交叉验证评估（5折）：")
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 性能对比箱线图
        ax1 = axes[0, 0]
        box_data = [results[name]['scores'] for name in models.keys()]
        bp = ax1.boxplot(box_data, labels=models.keys())
        ax1.set_ylabel('准确率')
        ax1.set_title('性能分布')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticklabels(models.keys(), rotation=45, ha='right')
        
        # 偏差-方差权衡
        ax2 = axes[0, 1]
        
        # 模拟偏差和方差（简化演示）
        methods = ['单树', 'Bagging', '随机森林', 'AdaBoost', 'GB']
        bias = [0.3, 0.3, 0.25, 0.15, 0.1]
        variance = [0.4, 0.2, 0.15, 0.3, 0.25]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax2.bar(x - width/2, bias, width, label='偏差', alpha=0.7)
        ax2.bar(x + width/2, variance, width, label='方差', alpha=0.7)
        ax2.set_xlabel('方法')
        ax2.set_ylabel('相对值')
        ax2.set_title('偏差-方差权衡（示意）')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 训练时间对比（模拟）
        ax3 = axes[1, 0]
        
        # 相对训练时间
        train_times = {
            '单决策树': 1.0,
            'Bagging': 10.5,
            '随机森林': 8.2,
            'AdaBoost': 15.3,
            'Gradient Boosting': 25.7
        }
        
        ax3.bar(train_times.keys(), train_times.values(), alpha=0.7)
        ax3.set_ylabel('相对训练时间')
        ax3.set_title('计算成本对比')
        ax3.set_xticklabels(train_times.keys(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 特性对比雷达图
        ax4 = axes[1, 1]
        
        categories = ['准确率', '训练速度', '预测速度', '可解释性', '鲁棒性']
        
        # 各方法的特性评分（1-5）
        scores_radar = {
            'Bagging': [4, 3, 3, 2, 4],
            '随机森林': [5, 3, 3, 2, 5],
            'AdaBoost': [4, 2, 4, 3, 3],
            'Gradient Boosting': [5, 2, 3, 2, 4]
        }
        
        # 雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        
        ax4 = plt.subplot(224, projection='polar')
        
        for method, scores in scores_radar.items():
            scores = scores + [scores[0]]  # 闭合
            ax4.plot(angles, scores, 'o-', linewidth=2, label=method)
            ax4.fill(angles, scores, alpha=0.15)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 5)
        ax4.set_title('方法特性对比', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        plt.suptitle('集成方法综合比较', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n方法选择建议：")
    print("┌─────────────┬──────────────────────────┐")
    print("│ 场景        │ 推荐方法                 │")
    print("├─────────────┼──────────────────────────┤")
    print("│ 降低方差    │ Bagging, 随机森林        │")
    print("│ 降低偏差    │ Boosting                 │")
    print("│ 高维数据    │ 随机森林                 │")
    print("│ 实时预测    │ 随机森林（并行）         │")
    print("│ 最高精度    │ Gradient Boosting        │")
    print("│ 可解释性    │ 单树或小集成             │")
    print("└─────────────┴──────────────────────────┘")
    
    print("\n关键见解：")
    print("1. 没有普适最优的方法")
    print("2. 集成通常优于单模型")
    print("3. 多样性是成功关键")
    print("4. 需权衡精度与效率")
    print("5. 可组合多种策略")