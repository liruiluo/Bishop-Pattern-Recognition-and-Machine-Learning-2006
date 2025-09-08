#!/usr/bin/env python3
"""
PRML (Pattern Recognition and Machine Learning) 完整演示
=========================================================

运行Bishop教材的所有章节实现。

本项目实现了PRML教材的核心算法，包括：
- 第8章：图模型
- 第9章：混合模型与EM
- 第10章：近似推理
- 第11章：采样方法
- 第12章：连续潜变量
- 第13章：序列数据
- 第14章：组合模型

每章都包含：
1. 理论推导的代码实现
2. 详细的中文注释
3. 可视化演示
4. 实际应用案例

使用方法：
python run_all_chapters.py                    # 运行所有章节
python run_all_chapters.py --chapter 10       # 运行特定章节
python run_all_chapters.py --list            # 列出所有章节

依赖：
- numpy, scipy, matplotlib
- scikit-learn
- hydra-core, omegaconf
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入各章节
from prml.chapter08 import run_chapter8
from prml.chapter09 import run_chapter9
from prml.chapter10 import run_chapter10
from prml.chapter11 import run_chapter11
from prml.chapter12 import run_chapter12
from prml.chapter13 import run_chapter13
from prml.chapter14 import run_chapter14


# 章节信息
CHAPTERS = {
    8: {
        'title': '图模型 (Graphical Models)',
        'topics': ['贝叶斯网络', '马尔可夫随机场', '因子图', '和积算法', '最大和算法'],
        'runner': run_chapter8
    },
    9: {
        'title': '混合模型与EM (Mixture Models and EM)',
        'topics': ['K-means', '高斯混合模型', 'EM算法', '贝叶斯GMM'],
        'runner': run_chapter9
    },
    10: {
        'title': '近似推理 (Approximate Inference)',
        'topics': ['变分推理', '期望传播', 'ELBO优化', '平均场近似'],
        'runner': run_chapter10
    },
    11: {
        'title': '采样方法 (Sampling Methods)',
        'topics': ['拒绝采样', '重要性采样', 'MCMC', 'Gibbs采样', 'HMC'],
        'runner': run_chapter11
    },
    12: {
        'title': '连续潜变量 (Continuous Latent Variables)',
        'topics': ['PCA', '概率PCA', '核PCA', 'ICA', '因子分析'],
        'runner': run_chapter12
    },
    13: {
        'title': '序列数据 (Sequential Data)',
        'topics': ['HMM', '卡尔曼滤波', '粒子滤波', 'Viterbi', 'Baum-Welch'],
        'runner': run_chapter13
    },
    14: {
        'title': '组合模型 (Combining Models)',
        'topics': ['Bagging', 'Boosting', '随机森林', '混合专家', 'Stacking'],
        'runner': run_chapter14
    }
}


def print_header():
    """打印项目头部信息"""
    print("\n" + "="*80)
    print(" "*20 + "PRML - Pattern Recognition and Machine Learning")
    print(" "*25 + "Bishop (2006) 算法实现")
    print("="*80)
    print("\n作者：Christopher M. Bishop")
    print("实现：Python 3.8+ with NumPy, SciPy, Scikit-learn")
    print("特点：超细粒度中文注释，教科书式代码讲解")
    print("="*80 + "\n")


def list_chapters():
    """列出所有可用章节"""
    print("\n可用章节：")
    print("-" * 60)
    for chapter_num, info in sorted(CHAPTERS.items()):
        print(f"\n第{chapter_num}章：{info['title']}")
        print("  主要内容：")
        for topic in info['topics']:
            print(f"    • {topic}")
    print("\n" + "-" * 60)
    print(f"共{len(CHAPTERS)}章实现完成")


def run_chapter(chapter_num: int, cfg: DictConfig):
    """运行指定章节"""
    if chapter_num not in CHAPTERS:
        print(f"错误：第{chapter_num}章未实现")
        print(f"可用章节：{sorted(CHAPTERS.keys())}")
        return False
    
    chapter_info = CHAPTERS[chapter_num]
    
    print("\n" + "="*80)
    print(f"第{chapter_num}章：{chapter_info['title']}")
    print("="*80)
    print("主要内容：" + ", ".join(chapter_info['topics']))
    print("="*80)
    
    try:
        # 运行章节
        chapter_info['runner'](cfg)
        print(f"\n第{chapter_num}章运行完成！")
        return True
    except Exception as e:
        print(f"\n运行第{chapter_num}章时出错：{e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_chapters(cfg: DictConfig):
    """运行所有章节"""
    print_header()
    
    successful = []
    failed = []
    
    for chapter_num in sorted(CHAPTERS.keys()):
        print(f"\n{'='*80}")
        print(f"开始运行第{chapter_num}章...")
        print('='*80)
        
        if run_chapter(chapter_num, cfg):
            successful.append(chapter_num)
        else:
            failed.append(chapter_num)
        
        # 章节之间暂停
        if chapter_num < max(CHAPTERS.keys()):
            input(f"\n按Enter继续下一章...")
    
    # 总结
    print("\n" + "="*80)
    print("运行总结")
    print("="*80)
    print(f"成功运行：{len(successful)}章 - {successful}")
    if failed:
        print(f"运行失败：{len(failed)}章 - {failed}")
    print("="*80)


def create_default_config():
    """创建默认配置"""
    cfg = OmegaConf.create({
        'visualization': {
            'show_plots': True,
            'save_plots': False,
            'plot_dir': 'plots'
        },
        'random_state': 42,
        'verbose': True
    })
    return cfg


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PRML算法实现演示',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--chapter', '-c',
        type=int,
        help='运行指定章节 (8-14)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用章节'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='不显示图形'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式（减少迭代次数）'
    )
    
    args = parser.parse_args()
    
    # 列出章节
    if args.list:
        print_header()
        list_chapters()
        return
    
    # 创建配置
    cfg = create_default_config()
    
    if args.no_plots:
        cfg.visualization.show_plots = False
    
    if args.quick:
        # 快速模式：减少采样和迭代
        cfg.n_samples = 100
        cfg.max_iter = 10
        cfg.n_iterations = 100
    
    # 运行指定章节或所有章节
    if args.chapter:
        print_header()
        run_chapter(args.chapter, cfg)
    else:
        run_all_chapters(cfg)


if __name__ == '__main__':
    main()