"""
Bishop - Pattern Recognition and Machine Learning
主入口文件

这是整个PRML教学代码的入口点。使用Hydra进行配置管理，
可以灵活运行不同章节的代码。

使用方法:
    uv run python main.py chapter=chapter01
    uv run python main.py chapter=chapter02
    等等...
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 设置matplotlib和numpy的配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings('ignore')


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    主函数：根据配置运行相应章节的代码
    
    这个函数是整个项目的入口点。它会：
    1. 接收Hydra配置
    2. 设置随机种子确保可重复性
    3. 根据配置运行相应章节的代码
    4. 处理输出和可视化
    
    Args:
        cfg: Hydra配置对象，包含所有运行参数
    """
    
    # 设置随机种子，确保结果可重复
    # 这是机器学习实验中的重要实践，让实验结果可以被其他人复现
    np.random.seed(cfg.general.seed)
    
    # 设置matplotlib样式
    if hasattr(plt.style, 'use'):
        try:
            plt.style.use(cfg.visualization.style)
        except:
            # 如果指定的样式不存在，使用默认样式
            pass
    
    # 打印欢迎信息
    print("=" * 80)
    print("Bishop - Pattern Recognition and Machine Learning")
    print("教学代码实现")
    print("=" * 80)
    print(f"\n正在运行: {cfg.chapter.name}")
    print("-" * 80)
    
    # 根据章节动态导入并运行相应的模块
    # 这种设计让我们可以独立开发和测试每一章的代码
    # 直接从命令行参数或配置文件获取章节名
    chapter_name = cfg.get('chapter_override', 'chapter01')  # 默认为chapter01
    
    # 尝试从Hydra的override中获取章节名
    import sys
    for arg in sys.argv:
        if arg.startswith('chapter='):
            chapter_name = arg.split('=')[1]
            break
    
    if chapter_name == "chapter01":
        from prml.chapter01 import run_chapter01
        run_chapter01(cfg)
    elif chapter_name == "chapter02":
        from prml.chapter02 import run_chapter02
        run_chapter02(cfg)
    elif chapter_name == "chapter03":
        from prml.chapter03 import run_chapter03
        run_chapter03(cfg)
    elif chapter_name == "chapter04":
        from prml.chapter04 import run_chapter04
        run_chapter04(cfg)
    elif chapter_name == "chapter05":
        from prml.chapter05 import run_chapter05
        run_chapter05(cfg)
    elif chapter_name == "chapter06":
        from prml.chapter06 import run_chapter06
        run_chapter06(cfg)
    elif chapter_name == "chapter07":
        from prml.chapter07 import run_chapter07
        run_chapter07(cfg)
    elif chapter_name == "chapter08":
        from prml.chapter08 import run_chapter08
        run_chapter08(cfg)
    elif chapter_name == "chapter09":
        from prml.chapter09 import run_chapter09
        run_chapter09(cfg)
    elif chapter_name == "chapter10":
        from prml.chapter10 import run_chapter10
        run_chapter10(cfg)
    elif chapter_name == "chapter11":
        from prml.chapter11 import run_chapter11
        run_chapter11(cfg)
    elif chapter_name == "chapter12":
        from prml.chapter12 import run_chapter12
        run_chapter12(cfg)
    elif chapter_name == "chapter13":
        from prml.chapter13 import run_chapter13
        run_chapter13(cfg)
    elif chapter_name == "chapter14":
        from prml.chapter14 import run_chapter14
        run_chapter14(cfg)
    else:
        print(f"章节 {chapter_name} 尚未实现")
        print("请选择已实现的章节运行")
    
    print("\n" + "=" * 80)
    print("运行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
