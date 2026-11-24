"""准备模块路径并提供最终第三、四问策略；本文件不自动连接模拟器。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in reversed(("问题一", "问题二", "问题三", "问题四", "公共代码")):
    sys.path.insert(0, str(ROOT / folder))

from current_policy import configuration, make_robot

__all__ = ["configuration", "make_robot"]
