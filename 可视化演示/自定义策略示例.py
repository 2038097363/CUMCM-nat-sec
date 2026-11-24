"""策略接入示例：先复用内置策略，再按需替换为自己的实现。"""
from 入口 import make_robot as 内置策略


def make_robot(backend, question):
    # 返回的对象需要有 run()；与本地环境交互时只使用 backend.post。
    return 内置策略(backend, question)
