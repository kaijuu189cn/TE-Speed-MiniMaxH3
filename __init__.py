"""TE-Speed-MiniMaxH3 Linux 兼容替代版

注册与 Windows 原版完全相同的节点 ID `TESpeedMiniMaxH3`，
工作流无需改线即可使用。

实现原理：Block-level residual caching
- 在 DiT Block 级别缓存残差状态，相邻步之间 block 输出变化较小时跳过计算
- 支持 Standard / 4-step LoRA / 8-step LoRA 三种模式
- 通过 ComfyUI patches_replace["dit"] 拦截 DiT block 调用
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "3.5.0-linux"