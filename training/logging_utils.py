# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
通用日志工具模块，支持 TensorBoard 和 Wandb 离线模式。

特点：
- 支持 GPU 集群离线训练（使用 Wandb offline 模式）
- 支持 CPU 集群联网后同步到 Wandb 服务器
- 与 Accelerate 无缝集成
- 向后兼容现有 TensorBoard 日志

使用方式：
    1. GPU 集群训练时设置 use_wandb=True, wandb_offline=True
    2. 训练完成后在 CPU 集群运行 sync_wandb.sh 脚本同步日志
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Wandb 可选导入
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


def setup_wandb_offline_mode():
    """
    设置 Wandb 离线模式。

    在无法联网的 GPU 集群上，Wandb 会将日志保存到本地目录。
    训练完成后可以在能联网的 CPU 集群上同步到 Wandb 服务器。
    """
    os.environ["WANDB_MODE"] = "offline"
    # 禁用 Wandb 的网络请求超时警告
    os.environ["WANDB_SILENT"] = "true"


def setup_wandb_online_mode():
    """设置 Wandb 在线模式（默认）。"""
    if "WANDB_MODE" in os.environ:
        del os.environ["WANDB_MODE"]


def get_wandb_run_dir(logging_dir: str, run_name: str) -> Path:
    """
    获取 Wandb 离线日志存储目录。

    Args:
        logging_dir: 日志根目录
        run_name: 运行名称

    Returns:
        Wandb 离线日志目录路径
    """
    return Path(logging_dir) / "wandb" / run_name


def init_logging(
    accelerator,
    args,
    use_wandb: bool = True,
    wandb_offline: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    初始化日志系统，支持 TensorBoard + Wandb。

    Args:
        accelerator: Accelerate 加速器实例
        args: 训练参数（argparse Namespace 或 dict）
        use_wandb: 是否启用 Wandb
        wandb_offline: 是否使用 Wandb 离线模式（GPU 集群无法联网时使用）
        wandb_project: Wandb 项目名称
        wandb_entity: Wandb 团队/用户名称
        extra_config: 额外的配置信息

    Returns:
        包含日志状态信息的字典
    """
    result = {
        "tensorboard_enabled": True,
        "wandb_enabled": False,
        "wandb_run": None,
        "wandb_dir": None,
    }

    # 转换 args 为 dict
    if hasattr(args, "__dict__"):
        config_dict = vars(args).copy()
    else:
        config_dict = dict(args)

    # 合并额外配置
    if extra_config:
        config_dict.update(extra_config)

    # 只在主进程初始化日志
    if not accelerator.is_main_process:
        return result

    run_name = config_dict.get("run_name", "train")
    logging_dir = config_dict.get("logging_dir", "./logs")

    # Wandb 初始化
    if use_wandb and HAS_WANDB:
        if wandb_offline:
            setup_wandb_offline_mode()

        # 设置 Wandb 目录
        wandb_dir = get_wandb_run_dir(logging_dir, run_name)
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Wandb
        try:
            wandb_run = wandb.init(
                project=wandb_project or "layert2v",
                entity=wandb_entity,
                name=run_name,
                config=config_dict,
                dir=str(wandb_dir),
                resume="allow",
                # 离线模式下自动设置
                mode="offline" if wandb_offline else "online",
            )
            result["wandb_enabled"] = True
            result["wandb_run"] = wandb_run
            result["wandb_dir"] = wandb_dir

            if wandb_offline:
                print(f"[Wandb] 离线模式已启用，日志保存到: {wandb_dir}")
                print(f"[Wandb] 训练完成后运行 'wandb sync {wandb_dir}' 同步到服务器")
        except Exception as e:
            print(f"[Wandb] 初始化失败: {e}")
            result["wandb_enabled"] = False
    elif use_wandb and not HAS_WANDB:
        print("[Wandb] 未安装 wandb，请运行: pip install wandb")

    return result


def log_metrics(
    accelerator,
    metrics: Dict[str, Union[float, int]],
    step: int,
    wandb_enabled: bool = False,
    prefix: Optional[str] = None,
):
    """
    记录训练指标到 TensorBoard 和 Wandb。

    Args:
        accelerator: Accelerate 加速器实例
        metrics: 指标字典
        step: 当前训练步数
        wandb_enabled: 是否启用 Wandb
        prefix: 指标前缀（例如 "train/" 或 "val/"）
    """
    # 添加前缀
    if prefix:
        metrics = {f"{prefix}{k}" if not k.startswith(prefix) else k: v for k, v in metrics.items()}

    # TensorBoard 日志（通过 Accelerate）
    accelerator.log(metrics, step=step)

    # Wandb 日志
    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        # Wandb 的 step 参数
        wandb.log(metrics, step=step)


def log_images(
    accelerator,
    images: Dict[str, Any],
    step: int,
    wandb_enabled: bool = False,
):
    """
    记录图像到 Wandb（TensorBoard 通过 Accelerate 自动处理）。

    Args:
        accelerator: Accelerate 加速器实例
        images: 图像字典，值可以是 numpy 数组或 torch 张量
        step: 当前训练步数
        wandb_enabled: 是否启用 Wandb
    """
    if not accelerator.is_main_process:
        return

    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        wandb_images = {}
        for name, img in images.items():
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            wandb_images[name] = wandb.Image(img)
        wandb.log(wandb_images, step=step)


def log_video(
    accelerator,
    videos: Dict[str, Any],
    step: int,
    fps: int = 8,
    wandb_enabled: bool = False,
):
    """
    记录视频到 Wandb。

    Args:
        accelerator: Accelerate 加速器实例
        videos: 视频字典，值为 [T, C, H, W] 或 [T, H, W, C] 格式的张量
        step: 当前训练步数
        fps: 视频帧率
        wandb_enabled: 是否启用 Wandb
    """
    if not accelerator.is_main_process:
        return

    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        wandb_videos = {}
        for name, video in videos.items():
            if hasattr(video, "cpu"):
                video = video.cpu().numpy()
            # 确保格式为 [T, H, W, C]
            if video.shape[1] in [1, 3]:  # [T, C, H, W]
                video = video.transpose(0, 2, 3, 1)
            wandb_videos[name] = wandb.Video(video, fps=fps)
        wandb.log(wandb_videos, step=step)


def log_model_graph(
    accelerator,
    model,
    input_example,
    wandb_enabled: bool = False,
):
    """
    记录模型计算图。

    Args:
        accelerator: Accelerate 加速器实例
        model: 模型实例
        input_example: 示例输入
        wandb_enabled: 是否启用 Wandb
    """
    if not accelerator.is_main_process:
        return

    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        try:
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"[Wandb] 模型图记录失败: {e}")


def finish_logging(
    accelerator,
    wandb_enabled: bool = False,
):
    """
    结束日志记录。

    Args:
        accelerator: Accelerate 加速器实例
        wandb_enabled: 是否启用 Wandb
    """
    # Accelerate 的 end_training 会处理 TensorBoard
    accelerator.end_training()

    # Wandb 结束
    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        wandb.finish()


def get_trackers_list(use_wandb: bool = True, wandb_offline: bool = True) -> List[str]:
    """
    获取要使用的日志记录器列表。

    注意：当 wandb_offline=True 时，我们不将 "wandb" 添加到 Accelerate 的 log_with，
    因为我们手动管理 Wandb 以支持离线模式。

    Args:
        use_wandb: 是否启用 Wandb
        wandb_offline: 是否使用离线模式

    Returns:
        日志记录器列表
    """
    trackers = ["tensorboard"]

    # 只有在线模式下才将 wandb 添加到 Accelerate
    # 离线模式下我们手动管理 Wandb
    if use_wandb and HAS_WANDB and not wandb_offline:
        trackers.append("wandb")

    return trackers


def save_wandb_info(logging_dir: str, run_name: str, run_id: Optional[str] = None):
    """
    保存 Wandb 运行信息，用于后续同步。

    Args:
        logging_dir: 日志目录
        run_name: 运行名称
        run_id: Wandb 运行 ID
    """
    info_file = Path(logging_dir) / "wandb" / run_name / "wandb_info.txt"
    info_file.parent.mkdir(parents=True, exist_ok=True)

    with open(info_file, "w") as f:
        f.write(f"run_name: {run_name}\n")
        f.write(f"run_id: {run_id or 'N/A'}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"sync_command: wandb sync {info_file.parent}\n")


class WandbOfflineLogger:
    """
    Wandb 离线日志记录器封装类。

    提供与 Accelerate 兼容的接口，支持离线模式。
    """

    def __init__(
        self,
        project: str = "layert2v",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        offline: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.dir = dir
        self.offline = offline
        self.run = None
        self._enabled = False

    def init(self) -> bool:
        """初始化 Wandb。返回是否成功。"""
        if not HAS_WANDB:
            print("[Wandb] 未安装，跳过初始化")
            return False

        if self.offline:
            setup_wandb_offline_mode()

        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                dir=self.dir,
                resume="allow",
                mode="offline" if self.offline else "online",
            )
            self._enabled = True

            if self.offline:
                print(f"[Wandb] 离线模式启用，日志目录: {self.dir}")

            return True
        except Exception as e:
            print(f"[Wandb] 初始化失败: {e}")
            return False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标。"""
        if self._enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def finish(self):
        """结束日志记录。"""
        if self._enabled and self.run is not None:
            wandb.finish()
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled
