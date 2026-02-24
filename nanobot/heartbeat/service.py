"""
心跳服务：定期唤醒 Agent 检查待办任务

实现了一个"心跳"机制，类似于闹钟定时响铃：
- 每隔固定时间（默认 30 分钟）唤醒一次 Agent
- Agent 读取工作空间中的 HEARTBEAT.md 文件
- 如果文件中有待办任务，Agent 会自动执行
- 如果没有任务，Agent 回复 HEARTBEAT_OK 后继续休眠

这使得 Agent 能够在无人干预的情况下，自动处理预设的周期性任务。
"""

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

# 默认心跳间隔：30 分钟（单位：秒）
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60

# 心跳时发送给 Agent 的提示词
# 直接将 HEARTBEAT.md 内容嵌入 prompt，让 LLM 立即看到任务，无需额外调用 read_file 工具
HEARTBEAT_PROMPT = """Below is the current content of HEARTBEAT.md in your workspace.
Execute any tasks listed under "Active Tasks". Use the appropriate tools (read_file, exec, write_file, etc.) to carry them out.
After completing all tasks, briefly summarize what you did.
If there are truly no actionable tasks, reply with just: HEARTBEAT_OK

--- HEARTBEAT.md ---
{content}
--- END ---"""

# "无事可做"的标记词，Agent 回复包含此关键词表示没有需要处理的任务
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"


def _is_heartbeat_empty(content: str | None) -> bool:
    """
    检查 HEARTBEAT.md 是否没有可执行的内容

    以下内容会被视为"无内容"而跳过：
    - 空行
    - 标题行（# 开头）
    - HTML 注释（<!-- 开头）
    - Markdown 复选框（- [ ] 或 - [x]）

    只要有一行不属于以上类型，就认为有可执行内容。

    Args:
        content: HEARTBEAT.md 文件内容

    Returns:
        True 表示没有可执行内容，False 表示有任务需要处理
    """
    if not content:
        return True
    
    # 需要跳过的特殊行模式（Markdown 复选框）
    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}
    
    for line in content.split("\n"):
        line = line.strip()
        # 跳过：空行、标题、HTML注释、复选框
        if not line or line.startswith("#") or line.startswith("<!--") or line in skip_patterns:
            continue
        return False  # 发现了有实际内容的行，说明有任务

    return True  # 所有行都是可跳过的，没有任务


class HeartbeatService:
    """
    心跳服务 - 定期唤醒 Agent 检查待办任务

    工作原理类似"定时闹钟"：
    1. 每隔固定时间（默认 30 分钟）触发一次心跳
    2. 先检查 HEARTBEAT.md 文件是否有内容
    3. 如果文件为空或不存在，直接跳过（节省 LLM 调用）
    4. 如果有内容，唤醒 Agent，让它读取并执行文件中的任务
    5. Agent 完成后回复结果，或回复 HEARTBEAT_OK 表示无需操作

    与定时任务（CronService）的区别：
    - CronService：执行预定义的具体任务，有明确的消息内容
    - HeartbeatService：通用的"巡检"机制，任务内容由 HEARTBEAT.md 文件动态决定
    """
    
    def __init__(
        self,
        workspace: Path,
        on_heartbeat: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
    ):
        """
        初始化心跳服务

        Args:
            workspace: 工作空间路径（HEARTBEAT.md 所在目录）
            on_heartbeat: 心跳回调函数，接收提示词字符串，返回 Agent 的响应
                         实际上就是调用 agent.process_direct()
            interval_s: 心跳间隔（秒），默认 30 分钟
            enabled: 是否启用心跳服务
        """
        self.workspace = workspace                # 工作空间路径
        self.on_heartbeat = on_heartbeat          # 心跳回调（调用 Agent）
        self.interval_s = interval_s              # 心跳间隔（秒）
        self.enabled = enabled                    # 是否启用
        self._running = False                     # 运行状态标志
        self._task: asyncio.Task | None = None    # 心跳循环的异步任务

    @property
    def heartbeat_file(self) -> Path:
        """HEARTBEAT.md 文件的完整路径"""
        return self.workspace / "HEARTBEAT.md"
    
    def _read_heartbeat_file(self) -> str | None:
        """
        读取 HEARTBEAT.md 文件内容

        Returns:
            文件内容字符串，文件不存在或读取失败则返回 None
        """
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None
    
    async def start(self) -> None:
        """
        启动心跳服务

        创建一个后台异步任务来运行心跳循环。
        如果服务被禁用，直接返回不启动。
        """
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Heartbeat started (every {self.interval_s}s)")
    
    def stop(self) -> None:
        """
        停止心跳服务

        标记为停止状态，并取消正在运行的心跳循环任务。
        """
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
    
    async def _run_loop(self) -> None:
        """
        心跳主循环

        每隔 interval_s 秒执行一次心跳检查（_tick）。
        注意：先等待一个间隔再执行，所以启动后不会立即触发。

        异常处理：
        - CancelledError：正常停止，退出循环
        - 其他异常：记录错误日志，继续循环（不会因单次失败而停止服务）
        """
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)   # 等待一个心跳间隔
                if self._running:
                    await self._tick()                  # 执行心跳检查
            except asyncio.CancelledError:
                break                                   # 被取消时正常退出
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")   # 记录错误，继续运行

    async def _tick(self) -> None:
        """
        执行一次心跳检查

        流程：
        1. 读取 HEARTBEAT.md 文件
        2. 如果文件为空或无可执行内容 → 跳过（不调用 LLM，节省开销）
        3. 如果有内容 → 调用 Agent（通过 on_heartbeat 回调）
        4. 检查 Agent 响应：
           - 包含 HEARTBEAT_OK → 记录"无需操作"
           - 其他内容 → 记录"已完成任务"
        """
        # 读取心跳文件
        content = self._read_heartbeat_file()
        
        # 如果文件为空或不存在，跳过本次心跳（避免不必要的 LLM 调用）
        if _is_heartbeat_empty(content):
            logger.debug("Heartbeat: no tasks (HEARTBEAT.md empty)")
            return
        
        logger.info("Heartbeat: checking for tasks...")
        
        if self.on_heartbeat:
            try:
                # 将文件内容嵌入 prompt，让 Agent 直接看到任务
                prompt = HEARTBEAT_PROMPT.format(content=content)
                # 调用 Agent 处理心跳（会访问 LLM）
                response = await self.on_heartbeat(prompt)

                # 判断 Agent 是否回复"无事可做"
                # 用 replace("_", "") 做模糊匹配，兼容 HEARTBEAT_OK / HEARTBEAT OK 等格式
                if HEARTBEAT_OK_TOKEN.replace("_", "") in response.upper().replace("_", ""):
                    logger.info("Heartbeat: OK (no action needed)")
                else:
                    logger.info(f"Heartbeat: completed task")
                    
            except Exception as e:
                logger.error(f"Heartbeat execution failed: {e}")
    
    async def trigger_now(self) -> str | None:
        """
        手动立即触发一次心跳

        不等待定时器，直接调用 Agent 处理 HEARTBEAT.md。
        用于调试或用户主动触发。

        Returns:
            Agent 的响应文本，如果没有回调则返回 None
        """
        if self.on_heartbeat:
            content = self._read_heartbeat_file() or "(file not found or empty)"
            prompt = HEARTBEAT_PROMPT.format(content=content)
            return await self.on_heartbeat(prompt)
        return None
