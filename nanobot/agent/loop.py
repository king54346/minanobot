"""
Agent 循环：核心处理引擎
这是 nanobot 的核心模块，负责接收消息、调用 LLM、执行工具并返回响应
"""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
from typing import Any

from loguru import logger

# 导入消息总线相关
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
# 导入 LLM 提供者
from nanobot.providers.base import LLMProvider
# 导入上下文构建器
from nanobot.agent.context import ContextBuilder
# 导入工具注册表和各种工具
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
# 导入内存和会话管理
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    Agent 循环 - 核心处理引擎

    这是整个 AI Agent 系统的核心，负责：
    1. 从消息总线接收消息
    2. 结合历史记录、记忆和技能构建上下文
    3. 调用大语言模型（LLM）
    4. 执行 LLM 返回的工具调用
    5. 将响应发送回消息总线

    工作流程：
    - 持续监听消息队列
    - 为每个对话维护会话状态
    - 支持工具调用（文件操作、shell 命令、网络搜索等）
    - 自动进行记忆整合，防止上下文过长
    """

    def __init__(
        self,
        bus: MessageBus,                          # 消息总线，用于接收和发送消息
        provider: LLMProvider,                    # LLM 提供者（如 OpenAI、Anthropic 等）
        workspace: Path,                          # 工作空间目录
        model: str | None = None,                 # 模型名称（如 gpt-4）
        max_iterations: int = 20,                 # 单次对话最大迭代次数（防止无限循环）
        temperature: float = 0.7,                 # LLM 温度参数（控制随机性）
        max_tokens: int = 4096,                   # 单次响应最大 token 数
        memory_window: int = 50,                  # 记忆窗口大小（保留多少条历史消息）
        brave_api_key: str | None = None,        # Brave 搜索 API 密钥
        exec_config: "ExecToolConfig | None" = None,  # Shell 执行配置
        cron_service: "CronService | None" = None,    # 定时任务服务
        restrict_to_workspace: bool = False,      # 是否限制文件操作在工作空间内
        session_manager: SessionManager | None = None,  # 会话管理器
        mcp_servers: dict | None = None,          # MCP（模型上下文协议）服务器配置
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService

        # 核心组件
        self.bus = bus                            # 消息总线
        self.provider = provider                  # LLM 提供者
        self.workspace = workspace                # 工作空间路径
        self.model = model or provider.get_default_model()  # 使用的模型

        # 运行参数
        self.max_iterations = max_iterations      # 最大迭代次数
        self.temperature = temperature            # 温度参数
        self.max_tokens = max_tokens              # 最大 token 数
        self.memory_window = memory_window        # 记忆窗口

        # 外部服务配置
        self.brave_api_key = brave_api_key        # 搜索 API 密钥
        self.exec_config = exec_config or ExecToolConfig()  # Shell 执行配置
        self.cron_service = cron_service          # 定时任务服务
        self.restrict_to_workspace = restrict_to_workspace  # 文件访问限制

        # 初始化各个管理器
        self.context = ContextBuilder(workspace)  # 上下文构建器
        self.sessions = session_manager or SessionManager(workspace)  # 会话管理器
        self.tools = ToolRegistry()               # 工具注册表
        self.subagents = SubagentManager(         # 子代理管理器
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        # 运行状态
        self._running = False                     # 是否正在运行
        self._mcp_servers = mcp_servers or {}     # MCP 服务器配置
        self._mcp_stack: AsyncExitStack | None = None  # MCP 连接栈
        self._mcp_connected = False               # MCP 是否已连接

        # 注册默认工具集
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """
        注册默认工具集

        根据配置注册以下工具：
        - 文件工具：读取、写入、编辑、列出目录
        - Shell 工具：执行命令
        - 网络工具：搜索、抓取网页
        - 消息工具：发送消息
        - 派生工具：创建子代理
        - 定时任务工具：安排定时任务
        """
        # 文件工具（如果配置了限制，则只能访问工作空间）
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))    # 读文件
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))   # 写文件
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))    # 编辑文件
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))     # 列出目录

        # Shell 工具（执行命令）
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # 网络工具
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))  # 网络搜索
        self.tools.register(WebFetchTool())                              # 抓取网页

        # 消息工具（用于主动发送消息）
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # 派生工具（用于创建子代理）
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # 定时任务工具（用于安排定时任务）
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def _connect_mcp(self) -> None:
        """
        连接到配置的 MCP 服务器（一次性、延迟加载）

        MCP（Model Context Protocol）是一个标准协议，用于扩展 AI Agent 的能力。
        这个方法会在首次需要时连接所有配置的 MCP 服务器。
        """
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """
        为所有需要路由信息的工具更新上下文

        某些工具（如消息工具、派生工具、定时任务工具）需要知道当前的渠道和对话 ID，
        以便将消息正确地路由回原始来源。

        Args:
            channel: 渠道名称（如 "telegram", "discord", "cli" 等）
            chat_id: 对话 ID
        """
        # 更新消息工具的上下文
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        # 更新派生工具的上下文
        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        # 更新定时任务工具的上下文
        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        运行 Agent 迭代循环（核心处理逻辑）

        这是 AI Agent 的核心处理流程：
        1. 调用 LLM，获取响应
        2. 如果 LLM 返回工具调用，执行这些工具
        3. 将工具执行结果反馈给 LLM
        4. 重复上述过程，直到 LLM 给出最终回复或达到最大迭代次数

        Args:
            initial_messages: LLM 对话的初始消息列表（包含系统提示、历史记录和当前消息）

        Returns:
            一个元组：(最终回复内容, 使用的工具名称列表)
        """
        messages = initial_messages
        iteration = 0                              # 当前迭代次数
        final_content = None                       # 最终回复内容
        tools_used: list[str] = []                # 使用过的工具列表

        while iteration < self.max_iterations:
            iteration += 1

            # 调用 LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),    # 传入可用工具的定义
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # 如果 LLM 返回了工具调用
            if response.has_tool_calls:
                # 将工具调用转换为标准格式
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                # 将助手的回复（包含工具调用）添加到消息历史
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # 执行每个工具调用
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

                    # 执行工具并获取结果
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)

                    # 将工具执行结果添加到消息历史
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # 添加提示，让 LLM 反思结果并决定下一步
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                # LLM 没有调用工具，说明已经给出最终回复
                final_content = response.content
                break

        return final_content, tools_used

    async def run(self) -> None:
        """
        运行 Agent 循环，处理来自消息总线的消息

        这是主循环，会：
        1. 标记为运行状态
        2. 连接 MCP 服务器（如果配置了）
        3. 持续从消息队列中获取消息
        4. 处理每条消息并发送响应
        5. 如果处理出错，发送错误消息
        """
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                # 等待接收入站消息（超时 1 秒，避免阻塞）
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    # 处理消息
                    response = await self._process_message(msg)
                    if response:
                        # 发送响应到出站队列
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    # 如果处理失败，发送错误消息
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                # 超时，继续下一次循环
                continue
    
    async def close_mcp(self) -> None:
        """
        关闭 MCP 连接

        清理所有 MCP 服务器的连接。
        捕获并忽略 MCP SDK 清理时可能产生的无害异常。
        """
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK 取消作用域清理时会产生噪音，但无害
            self._mcp_stack = None

    def stop(self) -> None:
        """
        停止 Agent 循环

        设置停止标志，主循环将在下次迭代时退出。
        """
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        处理单条入站消息

        这是消息处理的核心方法，负责：
        1. 路由系统消息
        2. 获取或创建会话
        3. 处理斜杠命令（/new、/help）
        4. 触发记忆整合（如果需要）
        5. 构建上下文并运行 Agent 循环
        6. 保存会话并返回响应

        Args:
            msg: 要处理的入站消息
            session_key: 覆盖会话密钥（由 process_direct 使用）

        Returns:
            响应消息，如果不需要响应则返回 None
        """
        # 系统消息通过 chat_id 路由回原始来源（格式："channel:chat_id"）
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        # 记录日志，显示消息预览
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # 获取或创建会话
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # 处理斜杠命令
        # todo 自定义指令
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # 在清除前捕获消息（避免与后台任务的竞态条件）
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            # 异步整合记忆
            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        # 如果会话消息过多，触发记忆整合
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        # 设置工具上下文（用于消息路由）
        self._set_tool_context(msg.channel, msg.chat_id)

        # 构建初始消息（包含历史记录和当前消息）
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # 运行 Agent 循环
        final_content, tools_used = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # 记录响应日志
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # 将消息添加到会话历史
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        # 返回响应消息
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # 传递元数据（如 Slack 的 thread_ts）
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        处理系统消息（例如，子代理通知）

        系统消息是由 Agent 内部组件生成的特殊消息（如子代理完成任务后的通知）。
        chat_id 字段包含"原始渠道:原始对话ID"，用于将响应路由回正确的目的地。

        Args:
            msg: 系统消息

        Returns:
            路由到原始渠道的响应消息
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # 从 chat_id 解析原始来源（格式："channel:chat_id"）
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # 回退方案
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # 获取原始会话
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # 设置工具上下文并构建消息
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )

        # 运行 Agent 循环
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        # 保存到会话历史
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        # 返回到原始渠道
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """
        将旧消息整合到 MEMORY.md 和 HISTORY.md 文件中

        当会话历史过长时，这个方法会：
        1. 提取旧的对话消息
        2. 使用 LLM 生成对话摘要（history_entry）
        3. 更新长期记忆（memory_update）- 提取用户偏好、项目信息等
        4. 将摘要写入 HISTORY.md，记忆写入 MEMORY.md

        这样可以保持上下文窗口在合理大小，同时保留重要信息供将来检索。

        Args:
            session: 要整合的会话
            archive_all: 如果为 True，清除所有消息并重置会话（用于 /new 命令）
                        如果为 False，只写入文件而不修改会话
        """
        memory = MemoryStore(self.workspace)

        # 确定要处理的消息
        if archive_all:
            # 归档所有消息
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            # 保留最近的一半消息
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            # 检查是否有新消息需要整合
            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            # 提取要整合的消息（不包括最近保留的）
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        # 将消息格式化为可读文本
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        # 构建记忆整合提示词
        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            # 调用 LLM 进行记忆整合
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return

            # 清理 markdown 代码块（如果有）
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            # 解析 JSON 响应
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            # 写入历史摘要
            if entry := result.get("history_entry"):
                memory.append_history(entry)

            # 更新长期记忆
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            # 更新整合位置标记
            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        直接处理消息（用于 CLI 或定时任务）

        这个方法提供了一个同步接口，可以直接发送消息并获取响应，
        而不需要通过消息总线。主要用于：
        - 命令行交互
        - 定时任务触发
        - 测试和调试

        Args:
            content: 消息内容
            session_key: 会话标识符（覆盖 channel:chat_id 用于会话查找）
            channel: 来源渠道（用于工具上下文路由）
            chat_id: 来源对话 ID（用于工具上下文路由）

        Returns:
            Agent 的响应文本
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
