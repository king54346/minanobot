"""
nanobot.channels.websocket
~~~~~~~~~~~~~~~~~~~~~~~~~~~

WebSocket 频道实现 — 允许客户端通过 WebSocket 直接与 Agent 实时对话。

设计思路
--------
- 基于 Python 标准库的 ``websockets`` 启动一个异步 WS 服务器。
- 每个 WS 连接视为一个独立的聊天会话（session），通过连接 ID 区分。
- 客户端发送 JSON 消息，Agent 处理后通过同一连接返回 JSON 响应。
- 支持可选的 Token 鉴权（Bearer Token），防止未授权访问。

协议格式
--------
**客户端 → 服务端（发送消息）**::

    {
        "type": "message",               # 消息类型（目前仅支持 "message"）
        "content": "你好，帮我查天气",     # 消息正文
        "session_id": "my-session"        # 可选，自定义会话 ID（不传则自动分配）
    }

**服务端 → 客户端（Agent 回复）**::

    {
        "type": "response",
        "content": "今天北京天气晴，温度 25°C",
        "session_id": "my-session"
    }

**服务端 → 客户端（错误）**::

    {
        "type": "error",
        "content": "Invalid message format",
        "session_id": null
    }

**服务端 → 客户端（连接确认）**::

    {
        "type": "connected",
        "content": "Connected to nanobot",
        "session_id": "ws-abc123"
    }

鉴权
----
如果配置了 ``token``，客户端连接时需要在 URL 查询参数或 HTTP 头中携带 Token::

    ws://localhost:18800?token=your-secret-token

    # 或通过 HTTP 头
    Authorization: Bearer your-secret-token

使用方式
--------
**启动 WS 服务（独立模式）**::

    nanobot ws --port 18800

**在 gateway 中自动启动**::

    # config.json 中启用 websocket channel
    {
        "channels": {
            "websocket": {
                "enabled": true,
                "host": "0.0.0.0",
                "port": 18800,
                "token": "optional-secret"
            }
        }
    }

**Python 客户端示例**::

    import asyncio, websockets, json

    async def chat():
        async with websockets.connect("ws://localhost:18800") as ws:
            # 收到连接确认
            greeting = json.loads(await ws.recv())
            print(f"Connected: {greeting}")

            # 发送消息
            await ws.send(json.dumps({
                "type": "message",
                "content": "Hello nanobot!"
            }))

            # 接收回复
            reply = json.loads(await ws.recv())
            print(f"Reply: {reply['content']}")

    asyncio.run(chat())
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import ServerConnection
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class WebSocketChannel(BaseChannel):
    """
    WebSocket 频道 — 通过 WS 协议直接与 Agent 实时对话。

    继承自 BaseChannel，完全融入 nanobot 的消息总线架构：
    - 客户端消息通过 MessageBus 转发给 AgentLoop 处理
    - Agent 的回复通过 send() 方法推送回对应的 WS 连接

    Attributes
    ----------
    name : str
        频道名称，固定为 ``"websocket"``。用于消息路由和会话标识。
    """

    name: str = "websocket"

    def __init__(self, config: Any, bus: MessageBus):
        """
        初始化 WebSocket 频道。

        Parameters
        ----------
        config : WebSocketConfig
            WebSocket 频道配置（host, port, token, allow_from 等）。
        bus : MessageBus
            消息总线实例，用于 Agent 通信。
        """
        super().__init__(config, bus)

        self._host: str = getattr(config, "host", "0.0.0.0")
        self._port: int = getattr(config, "port", 18800)
        self._token: str = getattr(config, "token", "")
        self._max_message_size: int = getattr(config, "max_message_size", 1_048_576)  # 1MB

        # 活跃连接映射：chat_id → WebSocket 连接对象
        # chat_id 格式为 "ws-{uuid}"，每个连接唯一
        self._connections: dict[str, ServerConnection] = {}

        # WS 服务器实例（start() 时创建）
        self._server: Any = None

    # ------------------------------------------------------------------
    # BaseChannel 接口实现
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        启动 WebSocket 服务器，开始监听客户端连接。

        启动后会持续运行，直到 stop() 被调用。
        每个新连接会在 _handle_connection() 中独立处理。
        """
        self._running = True

        logger.info(f"WebSocket channel starting on ws://{self._host}:{self._port}")

        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=self._max_message_size,
        )

        logger.info(f"WebSocket channel listening on ws://{self._host}:{self._port}")

        # 保持服务运行，直到被取消
        try:
            await asyncio.Future()  # 永远阻塞，直到 task 被 cancel
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """停止 WebSocket 服务器，关闭所有活跃连接。"""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # 关闭所有活跃连接
        for chat_id, ws in list(self._connections.items()):
            try:
                await ws.close(1001, "Server shutting down")
            except Exception:
                pass
        self._connections.clear()

        logger.info("WebSocket channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """
        将 Agent 的回复发送到对应的 WebSocket 客户端。

        Parameters
        ----------
        msg : OutboundMessage
            包含 chat_id（用于定位连接）和 content（回复内容）的消息。
        """
        ws = self._connections.get(msg.chat_id)
        if ws is None:
            logger.warning(f"WebSocket connection not found for chat_id={msg.chat_id}")
            return

        response = {
            "type": "response",
            "content": msg.content,
            "session_id": msg.chat_id,
        }

        try:
            await ws.send(json.dumps(response, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send WS message to {msg.chat_id}: {e}")
            # 连接可能已断开，清理
            self._connections.pop(msg.chat_id, None)

    # ------------------------------------------------------------------
    # 连接处理
    # ------------------------------------------------------------------

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """
        处理单个 WebSocket 连接的完整生命周期。

        流程：
        1. 鉴权检查（如配置了 token）
        2. 为连接分配唯一的 chat_id
        3. 发送连接确认消息
        4. 进入消息接收循环
        5. 连接断开时清理资源

        Parameters
        ----------
        websocket : ServerConnection
            新建立的 WebSocket 连接对象。
        """
        # ---- 鉴权 ----
        if not self._authenticate(websocket):
            await websocket.close(4001, "Unauthorized: invalid or missing token")
            return

        # ---- 分配连接 ID ----
        chat_id = f"ws-{uuid.uuid4().hex[:12]}"
        sender_id = chat_id  # WS 连接没有独立的 sender 概念，用 chat_id 代替
        self._connections[chat_id] = websocket

        logger.info(f"WebSocket client connected: {chat_id} "
                     f"(remote={websocket.remote_address})")

        # ---- 发送连接确认 ----
        try:
            welcome = {
                "type": "connected",
                "content": "Connected to nanobot",
                "session_id": chat_id,
            }
            await websocket.send(json.dumps(welcome, ensure_ascii=False))
        except Exception:
            self._connections.pop(chat_id, None)
            return

        # ---- 消息接收循环 ----
        try:
            async for raw_message in websocket:
                await self._process_ws_message(raw_message, chat_id, sender_id)
        except websockets.exceptions.ConnectionClosed as e:
            code = e.rcvd.code if e.rcvd else "unknown"
            logger.info(f"WebSocket client disconnected: {chat_id} (code={code})")
        except Exception as e:
            logger.error(f"WebSocket error for {chat_id}: {e}")
        finally:
            # ---- 清理 ----
            self._connections.pop(chat_id, None)
            logger.debug(f"WebSocket connection cleaned up: {chat_id}")

    async def _process_ws_message(
        self, raw: str | bytes, chat_id: str, sender_id: str
    ) -> None:
        """
        解析并处理一条来自客户端的 WebSocket 消息。

        Parameters
        ----------
        raw : str | bytes
            客户端发送的原始消息（应为 JSON 字符串）。
        chat_id : str
            连接的唯一标识。
        sender_id : str
            发送者标识（WS 中等同于 chat_id）。
        """
        # ---- 解析 JSON ----
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            await self._send_error(chat_id, f"Invalid JSON: {e}")
            return

        # ---- 提取字段 ----
        msg_type = data.get("type", "message")
        content = data.get("content", "").strip()

        # 客户端可以指定自定义 session_id，实现跨连接的会话保持
        session_id = data.get("session_id") or chat_id

        # ---- ping/pong 心跳 ----
        if msg_type == "ping":
            ws = self._connections.get(chat_id)
            if ws:
                try:
                    await ws.send(json.dumps({"type": "pong"}))
                except Exception:
                    pass
            return

        # ---- 消息内容校验 ----
        if msg_type != "message":
            await self._send_error(chat_id, f"Unknown message type: {msg_type}")
            return

        if not content:
            await self._send_error(chat_id, "Empty message content")
            return

        # ---- 权限检查 ----
        if not self.is_allowed(sender_id):
            await self._send_error(chat_id, "Access denied")
            return

        # ---- 转发到消息总线 ----
        # 通过 BaseChannel._handle_message() 将消息发布到 MessageBus，
        # AgentLoop 会从 bus 中消费并处理，处理完成后通过 send() 回调返回。
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata={"session_id": session_id},
        )

    # ------------------------------------------------------------------
    # 鉴权
    # ------------------------------------------------------------------

    def _authenticate(self, websocket: ServerConnection) -> bool:
        """
        验证 WebSocket 连接的身份。

        支持两种传递 Token 的方式：
        1. URL 查询参数：``ws://host:port?token=xxx``
        2. HTTP 头：``Authorization: Bearer xxx``

        如果配置中未设置 token（空字符串），则允许所有连接。

        Parameters
        ----------
        websocket : ServerConnection
            待验证的 WebSocket 连接。

        Returns
        -------
        bool
            True 表示鉴权通过，False 表示拒绝。
        """
        if not self._token:
            return True  # 未配置 token，允许所有连接

        # 方式 1：URL 查询参数
        try:
            path = websocket.request.path if websocket.request else ""
            parsed = urlparse(path)
            params = parse_qs(parsed.query)
            if params.get("token", [None])[0] == self._token:
                return True
        except Exception:
            pass

        # 方式 2：HTTP Authorization 头
        try:
            headers = websocket.request.headers if websocket.request else {}
            auth = headers.get("Authorization", "")
            if auth.startswith("Bearer ") and auth[7:] == self._token:
                return True
        except Exception:
            pass

        logger.warning(f"WebSocket auth failed from {websocket.remote_address}")
        return False

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    async def _send_error(self, chat_id: str, error_msg: str) -> None:
        """
        向指定连接发送错误消息。

        Parameters
        ----------
        chat_id : str
            目标连接 ID。
        error_msg : str
            错误描述。
        """
        ws = self._connections.get(chat_id)
        if ws is None:
            return

        error = {
            "type": "error",
            "content": error_msg,
            "session_id": chat_id,
        }
        try:
            await ws.send(json.dumps(error, ensure_ascii=False))
        except Exception:
            pass

    @property
    def active_connections(self) -> int:
        """当前活跃的 WebSocket 连接数。"""
        return len(self._connections)



