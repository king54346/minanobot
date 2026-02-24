####  `/nanobot/bus/` - 消息总线

| 文件 | 作用 |
|------|------|
| `queue.py` | **MessageBus核心**：异步消息队列，解耦渠道和Agent |
| `events.py` | 消息事件定义 (InboundMessage, OutboundMessage) |