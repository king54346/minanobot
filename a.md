###  `/workspace/` - Agent工作空间

| 文件 | 作用 |
|------|------|
| `SOUL.md` | **Agent人格定义**：个性、价值观、沟通风格 |
| `AGENTS.md` | **Agent指令**：行为准则、工具使用指南、记忆系统说明 |
| `TOOLS.md` | **工具文档**：所有可用工具的详细说明 |
| `USER.md` | 用户信息（可选） |
| `HEARTBEAT.md` | 心跳任务列表（每30分钟检查一次） |
| `memory/MEMORY.md` | 长期记忆：用户偏好、关系、重要事实 |
| `memory/HISTORY.md` | 事件日志（仅追加，可搜索） |
| `skills/` | 自定义技能目录（覆盖内置技能） |



### 1. 启动流程

```
1. 用户执行 `nanobot gateway`
2. 加载配置 (~/.nanobot/config.json)
3. 初始化 MessageBus
4. 初始化 LLM Provider (LiteLLM)
5. 创建 AgentLoop
6. 启动 ChannelManager（初始化所有启用的聊天渠道）
7. 启动 CronService（定时任务）
8. 启动 HeartbeatService（心跳检查）
9. 进入主循环，等待消息
```


### 2. 消息处理流程

```
1. 用户在Telegram发送消息 "帮我创建一个待办清单"
   ↓
2. TelegramChannel 接收消息 → 创建 InboundMessage
   ↓
3. MessageBus.publish_inbound(msg)
   ↓
4. AgentLoop.consume_inbound() 获取消息
   ↓
5. ContextBuilder 构建上下文：
   - 加载 SOUL.md (人格)
   - 加载 AGENTS.md (指令)
   - 加载 TOOLS.md (工具文档)
   - 加载技能 (SKILL.md)
   - 加载记忆 (MEMORY.md)
   - 加载对话历史
   ↓
6. 调用 LLM (如 OpenRouter/GPT-4)
   ↓
7. LLM 决定使用 write_file 工具创建 todo.md
   ↓
8. ToolRegistry 执行 WriteFileTool
   ↓
9. 工具返回结果 → AgentLoop 继续循环
   ↓
10. LLM 生成最终回复 "已创建待办清单文件"
   ↓
11. AgentLoop 创建 OutboundMessage
   ↓
12. MessageBus.publish_outbound(msg)
   ↓
13. TelegramChannel 接收消息并发送给用户
```





