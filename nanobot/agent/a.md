
| 文件 | 作用 |
|------|------|
| `loop.py` | **Agent处理循环**：接收消息 → 构建上下文 → 调用LLM → 执行工具 → 返回响应 |
| `context.py` | **上下文构建器**：从workspace加载系统指令、工具文档、技能、记忆等 |
| `skills.py` | **技能加载器**：加载 SKILL.md 文件，扩展Agent能力 |
| `memory.py` | **记忆管理**：读写长期记忆文件 (MEMORY.md, HISTORY.md) |
| `subagent.py` | **子Agent管理**：支持创建专门化的子Agent处理特定任务 |

#### 1.2 `/nanobot/agent/tools/` - 工具系统

| 文件 | 作用 |
|------|------|
| `base.py` | 工具基类定义 |
| `filesystem.py` | 文件操作工具：读写编辑文件、列出目录 |
| `shell.py` | Shell命令执行工具（带安全限制） |
| `web.py` | Web工具：搜索（Brave API）、抓取网页 |
| `message.py` | 消息发送工具：主动向用户发送消息 |
| `spawn.py` | 后台任务生成工具 |
| `cron.py` | 定时任务管理工具 |
| `mcp.py` | MCP (Model Context Protocol) 集成 |
| `registry.py` | 工具注册表，管理所有可用工具 |