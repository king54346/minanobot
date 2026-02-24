### 3. `/bridge/` - WhatsApp Bridge (TypeScript)

| 文件 | 作用 |
|------|------|
| `src/index.ts` | 入口文件 |
| `src/server.ts` | WebSocket服务器，连接WhatsApp Web和Python后端 |
| `src/whatsapp.ts` | WhatsApp Web客户端（使用 Baileys 库） |
| `src/types.d.ts` | TypeScript类型定义 |

**作用**：由于 WhatsApp Web 需要 Node.js 环境，这个 TypeScript 桥接服务负责：
- 连接 WhatsApp Web
- 接收WhatsApp消息并通过WebSocket转发给Python后端
- 接收Python后端的响应并发送到WhatsApp
