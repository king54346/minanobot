####`/nanobot/channels/` - 聊天渠道

| 文件 | 作用 |
|------|------|
| `manager.py` | **渠道管理器**：初始化和协调所有聊天渠道 |
| `base.py` | 渠道基类 |
| `telegram.py` | Telegram Bot 集成 |
| `whatsapp.py` | WhatsApp 集成（通过 bridge） |
| `discord.py` | Discord Bot 集成 |
| `feishu.py` | 飞书 Bot 集成 |
| `dingtalk.py` | 钉钉 Bot 集成 |
| `slack.py` | Slack Bot 集成 |
| `qq.py` | QQ Bot 集成 |
| `email.py` | 邮件渠道 (IMAP/SMTP) |
| `mochat.py` | Mochat 集成 |