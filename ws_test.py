import asyncio, websockets, json

async def chat():
    async with websockets.connect("ws://localhost:18801") as ws:
        # 收到连接确认
        greeting = json.loads(await ws.recv())
        print(f"Connected: {greeting['session_id']}")

        # 发送消息
        await ws.send(json.dumps({
            "type": "message",
            # "content": "创建一个定时任务在桌面上的test.txt文件里写入当前时间，每分钟执行一次",
            "content": "现在几点",
        }))

        # 接收 Agent 回复
        while True:
            try:
                reply = json.loads(await ws.recv())
                print(f"Reply: {reply['content']}")
            except websockets.exceptions.ConnectionClosed:
                print("连接已关闭")
                break
            except Exception as e:
                print(f"接收消息时出错: {e}")
                break
        print(f"Reply: {reply['content']}")

asyncio.run(chat())
