
| 文件 | 作用 |
|------|------|
| `service.py` | **心跳服务**：每30分钟检查 `HEARTBEAT.md`，执行周期性任务 |



HEARTBEAT.md:
 ## Active Tasks 
下面加上你想让 Agent 定期做的事就行了
只有标题 #、注释 <!--、空行、复选框 - [ ] → ❌ 被跳过，不调用 LLM
位置处于项目\workspace\HEARTBEAT.md