"""
定时任务服务
"""

import asyncio
import contextlib
import heapq
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int, last_scheduled_ms: int | None = None) -> int | None:
    """
    计算下次执行时间。

    新增 last_scheduled_ms 参数用于 "every" 模式的漂移修正：
    基于上次【计划】执行时间而非实际完成时间来推算，避免累积误差。

    Args:
        schedule: 调度配置
        now_ms: 当前时间（毫秒）
        last_scheduled_ms: 上次计划执行时间，仅用于 every 模式漂移修正
    """
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # ✅ 漂移修正：从上次计划时间推算，而不是从"现在"推算
        # 如果任务执行耗时 2s，使用 now+interval 会导致每次都晚 2s，长期累积
        # 使用 last_scheduled+interval 则保持严格周期
        base = last_scheduled_ms if last_scheduled_ms else now_ms
        next_t = base + schedule.every_ms
        # 如果计算出的时间已过期（服务停机恢复场景），跳到最近的未来周期
        if next_t <= now_ms:
            elapsed = now_ms - base
            cycles_missed = elapsed // schedule.every_ms + 1
            next_t = base + cycles_missed * schedule.every_ms
        return next_t

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(time.time(), tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


# ─── 最小堆节点 ─────────────────────────────────────────────────────────────

class _HeapEntry:
    """
    堆节点，按 next_run_at_ms 排序。
    使用独立类而非 tuple，方便后续扩展且避免 id 比较问题。
    """
    __slots__ = ("next_run_at_ms", "job_id")

    def __init__(self, next_run_at_ms: int, job_id: str):
        self.next_run_at_ms = next_run_at_ms
        self.job_id = job_id

    def __lt__(self, other: "_HeapEntry") -> bool:
        return self.next_run_at_ms < other.next_run_at_ms


class CronService:
    """
    定时任务服务（优化版）

    核心改进：
    ┌──────────────────────────────────────────────────────────┐
    │  组件                原实现           优化实现            │
    │  ──────────────────  ─────────────    ─────────────────  │
    │  查找最早任务        O(n) min()       O(1) 堆顶          │
    │  更新单任务          O(n) 重建        O(log n) heappush  │
    │  任务执行            串行 await       asyncio.gather     │
    │  重置定时器          无条件 cancel    仅在必要时重置      │
    │  every 漂移          累积             基于计划时间修正   │
    │  持久化              同步写文件        线程池 offload     │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job    # 回调任务方法会调用agent
        self._store: CronStore | None = None
        self._timer_task: asyncio.Task | None = None
        self._running = False

        # ✅ 最小堆：元素为 _HeapEntry，堆顶始终是最早到期任务
        # 注意：堆中可能存在"过期条目"（任务已删除/禁用），
        # 处理时通过 _is_valid_heap_entry() 惰性过滤。
        self._heap: list[_HeapEntry] = []

        # 当前定时器的目标唤醒时间，用于智能重置判断
        self._armed_wake_ms: int | None = None

    # ─── 持久化 ─────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _file_lock(self):
        """
        跨进程文件锁（cross-platform）。

        使用 .lock 伴随文件实现互斥，确保守护进程和 CLI 不会并发读写 jobs.json。

        Windows  → msvcrt.locking（字节范围锁）
        Unix     → fcntl.flock（advisory lock）

        锁文件独立于数据文件，这样即使数据文件被原子替换，锁依然有效。
        超时 5s 后放弃，避免死锁（进程崩溃后锁会被 OS 自动释放）。
        """
        lock_path = self.store_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "w")
        try:
            if sys.platform == "win32":
                import msvcrt
                deadline = time.time() + 5
                while True:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.time() > deadline:
                            logger.warning("Cron: file lock timeout, proceeding anyway")
                            break
                        time.sleep(0.05)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

            yield

        finally:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_UN)
            finally:
                f.close()

    @staticmethod
    def _parse_jobs(data: dict) -> list[CronJob]:
        """将 JSON dict 反序列化为 CronJob 列表（供 _load 和合并写使用）。"""
        jobs = []
        for j in data.get("jobs", []):
            jobs.append(CronJob(
                id=j["id"],
                name=j["name"],
                enabled=j.get("enabled", True),
                schedule=CronSchedule(
                    kind=j["schedule"]["kind"],
                    at_ms=j["schedule"].get("atMs"),
                    every_ms=j["schedule"].get("everyMs"),
                    expr=j["schedule"].get("expr"),
                    tz=j["schedule"].get("tz"),
                ),
                payload=CronPayload(
                    kind=j["payload"].get("kind", "agent_turn"),
                    message=j["payload"].get("message", ""),
                    deliver=j["payload"].get("deliver", False),
                    channel=j["payload"].get("channel"),
                    to=j["payload"].get("to"),
                ),
                state=CronJobState(
                    next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                    last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                    last_status=j.get("state", {}).get("lastStatus"),
                    last_error=j.get("state", {}).get("lastError"),
                ),
                created_at_ms=j.get("createdAtMs", 0),
                updated_at_ms=j.get("updatedAtMs", 0),
                delete_after_run=j.get("deleteAfterRun", False),
            ))
        return jobs

    @staticmethod
    def _serialize_jobs(store: CronStore) -> dict:
        """将 CronStore 序列化为 JSON dict。"""
        return {
            "version": store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                }
                for j in store.jobs
            ],
        }

    def _load_store(self) -> CronStore:
        """从磁盘加载任务（首次调用时），之后使用内存缓存。"""
        if self._store:
            return self._store

        with self._file_lock():
            if self.store_path.exists():
                try:
                    data = json.loads(self.store_path.read_text())
                    self._store = CronStore(jobs=self._parse_jobs(data))
                except Exception as e:
                    logger.warning(f"Failed to load cron store: {e}")
                    self._store = CronStore()
            else:
                self._store = CronStore()

        return self._store

    def _write_locked(self, store: CronStore) -> None:
        """
        在文件锁保护下原子写入。

        使用临时文件 + rename 确保写操作原子性：
        其他进程看到的永远是完整文件，不会读到写了一半的内容。
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialize_jobs(store)
        text = json.dumps(data, indent=2)
        # 写临时文件再 rename，保证原子性
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(self.store_path)

    def _save_store(self) -> None:
        """
        同步写入（CLI / 启动时使用）。

        直接覆盖写，适合本进程是唯一写入者的场景（如服务初始化）。
        公共 API 的增删改调用 _save_merge() 以避免覆盖其他进程的修改。
        """
        if not self._store:
            return
        with self._file_lock():
            self._write_locked(self._store)

    def _save_merge(self) -> None:
        """
        合并写入（守护进程执行任务后调用）。

        解决核心竞争问题：
        守护进程在内存中缓存了任务列表。若 CLI 在守护进程执行任务期间
        删除/添加了任务，守护进程直接覆盖写会丢失 CLI 的修改。

        合并策略（在文件锁保护下）：
        1. 重新读取磁盘上的最新任务列表（可能含 CLI 的增删）
        2. 以磁盘状态为权威（决定哪些任务存在）
        3. 用内存中的执行状态（last_run, next_run 等）更新磁盘任务
        4. 写回合并结果

        这样 CLI 删除任务 → 守护进程执行 → 保存，删除操作不会被覆盖。
        """
        if not self._store:
            return

        # 构建内存状态索引：id -> job（仅用于状态合并）
        mem_index = {j.id: j for j in self._store.jobs}

        with self._file_lock():
            # 1. 读取磁盘最新状态（权威来源）
            disk_jobs: list[CronJob] = []
            if self.store_path.exists():
                try:
                    data = json.loads(self.store_path.read_text())
                    disk_jobs = self._parse_jobs(data)
                except Exception:
                    disk_jobs = list(self._store.jobs)  # fallback

            # 2. 合并：以磁盘任务列表为准，用内存执行状态覆盖 state 字段
            for disk_job in disk_jobs:
                mem_job = mem_index.get(disk_job.id)
                if mem_job:
                    # 仅同步执行状态，不覆盖磁盘上的 enabled/schedule/payload
                    disk_job.state = mem_job.state
                    disk_job.updated_at_ms = mem_job.updated_at_ms
                    # 同步 at 类型任务的 enabled 状态（执行后会被禁用）
                    if disk_job.schedule.kind == "at":
                        disk_job.enabled = mem_job.enabled

            # 3. 写回合并结果，并同步更新内存缓存
            merged_store = CronStore(jobs=disk_jobs)
            self._write_locked(merged_store)
            self._store.jobs = disk_jobs   # 保持内存与磁盘一致

    async def _save_store_async(self) -> None:
        """异步合并写入，offload 到线程池避免阻塞事件循环。"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_merge)

    # ─── 堆管理 ─────────────────────────────────────────────────────────────

    def _rebuild_heap(self) -> None:
        """
        从当前任务列表重建堆。
        仅在服务启动或批量变更时调用，日常增量操作用 _push_job。
        """
        if not self._store:
            return
        self._heap = [
            _HeapEntry(j.state.next_run_at_ms, j.id)
            for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms
        ]
        heapq.heapify(self._heap)

    def _push_job(self, job: CronJob) -> None:
        """将单个任务推入堆（O(log n)）。"""
        if job.enabled and job.state.next_run_at_ms:
            heapq.heappush(self._heap, _HeapEntry(job.state.next_run_at_ms, job.id))

    def _job_map(self) -> dict[str, CronJob]:
        """构建 id->job 查找字典，用于堆条目校验。"""
        if not self._store:
            return {}
        return {j.id: j for j in self._store.jobs}

    def _peek_next_wake_ms(self) -> int | None:
        """
        O(1) 查看堆顶（最早到期任务），惰性清理无效条目。

        堆中可能存在"过期条目"（任务删除/禁用后堆未立即更新）。
        Python heapq 不支持 decrease-key，采用惰性删除策略：
        推入时不删旧条目，查看/弹出时检查有效性并跳过无效条目。
        """
        job_map = self._job_map()
        while self._heap:
            entry = self._heap[0]
            job = job_map.get(entry.job_id)
            # 有效条目：任务存在、启用，且堆中记录的时间与当前一致
            if job and job.enabled and job.state.next_run_at_ms == entry.next_run_at_ms:
                return entry.next_run_at_ms
            # 惰性删除无效条目
            heapq.heappop(self._heap)
        return None

    def _pop_due_jobs(self, now_ms: int) -> list[CronJob]:
        """
        弹出所有已到期的任务（next_run_at_ms <= now_ms）。
        同时执行惰性过滤，跳过无效堆条目。
        """
        job_map = self._job_map()
        due: list[CronJob] = []
        while self._heap and self._heap[0].next_run_at_ms <= now_ms:
            entry = heapq.heappop(self._heap)
            job = job_map.get(entry.job_id)
            if job and job.enabled and job.state.next_run_at_ms == entry.next_run_at_ms:
                due.append(job)
        return due

    # ─── 定时器 ─────────────────────────────────────────────────────────────

    def _arm_timer(self, force: bool = False) -> None:
        """
        ✅ 智能重置定时器：仅当最早唤醒时间发生变化时才重建 Task。

        Args:
            force: 强制重建（任务删除等导致最早时间可能变晚的场景）
        """
        next_wake = self._peek_next_wake_ms()

        # 如果新的最早时间比当前 armed 时间更早（或强制），才需要重建
        needs_rearm = (
            force
            or next_wake is None
            or self._armed_wake_ms is None
            or next_wake < self._armed_wake_ms
        )

        if not needs_rearm:
            return

        # 取消当前定时器
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        self._armed_wake_ms = None

        if not next_wake or not self._running:
            return

        delay_s = max(0.0, (next_wake - _now_ms()) / 1000)
        self._armed_wake_ms = next_wake

        async def tick():
            await asyncio.sleep(delay_s)
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    # ─── 执行 ────────────────────────────────────────────────────────────────

    async def _on_timer(self) -> None:
        """
        定时器到期回调。

        ✅ 使用 asyncio.gather 并行执行所有同批到期任务，
        避免串行执行时任务间互相阻塞。
        """
        self._armed_wake_ms = None
        if not self._store:
            return

        now = _now_ms()
        due_jobs = self._pop_due_jobs(now)

        if due_jobs:
            # ✅ 并行执行，所有到期任务同时启动
            await asyncio.gather(*[self._execute_job(job) for job in due_jobs])

        await self._save_store_async()
        self._arm_timer(force=True)

    async def _execute_job(self, job: CronJob) -> None:
        """
        执行单个任务。

        ✅ "every" 模式漂移修正：将执行前的计划时间传给 _compute_next_run，
        确保下次执行时间基于【计划时间 + 间隔】，而非【实际完成时间 + 间隔】。
        """
        scheduled_ms = job.state.next_run_at_ms   # 保存本次计划执行时间
        start_ms = _now_ms()
        logger.info(f"Cron: executing job '{job.name}' ({job.id})")

        try:
            if self.on_job:
                await self.on_job(job)
            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info(f"Cron: job '{job.name}' completed in {_now_ms() - start_ms}ms")
        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error(f"Cron: job '{job.name}' failed: {e}")

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = _now_ms()

        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            # ✅ 传入 scheduled_ms 用于漂移修正
            job.state.next_run_at_ms = _compute_next_run(
                job.schedule, _now_ms(), last_scheduled_ms=scheduled_ms
            )
            # 将新的执行时间推入堆
            self._push_job(job)

    # ─── 生命周期 ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._rebuild_heap()
        self._arm_timer(force=True)
        logger.info(f"Cron service started with {len(self._store.jobs if self._store else [])} jobs")

    def stop(self) -> None:
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        self._armed_wake_ms = None

    def _save(self) -> None:
        """
        公共 API 的持久化入口（增删改操作调用此方法）。

        - async 上下文：后台线程调用 _save_store（完整覆盖写 + 文件锁）
        - sync 上下文（CLI）：同步调用 _save_store

        CLI 的修改是权威的（用户意图），直接覆盖写。
        守护进程后续保存时会通过 _save_merge 读取最新磁盘状态再合并。
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._save_store))
        except RuntimeError:
            self._save_store()

    def _recompute_next_runs(self) -> None:
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    # ─── 公共 API ─────────────────────────────────────────────────────────────

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        store = self._load_store()
        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )
        store.jobs.append(job)

        # ✅ 增量推入堆，不重建
        self._push_job(job)
        # ✅ 仅在新任务比当前最早时间更早时才重置定时器
        self._arm_timer()

        self._save()
        logger.info(f"Cron: added job '{name}' ({job.id})")
        return job

    def remove_job(self, job_id: str) -> bool:
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        removed = len(store.jobs) < before
        if removed:
            # 堆中的过期条目会在下次 _peek_next_wake_ms 时惰性清理
            # 必须 force=True，因为最早任务可能已被删除（时间可能变晚）
            self._arm_timer(force=True)
            self._save()
            logger.info(f"Cron: removed job {job_id}")
        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                if enabled:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                    self._push_job(job)
                    self._arm_timer()       # 新启用的任务可能更早
                else:
                    job.state.next_run_at_ms = None
                    self._arm_timer(force=True)  # 可能需要延后唤醒时间
                self._save()
                return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                await self._save_store_async()
                self._arm_timer(force=True)
                return True
        return False

    def status(self) -> dict:
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._peek_next_wake_ms(),
        }