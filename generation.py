"""FIFO generation and atomic daily quota. No NovelAI credentials are stored."""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CURRENT_JOB = ContextVar("novelai_enhanced_generation_job", default=None)
SAVED = ContextVar("novelai_enhanced_generation_saved", default=False)
SHANGHAI = timezone(timedelta(hours=8))


class Rejected(Exception):
    """A user-readable admission or generation error."""


class Duplicate(Rejected):
    pass


def read_entries(raw):
    result = set()
    for line in str(raw or "").replace("，", "\n").replace(",", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            result.add(line.split()[0])
    return result


def read_limits(raw):
    result = {}
    for line in str(raw or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split("[\\s=：:]+", line, maxsplit=1)
        if "=" in line:
            parts = line.split("=", 1)
        elif " " in line or "\t" in line:
            parts = line.split(None, 1)
        try:
            key, value = parts
            value = int(value)
            if value < -1:
                raise ValueError()
            result[key.strip()] = value
        except (ValueError, TypeError):
            raise Rejected("会话额度配置格式错误，请在插件配置页检查。") from None
    return result


@dataclass(frozen=True)
class Source:
    chat: str
    kind: str
    channel: str
    user: str
    user_key: str


class Policy:
    def __init__(self, config):
        self.config = config

    def get(self, name, default):
        return getattr(self.config(), name, default)

    def is_admin(self, source):
        return bool(
            {source.user, source.user_key} & read_entries(self.get("PLUGIN_ADMINS", ""))
        )

    def check_access(self, source):
        banned = read_entries(self.get("DRAW_BLACKLIST", ""))
        if {source.chat, source.channel, source.user, source.user_key} & banned:
            raise Rejected("画图功能已被黑名单禁用。")
        name = "GROUP" if source.kind == "group" else "PRIVATE"
        if self.get(name + "_WHITELIST_ENABLED", False):
            allowed = read_entries(self.get(name + "_WHITELIST", ""))
            identifiers = {source.chat, source.channel}
            if source.kind == "private":
                identifiers |= {source.user, source.user_key}
            if not identifiers & allowed:
                raise Rejected("此会话不在生图白名单中。")

    def limits(self, source):
        if not self.get("DAILY_QUOTA_ENABLED", False) or self.is_admin(source):
            return (-1, -1)
        prefix = "GROUP" if source.kind == "group" else "PRIVATE"
        overrides = read_limits(self.get(prefix + "_DAILY_OVERRIDES", ""))
        limit = overrides.get(
            source.chat,
            overrides.get(
                source.channel,
                self.get(prefix + "_DAILY_LIMIT", 20 if source.kind == "group" else 5),
            ),
        )
        return (int(self.get("DAILY_TOTAL_LIMIT", 100)), int(limit))


@dataclass
class Ticket:
    job_id: str
    future: asyncio.Future


class Engine:
    def __init__(self, root, config, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = Policy(config)
        self.clock = clock
        self.queue = asyncio.Queue()
        self.worker = None
        self.closed = False
        self.pending = {}
        with self.db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (\n                id TEXT PRIMARY KEY, day TEXT NOT NULL, chat TEXT NOT NULL,\n                user_key TEXT NOT NULL, dedup TEXT NOT NULL UNIQUE,\n                state TEXT NOT NULL, charged INTEGER NOT NULL DEFAULT 0,\n                exempt INTEGER NOT NULL, created REAL NOT NULL, finished REAL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS jobs_day_chat ON jobs(day,chat,state)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS jobs_state_user ON jobs(state,user_key)"
            )
            db.execute(
                "UPDATE jobs SET state='interrupted',charged=0,finished=? WHERE state IN ('queued','running')",
                (clock(),),
            )

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.root / "generation.sqlite3", timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def day(self):
        return datetime.fromtimestamp(self.clock(), SHANGHAI).strftime("%Y-%m-%d")

    def usage(self, db, day, chat=None):
        where = "day=? AND exempt=0 AND (charged=1 OR state IN ('queued','running'))"
        values = [day]
        if chat:
            where += " AND chat=?"
            values.append(chat)
        return db.execute(
            "SELECT COUNT(*) FROM jobs WHERE " + where, values
        ).fetchone()[0]

    def submit(self, source, request_id, operation):
        if self.closed:
            raise Rejected("插件正在重新加载，请稍后再试。")
        self.policy.check_access(source)
        total_limit, chat_limit = self.policy.limits(source)
        now, day = (self.clock(), self.day())
        dedup = hashlib.sha256(
            (source.chat + "|" + str(request_id)).encode()
        ).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE dedup=?", (dedup,)).fetchone():
                raise Duplicate("这条消息已经处理过，不会重复生图。")
            if db.execute(
                "SELECT 1 FROM jobs WHERE user_key=? AND state IN ('queued','running')",
                (source.user_key,),
            ).fetchone():
                raise Rejected("你还有一张图片正在生成，请完成后再试。")
            pending = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')"
            ).fetchone()[0]
            if pending >= int(self.policy.get("GENERATION_QUEUE_LIMIT", 20)):
                raise Rejected("当前生图任务较多，请稍后再试。")
            if total_limit >= 0 and self.usage(db, day) >= total_limit:
                raise Rejected("今日生图总额度已用完，次日零点恢复。")
            if chat_limit >= 0 and self.usage(db, day, source.chat) >= chat_limit:
                raise Rejected("本会话今日生图额度已用完，次日零点恢复。")
            job_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO jobs(id,day,chat,user_key,dedup,state,exempt,created) VALUES (?,?,?,?,?,'queued',?,?)",
                (
                    job_id,
                    day,
                    source.chat,
                    source.user_key,
                    dedup,
                    int(self.policy.is_admin(source)),
                    now,
                ),
            )
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        ticket = Ticket(job_id, future)
        import contextvars

        self.queue.put_nowait((ticket, source, operation, contextvars.copy_context()))
        self.pending[job_id] = ticket
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._work())
        return ticket

    def finish(self, job_id, state, saved=False):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET state=?,charged=CASE WHEN exempt=0 THEN ? ELSE 0 END,finished=? WHERE id=?",
                (state, int(saved), self.clock(), job_id),
            )

    async def _execute(self, ticket, operation):
        token, saved = (CURRENT_JOB.set(ticket.job_id), SAVED.set(False))
        try:
            value = await operation()
            self.finish(
                ticket.job_id, "succeeded" if SAVED.get() else "failed", SAVED.get()
            )
            return value
        except BaseException:
            self.finish(
                ticket.job_id, "succeeded" if SAVED.get() else "failed", SAVED.get()
            )
            raise
        finally:
            CURRENT_JOB.reset(token)
            SAVED.reset(saved)

    async def _work(self):
        while not self.closed:
            ticket, source, operation, context = await self.queue.get()
            try:
                self.policy.check_access(source)
                limits = self.policy.limits(source)
                with self.db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute(
                        "SELECT * FROM jobs WHERE id=?", (ticket.job_id,)
                    ).fetchone()
                    day = self.day()
                    if row["day"] != day or row["exempt"] != int(
                        self.policy.is_admin(source)
                    ):
                        own = int(row["day"] == day and (not row["exempt"]))
                        if limits[0] >= 0 and self.usage(db, day) - own >= limits[0]:
                            raise Rejected("今日生图总额度已用完，次日零点恢复。")
                        if (
                            limits[1] >= 0
                            and self.usage(db, day, source.chat) - own >= limits[1]
                        ):
                            raise Rejected("本会话今日生图额度已用完，次日零点恢复。")
                    db.execute(
                        "UPDATE jobs SET state='running',day=?,exempt=? WHERE id=?",
                        (day, int(self.policy.is_admin(source)), ticket.job_id),
                    )
                task = asyncio.create_task(
                    self._execute(ticket, operation), context=context
                )
                value = await task
                if not ticket.future.done():
                    ticket.future.set_result(value)
            except asyncio.CancelledError:
                with self.db() as db:
                    row = db.execute(
                        "SELECT state FROM jobs WHERE id=?", (ticket.job_id,)
                    ).fetchone()
                if row and row["state"] in ("queued", "running"):
                    self.finish(ticket.job_id, "interrupted")
                if not ticket.future.done():
                    ticket.future.set_exception(
                        Rejected("生图因插件重载而中断，未扣除额度。")
                    )
                raise
            except Exception as exc:
                with self.db() as db:
                    row = db.execute(
                        "SELECT state FROM jobs WHERE id=?", (ticket.job_id,)
                    ).fetchone()
                if row and row["state"] in ("queued", "running"):
                    self.finish(ticket.job_id, "failed")
                if not ticket.future.done():
                    ticket.future.set_exception(exc)
            finally:
                self.pending.pop(ticket.job_id, None)
                self.queue.task_done()

    async def stop(self):
        self.closed = True
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        for ticket in list(self.pending.values()):
            self.finish(ticket.job_id, "interrupted")
            if not ticket.future.done():
                ticket.future.set_exception(
                    Rejected("生图因插件重载而中断，未扣除额度。")
                )
        self.pending.clear()

    def quota_text(self, source):
        self.policy.check_access(source)
        if self.policy.is_admin(source):
            return "你是插件管理员：生图免额度，不占用每日总额度。"
        if not self.policy.get("DAILY_QUOTA_ENABLED", False):
            return "每日生图限额尚未启用。"
        total_limit, chat_limit = self.policy.limits(source)
        with self.db() as db:
            total = self.usage(db, self.day())
            chat = self.usage(db, self.day(), source.chat)

        def show(used, limit):
            return "不限" if limit < 0 else f"{max(0, limit - used)} / {limit}"

        return f"今日剩余额度（包含已预占任务）：\n全部会话共享：{show(total, total_limit)}\n当前会话：{show(chat, chat_limit)}\n北京时间每日 00:00 重置。"
