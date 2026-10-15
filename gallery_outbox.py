"""Durable NovelAI upload outbox. Also installed beside the existing plugin.

Only PNGs and metadata are persisted; retry credentials are read from the
plugin's existing configuration. One worker sends jobs, including after restart.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

log = logging.getLogger("novelai_enhanced.outbox")


class Outbox:
    def __init__(
        self,
        root,
        sender,
        config,
        tagger=None,
        retry_base=30,
        clock=time.time,
        recover=True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sender, self.config, self.tagger = sender, config, tagger
        self.retry_base, self.clock = retry_base, clock
        self.worker = self.reporter = None
        self.wake = asyncio.Event()
        self.boot_id = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:20]
        with self.db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, name TEXT NOT NULL,
                meta TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL, next_try REAL NOT NULL, error TEXT NOT NULL DEFAULT '')""")
            # A previous process may have died after the remote server saved
            # the PNG. Retrying is safe because the gallery deduplicates bytes.
            if recover:
                db.execute(
                    "UPDATE jobs SET state='retry',next_try=0 WHERE state='uploading'"
                )

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.root / "outbox.db", timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def enqueue(self, image, meta, name):
        if not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 32 * 1024 * 1024:
            raise ValueError("Invalid PNG upload")
        meta = dict(meta or {})
        if self.tagger:
            meta["gallery_tags"] = self.tagger(meta)
        encoded = json.dumps(meta, ensure_ascii=False)
        if len(encoded.encode()) > 65536:
            raise ValueError("Metadata is too large")
        job_id = hashlib.sha256(image).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old:
                return old["name"]
            path = self.root / (job_id + ".png")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(image)
            temporary.replace(path)
            now = self.clock()
            db.execute(
                "INSERT INTO jobs(id,name,meta,state,created,updated,next_try) VALUES (?,?,?,'queued',?,?,0)",
                (job_id, name, encoded, now, now),
            )
        self.wake.set()
        return name

    def snapshot(self):
        with self.db() as db:
            counts = {
                r["state"]: r["n"]
                for r in db.execute("SELECT state,COUNT(*) n FROM jobs GROUP BY state")
            }
            recent = [
                dict(r)
                for r in db.execute(
                    "SELECT name,state,attempts,created,updated,next_try,error FROM jobs ORDER BY created DESC LIMIT 12"
                )
            ]
        return {"boot_id": self.boot_id, "counts": counts, "recent": recent}

    async def report(self):
        try:
            config = self.config()
            url, key = (
                config.get("GALLERY_UPLOAD_URL", ""),
                config.get("GALLERY_UPLOAD_KEY", ""),
            )
            if not url or not key:
                return
            parts = urlsplit(url)
            endpoint = urlunsplit(
                (parts.scheme, parts.netloc, "/api/ingest/status", "", "")
            )
            async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
                await client.post(
                    endpoint, json=self.snapshot(), headers={"X-Upload-Key": key}
                )
        except Exception:
            # Reporting must never block or invalidate an image upload.
            pass

    async def process_once(self):
        config = self.config()
        url, key = (
            config.get("GALLERY_UPLOAD_URL", ""),
            config.get("GALLERY_UPLOAD_KEY", ""),
        )
        if not url or not key:
            return False
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE state IN ('queued','retry') AND next_try<=? ORDER BY created LIMIT 1",
                (self.clock(),),
            ).fetchone()
            if not row:
                return False
            db.execute(
                "UPDATE jobs SET state='uploading',attempts=attempts+1,updated=? WHERE id=?",
                (self.clock(), row["id"]),
            )
        await self.report()
        try:
            image = (self.root / (row["id"] + ".png")).read_bytes()
            ok, detail = await self.sender(
                image, json.loads(row["meta"]), url, key, name=row["name"], timeout=300
            )
            # Never record URLs, headers, credentials or arbitrary remote bodies.
            error = (
                ""
                if ok
                else (
                    detail.split(":", 1)[0][:80]
                    if detail.startswith("HTTP ")
                    else "网络连接或上传失败"
                )
            )
        except asyncio.CancelledError:
            with self.db() as db:
                db.execute(
                    "UPDATE jobs SET state='retry',next_try=0 WHERE id=?", (row["id"],)
                )
            raise
        except Exception as exc:
            ok, error = False, type(exc).__name__
        attempts = row["attempts"] + 1
        delay = min(900, self.retry_base * 2 ** min(attempts - 1, 5))
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET state=?,updated=?,next_try=?,error=? WHERE id=?",
                (
                    "uploaded" if ok else "retry",
                    self.clock(),
                    0 if ok else self.clock() + delay,
                    error,
                    row["id"],
                ),
            )
        if ok:
            (self.root / (row["id"] + ".png")).unlink(missing_ok=True)
            log.info("Gallery job archived (%s attempt(s))", attempts)
        else:
            log.warning("Gallery job retained for retry: %s", error)
        await self.report()
        return True

    async def run(self):
        while True:
            try:
                if await self.process_once():
                    continue
                self.wake.clear()
                await asyncio.wait_for(self.wake.wait(), timeout=15)
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Outbox worker paused; jobs retained")
                await asyncio.sleep(15)

    async def heartbeat(self):
        while True:
            await self.report()
            await asyncio.sleep(45)

    def start(self):
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self.run())
        if self.reporter is None or self.reporter.done():
            self.reporter = asyncio.create_task(self.heartbeat())

    async def stop(self):
        tasks = [t for t in (self.worker, self.reporter) if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.worker = self.reporter = None
