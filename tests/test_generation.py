import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from enhanced_test.adapter import source_from_event
from enhanced_test.generation import SAVED, Duplicate, Engine, Policy, Rejected, Source


def config(**kwargs):
    return SimpleNamespace(
        DAILY_QUOTA_ENABLED=True,
        DAILY_TOTAL_LIMIT=4,
        GROUP_DAILY_LIMIT=2,
        PRIVATE_DAILY_LIMIT=1,
        **kwargs,
    )


def source(user="10", chat="onebot_v11-bot:group_100"):
    kind, channel = chat.split(":")[-1].split("_", 1)
    return Source(chat, kind, channel, user, chat.split(":")[0] + ":user_" + user)


async def saved(value=True):
    SAVED.set(True)
    return value


def run(f):
    return asyncio.run(f())


def test_whitelists_blacklist_and_independent_admin():
    cfg = config(
        GROUP_WHITELIST_ENABLED=True,
        GROUP_WHITELIST="100 good group",
        PRIVATE_WHITELIST_ENABLED=True,
        PRIVATE_WHITELIST="10",
        PLUGIN_ADMINS="99",
        DRAW_BLACKLIST="11 banned",
    )
    policy = Policy(lambda: cfg)
    policy.check_access(source())
    policy.check_access(source(chat="onebot_v11-bot:private_10"))
    for s in [
        source("11"),
        source("99", chat="onebot_v11-bot:group_101"),
        source("12", chat="onebot_v11-bot:private_12"),
    ]:
        with pytest.raises(Rejected):
            policy.check_access(s)
    assert policy.limits(source("99")) == (-1, -1)
    assert policy.limits(source("10")) == (4, 2)


def test_empty_enabled_whitelist_denies():
    policy = Policy(lambda: config(GROUP_WHITELIST_ENABLED=True, GROUP_WHITELIST=""))
    with pytest.raises(Rejected):
        policy.check_access(source())


def test_full_chat_overrides_and_group_private_separation():
    cfg = config(
        GROUP_DAILY_OVERRIDES="100=3\nonebot_v11-bot:group_100=7",
        PRIVATE_DAILY_OVERRIDES="100=6",
    )
    policy = Policy(lambda: cfg)
    assert policy.limits(source())[1] == 7
    assert policy.limits(source("100", "onebot_v11-bot:private_100"))[1] == 6


def test_cannot_borrow_admin_from_framework_role():
    s = source()
    assert not Policy(lambda: config()).is_admin(s)


def test_unknown_group_sender_denied():
    event = SimpleNamespace(
        get_sender_id=lambda: "",
        get_group_id=lambda: "100",
        get_platform_id=lambda: "onebot",
        get_self_id=lambda: "bot",
    )
    with pytest.raises(Rejected):
        source_from_event(event)


def test_fifo_single_request_and_no_cross_task_context(tmp_path):
    async def scenario():
        from contextvars import ContextVar

        origin = ContextVar("origin", default="missing")
        cfg = config()
        engine = Engine(tmp_path, lambda: cfg)
        gate = asyncio.Event()
        order = []
        concurrent = 0
        maximum = 0

        async def operation(label):
            nonlocal concurrent, maximum
            concurrent += 1
            maximum = max(maximum, concurrent)
            order.append((label, origin.get()))
            if label == "A":
                await gate.wait()
            SAVED.set(True)
            concurrent -= 1
            return label

        origin.set("first")
        a = engine.submit(source("10"), "m1", lambda: operation("A"))
        origin.set("second")
        b = engine.submit(source("11"), "m2", lambda: operation("B"))
        with pytest.raises(Rejected):
            engine.submit(source("10"), "m3", lambda: saved())
        await asyncio.sleep(0)
        gate.set()
        assert await a.future == "A" and await b.future == "B"
        assert order == [("A", "first"), ("B", "second")] and maximum == 1
        await engine.stop()

    run(scenario)


def test_group_shared_and_global_shared_with_private(tmp_path):
    async def scenario():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 3
        engine = Engine(tmp_path, lambda: cfg)
        for user, chat in [
            ("10", "group_100"),
            ("11", "group_100"),
            ("12", "private_12"),
        ]:
            await engine.submit(
                source(user, "onebot_v11-bot:" + chat), user, lambda: saved()
            ).future
        with pytest.raises(Rejected, match="本会话"):
            engine.submit(source("13"), "four", lambda: saved())
        with pytest.raises(Rejected, match="总额度"):
            engine.submit(
                source("14", "onebot_v11-bot:group_101"), "five", lambda: saved()
            )
        await engine.stop()

    # The exhausted total takes precedence over the exhausted chat.
    async def actual():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 3
        engine = Engine(tmp_path, lambda: cfg)
        for user in ["10", "11"]:
            await engine.submit(source(user), user, lambda: saved()).future
        with pytest.raises(Rejected, match="本会话"):
            engine.submit(source("13"), "three", lambda: saved())
        await engine.submit(
            source("12", "onebot_v11-bot:private_12"), "four", lambda: saved()
        ).future
        with pytest.raises(Rejected, match="总额度"):
            engine.submit(
                source("14", "onebot_v11-bot:group_101"), "five", lambda: saved()
            )
        await engine.stop()

    run(actual)


def test_admin_does_not_use_any_quota(tmp_path):
    async def scenario():
        cfg = config(PLUGIN_ADMINS="99")
        cfg.DAILY_TOTAL_LIMIT = 0
        engine = Engine(tmp_path, lambda: cfg)
        await engine.submit(source("99"), "one", lambda: saved()).future
        with engine.db() as db:
            assert engine.usage(db, engine.day()) == 0
        with pytest.raises(Rejected):
            engine.submit(source("10"), "two", lambda: saved())
        await engine.stop()

    run(scenario)


def test_failed_and_uncertain_refund_and_no_duplicate(tmp_path):
    async def scenario():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 1
        engine = Engine(tmp_path, lambda: cfg)

        async def fail():
            raise Rejected("uncertain")

        with pytest.raises(Rejected):
            await engine.submit(source(), "one", fail).future
        with pytest.raises(Duplicate):
            engine.submit(source(), "one", lambda: saved())
        await engine.submit(source(), "two", lambda: saved()).future
        with engine.db() as db:
            assert engine.usage(db, engine.day()) == 1
        await engine.stop()

    run(scenario)


def test_atomic_reservations_last_quota(tmp_path):
    async def scenario():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 1
        engine = Engine(tmp_path, lambda: cfg)
        gate = asyncio.Event()

        async def operation():
            await gate.wait()
            return await saved()

        ticket = engine.submit(source("10"), "one", operation)
        with pytest.raises(Rejected, match="总额度"):
            engine.submit(source("11"), "two", lambda: saved())
        gate.set()
        await ticket.future
        await engine.stop()

    run(scenario)


def test_reload_refunds_and_does_not_regenerate(tmp_path):
    async def scenario():
        cfg = config()
        engine = Engine(tmp_path, lambda: cfg)
        gate = asyncio.Event()
        a = engine.submit(source("10"), "one", lambda: gate.wait())
        b = engine.submit(source("11"), "two", lambda: saved())
        await asyncio.sleep(0.01)
        await engine.stop()
        for ticket in [a, b]:
            with pytest.raises(Rejected):
                await ticket.future
        restored = Engine(tmp_path, lambda: cfg)
        with restored.db() as db:
            assert restored.usage(db, restored.day()) == 0
        with pytest.raises(Duplicate):
            restored.submit(source("10"), "one", lambda: saved())
        await restored.stop()

    run(scenario)


def test_quota_persists_and_resets_shanghai_midnight(tmp_path):
    async def scenario():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 1
        now = [datetime(2026, 10, 13, 15, 59, 59, tzinfo=UTC).timestamp()]
        engine = Engine(tmp_path, lambda: cfg, clock=lambda: now[0])
        await engine.submit(source(), "one", lambda: saved()).future
        await engine.stop()
        engine = Engine(tmp_path, lambda: cfg, clock=lambda: now[0])
        with pytest.raises(Rejected):
            engine.submit(source(), "two", lambda: saved())
        now[0] += 1
        assert engine.day() == "2026-10-14"
        await engine.submit(source(), "three", lambda: saved()).future
        await engine.stop()

    run(scenario)


def test_waiting_job_rechecks_whitelist(tmp_path):
    async def scenario():
        cfg = config(GROUP_WHITELIST_ENABLED=True, GROUP_WHITELIST="100")
        engine = Engine(tmp_path, lambda: cfg)
        gate = asyncio.Event()

        async def first():
            await gate.wait()
            return await saved()

        a = engine.submit(source("10"), "one", first)
        b = engine.submit(source("11"), "two", lambda: saved())
        await asyncio.sleep(0.01)
        cfg.GROUP_WHITELIST = ""
        gate.set()
        await a.future
        with pytest.raises(Rejected):
            await b.future
        await engine.stop()

    run(scenario)


def test_max_queue_length(tmp_path):
    async def scenario():
        cfg = config(GENERATION_QUEUE_LIMIT=1)
        engine = Engine(tmp_path, lambda: cfg)
        gate = asyncio.Event()
        a = engine.submit(source("10"), "one", lambda: gate.wait())
        with pytest.raises(Rejected, match="较多"):
            engine.submit(source("11"), "two", lambda: saved())
        gate.set()
        await a.future
        await engine.stop()

    run(scenario)


def test_saved_image_is_charged_even_later_delivery_fails(tmp_path):
    async def scenario():
        engine = Engine(tmp_path, lambda: config())

        async def operation():
            SAVED.set(True)
            raise RuntimeError("delivery")

        with pytest.raises(RuntimeError):
            await engine.submit(source(), "one", operation).future
        with engine.db() as db:
            assert engine.usage(db, engine.day()) == 1
        await engine.stop()

    run(scenario)


def test_waiting_job_rebases_quota_after_midnight(tmp_path):
    async def scenario():
        cfg = config()
        cfg.DAILY_TOTAL_LIMIT = 2
        now = [datetime(2026, 10, 13, 15, 59, 59, tzinfo=UTC).timestamp()]
        engine = Engine(tmp_path, lambda: cfg, clock=lambda: now[0])
        gate = asyncio.Event()

        async def first():
            await gate.wait()
            return await saved()

        a = engine.submit(source("10"), "one", first)
        b = engine.submit(source("11"), "two", lambda: saved())
        await asyncio.sleep(0.01)
        now[0] += 1
        gate.set()
        await a.future
        await b.future
        with engine.db() as db:
            assert engine.usage(db, "2026-10-13") == 1
            assert engine.usage(db, "2026-10-14") == 1
        await engine.stop()

    run(scenario)
