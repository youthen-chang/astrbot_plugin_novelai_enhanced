import asyncio
from types import SimpleNamespace

import httpx
import pytest
from enhanced_test.adapter import match_command, validate_triggers
from enhanced_test.generation import Rejected
from enhanced_test.request_retry import generate_request


def match_trigger(text, draw, redraw):
    return match_command(
        text,
        SimpleNamespace(
            DRAW_TRIGGER=draw, REDRAW_TRIGGER=redraw, KEEP_LEGACY_DRAW_COMMANDS=False
        ),
    )


class Client:
    def __init__(self, codes, header=None):
        self.codes = list(codes)
        self.calls = []
        self.header = header or {}

    async def post(self, url, headers, json):
        self.calls.append((headers.copy(), json.copy()))
        value = self.codes.pop(0)
        if isinstance(value, Exception):
            raise value
        return httpx.Response(value, headers=self.header, content=b"fake png")


def test_429_wait_retry_no_rotation_same_seed():
    async def scenario():
        waits = []
        client = Client([429, 429, 200], {"Retry-After": "20"})

        async def sleep(seconds):
            waits.append(seconds)

        def rotate(_):
            raise AssertionError("429 must not rotate tokens")

        result = await generate_request(
            client,
            "mock",
            {"seed": 123},
            lambda: {"Authorization": "test"},
            wait=15,
            sleep=sleep,
            rotate=rotate,
        )
        assert result.status_code == 200 and waits == [20, 20]
        assert len(client.calls) == 3 and all(v[1]["seed"] == 123 for v in client.calls)

    asyncio.run(scenario())


def test_retry_bound_final_error_no_response_body_leak():
    async def scenario():
        client = Client([429] * 4)

        async def sleep(_):
            pass

        with pytest.raises(Rejected, match="重试"):
            await generate_request(client, "mock", {}, dict, retries=3, sleep=sleep)
        assert len(client.calls) == 4

    asyncio.run(scenario())


def test_network_uncertain_not_retried():
    async def scenario():
        client = Client([httpx.ReadTimeout("secret proxy details")])
        with pytest.raises(Rejected, match="不确定") as exc:
            await generate_request(client, "mock", {}, dict)
        assert len(client.calls) == 1 and "secret" not in str(exc.value)

    asyncio.run(scenario())


@pytest.mark.parametrize("code", [401, 402, 400, 500])
def test_other_http_errors_are_sanitized(code):
    async def scenario():
        client = Client([code])
        with pytest.raises(Rejected):
            await generate_request(client, "mock", {}, dict)
        assert len(client.calls) == 1

    asyncio.run(scenario())


def test_literal_bare_trigger_and_slash_variants():
    assert match_trigger("绘图来张示例角色，风格040", "绘图来张", "绘图重画") == (
        "draw",
        "示例角色，风格040",
    )
    assert match_trigger("绘图来张+示例角色", "绘图来张", "绘图重画") == (
        "draw",
        "示例角色",
    )
    assert match_trigger("绘图重画 花田", "绘图来张", "绘图重画") == ("redraw", "花田")
    assert match_trigger("我想讨论绘图来张这个指令", "绘图来张", "绘图重画") is None
    assert match_trigger("/绘图 示例角色", "/绘图", "/重绘") == ("draw", "示例角色")
    assert match_trigger("/绘图其他描述", "/绘图", "/重绘") is None
    assert match_trigger("a.*描述", "a.*", "b") == ("draw", "描述")


@pytest.mark.parametrize(
    "draw,redraw", [("", "x"), ("x", "x"), ("x y", "z"), ("/额度", "z")]
)
def test_invalid_trigger_settings(draw, redraw):
    with pytest.raises(Rejected):
        validate_triggers(draw, redraw)


def test_429_retains_original_token_even_if_configuration_changes():
    async def scenario():
        current = ["old-test-token"]
        client = Client([429, 200])

        async def sleep(_):
            current[0] = "new-test-token"

        await generate_request(
            client,
            "mock",
            {"seed": 1},
            lambda: {"Authorization": current[0]},
            sleep=sleep,
        )
        assert all(
            call[0]["Authorization"] == "old-test-token" for call in client.calls
        )

    asyncio.run(scenario())
