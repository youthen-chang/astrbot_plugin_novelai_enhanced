"""Integration against installed AstrBot; all provider and chat I/O is simulated."""

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image as PILImage

if not os.environ.get("ASTRBOT_SOURCE_DIR"):
    pytest.skip(
        "Set ASTRBOT_SOURCE_DIR to run real-framework integration",
        allow_module_level=True,
    )

from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.web import PluginRequest, bind_request_context
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from enhanced_test.adapter import match_command, read_reference, source_from_event
from enhanced_test.generation import Duplicate, Rejected
from enhanced_test.main import NovelAIEnhanced
from enhanced_test.settings import DEFAULTS
from starlette.requests import Request


class Config(dict):
    def save_config(self):
        self.saved = True


class Event:
    def __init__(
        self, text="/画图 1girl", user="10", group="100", mid="1", images=None
    ):
        self.text, self.user, self.group = text, user, group
        self.message_obj = SimpleNamespace(message_id=mid)
        self.parts = [Plain(text), *(images or [])]
        self.sent = []
        self.stopped = False

    def get_messages(self):
        return self.parts

    def get_message_str(self):
        return self.text.lstrip("/")

    def get_sender_id(self):
        return self.user

    def get_group_id(self):
        return self.group

    def get_platform_id(self):
        return "napcat"

    def get_self_id(self):
        return "123456"

    def stop_event(self):
        self.stopped = True

    async def send(self, chain):
        self.sent.append(chain)


def png():
    output = io.BytesIO()
    PILImage.new("RGB", (32, 48), "#be8caf").save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    from enhanced_test.main import StarTools

    monkeypatch.setattr(StarTools, "get_data_dir", lambda *a: tmp_path)
    context = SimpleNamespace(registered_web_apis=[])
    context.register_web_api = lambda *args: context.registered_web_apis.append(args)
    config = Config(DEFAULTS | {"API_TOKEN": "test-dummy-token"})
    instance = NovelAIEnhanced(context, config)

    async def fake_generate(prompt, width=0, height=0, **kwargs):
        return png(), {
            "prompt": prompt,
            "raw_prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "model": "nai-diffusion-4-5-full",
            "width": width or 832,
            "height": height or 1216,
            "steps": 28,
            "scale": 7,
            "sampler": "k_euler",
            "seed": 123,
            "image_type": "txt2img",
        }

    monkeypatch.setattr(instance.service, "_call_txt2img", fake_generate)
    return instance


def test_actual_framework_combined_message_and_wake_prefix(plugin):
    async def scenario():
        event = Event()
        await plugin.on_message(event)
        await asyncio.gather(*list(plugin.tasks))
        assert event.stopped and len(event.sent) == 2
        assert [type(p) for p in event.sent[-1].chain] == [Plain, Image]
        assert event.sent[-1].chain[0].text == "NovelAI 画图完成"
        assert "≡NA≡" not in event.sent[-1].chain[0].text
        calls = []

        async def send_group_msg(**kw):
            calls.append(kw)

        bot = SimpleNamespace(send_group_msg=send_group_msg)
        await AiocqhttpMessageEvent.send_message(
            bot, event.sent[-1], is_group=True, session_id="100"
        )
        assert len(calls) == 1 and calls[0]["group_id"] == 100
        assert [part["type"] for part in calls[0]["message"]] == ["text", "image"]
        assert calls[0]["message"][1]["data"]["file"].startswith("base64://")
        await plugin.terminate()

    asyncio.run(scenario())


def test_manual_and_agent_share_quota_and_dedup(plugin):
    async def scenario():
        plugin.config.update(
            DAILY_QUOTA_ENABLED=True,
            DAILY_TOTAL_LIMIT=2,
            GROUP_DAILY_OVERRIDES="100=1",
            PRIVATE_DAILY_LIMIT=1,
        )
        first = Event()
        ticket = plugin.submit(first, "draw", "1girl -N：hat, glasses。")
        with pytest.raises(Duplicate):
            plugin.submit(first, "draw", "1girl")
        with pytest.raises(Rejected, match="本会话"):
            plugin.submit(Event(user="11", mid="2"), "draw", "1girl")
        await ticket.future
        state = plugin.last[source_from_event(first).chat]
        assert state["extra_negative"] == "hat, glasses"
        result = await plugin.novelai_generate(Event(user="11", mid="3"), "1girl")
        assert "本会话" in result
        private = plugin.submit(Event(user="20", group="", mid="4"), "draw", "1girl")
        await private.future
        with pytest.raises(Rejected):
            plugin.submit(Event(user="20", group="", mid="5"), "draw", "1girl")
        plugin.config["PLUGIN_ADMINS"] = "99"
        for mid in ("6", "7"):
            await plugin.submit(Event(user="99", mid=mid), "draw", "1girl").future
        with plugin.engine.db() as db:
            assert plugin.engine.usage(db, plugin.engine.day()) == 2
        await plugin.terminate()

    asyncio.run(scenario())


def test_redraw_snapshot_negative_replace_and_clear(plugin):
    async def scenario():
        event = Event()
        await plugin.submit(event, "draw", "1girl -n:hat。").future
        await plugin.submit(Event(mid="2"), "redraw", "smile").future
        key = source_from_event(event).chat
        assert plugin.last[key]["extra_negative"] == "hat"
        await plugin.submit(Event(mid="3"), "redraw", "-n:glasses。").future
        assert plugin.last[key]["extra_negative"] == "glasses"
        await plugin.submit(Event(mid="4"), "redraw", "-n:。").future
        assert plugin.last[key]["extra_negative"] == ""
        parameters = await plugin.command(event, "看参数", "")
        assert "seed: 123" in parameters and "1girl" in parameters
        await plugin.terminate()

    asyncio.run(scenario())


def test_reference_image_and_img2img_use_shared_queue(plugin, monkeypatch):
    async def scenario():
        calls = []

        async def img(prompt, image_b64, *args, **kwargs):
            calls.append(base64.b64decode(image_b64))
            return await plugin.service._call_txt2img(prompt, **kwargs)

        monkeypatch.setattr(plugin.service, "_call_img2img", img)
        image = Image.fromBytes(png())
        event = Event(images=[Reply(id="20", chain=[image])])
        await plugin.submit(event, "img2img", "1girl").future
        assert len(calls) == 1 and calls[0].startswith(b"\x89PNG")
        with pytest.raises(Rejected, match="尺寸"):
            plugin.submit(Event(mid="2"), "draw", "1girl", size="4000x4000")
        result = await read_reference(image, normalize=False)
        assert result == png()
        await plugin.terminate()

    asyncio.run(scenario())


def test_delivery_failure_does_not_regenerate_or_refund_saved_image(
    plugin, monkeypatch
):
    async def scenario():
        plugin.config.update(DAILY_QUOTA_ENABLED=True, DAILY_TOTAL_LIMIT=1)
        event = Event()

        async def failed_send(chain):
            raise RuntimeError("Simulated NapCat failure")

        event.send = failed_send
        ticket = plugin.submit(event, "draw", "1girl")
        await plugin.deliver(event, ticket)
        assert len(list((plugin.root / "generated").glob("*.png"))) == 1
        with plugin.engine.db() as db:
            assert plugin.engine.usage(db, plugin.engine.day()) == 1
        with pytest.raises(Rejected):
            plugin.submit(Event(user="20", mid="2"), "draw", "1girl")
        await plugin.terminate()

    asyncio.run(scenario())


def test_white_blacklist_admin_and_instance_checks_on_tools(plugin):
    async def scenario():
        plugin.config.update(
            GROUP_WHITELIST_ENABLED=True,
            GROUP_WHITELIST="100",
            PLUGIN_ADMINS="99",
            DRAW_BLACKLIST="99 blocked",
        )
        result = await plugin.novelai_generate(Event(user="99"), "1girl")
        assert "黑名单" in result
        result = await plugin.novelai_img2img(Event(group="101"), "1girl")
        assert "白名单" in result
        plugin.config["ONLY_INSTANCE_ID"] = "654321"
        assert "指定机器人" in await plugin.novelai_generate(Event(), "1girl")
        await plugin.terminate()

    asyncio.run(scenario())


def test_presets_page_uses_framework_request_and_persists(plugin):
    async def scenario():
        payload = json.dumps(
            {"kind": "styles", "name": "test", "prompt": "watercolor"}
        ).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        http_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/test",
                "headers": [],
                "query_string": b"",
                "server": ("test", 80),
                "scheme": "http",
            },
            receive,
        )
        with bind_request_context(PluginRequest(http_request, username="test-admin")):
            result = await plugin.page_save()
        assert (
            result.status_code == 200
            and plugin.presets.get("styles", "test") == "watercolor"
        )
        assert len(plugin.presets.all()["characters"]) == 1062
        assert all(
            api[0].startswith("/astrbot_plugin_novelai_enhanced/")
            for api in plugin.context.registered_web_apis
        )
        await plugin.terminate()
        assert not plugin.context.registered_web_apis

    asyncio.run(scenario())


def test_custom_triggers_are_literal_and_unrelated_chats_untouched(plugin):
    async def scenario():
        plugin.config.update(DRAW_TRIGGER="西西来点", REDRAW_TRIGGER="西西重画")
        assert match_command("西西来点+1girl", plugin.service.cfg) == ("draw", "1girl")
        event = Event("普通讨论画图")
        await plugin.on_message(event)
        assert not event.stopped and not event.sent and not plugin.tasks
        await plugin.terminate()

    asyncio.run(scenario())


def test_schema_has_editable_token_and_safe_install_defaults():
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    assert schema["API_TOKEN"]["secret"] and not schema["API_TOKEN"].get("invisible")
    for key in (
        "GROUP_WHITELIST_ENABLED",
        "PRIVATE_WHITELIST_ENABLED",
        "DAILY_QUOTA_ENABLED",
    ):
        assert schema[key]["default"] is False
    assert schema["API_PROXY"]["default"] == schema["HTTP_PROXY"]["default"] == ""
    assert schema["TRANSLATE_PROVIDER_ID"]["_special"] == "select_provider"


def test_character_swap_accepts_ascii_comma_description(plugin, monkeypatch):
    async def scenario():
        calls = []

        async def img(prompt, image_b64, *args, **kwargs):
            calls.append(prompt)
            return await plugin.service._call_txt2img(prompt, **kwargs)

        monkeypatch.setattr(plugin.service, "_call_img2img", img)
        character = next(iter(plugin.presets.all()["characters"]))
        event = Event(images=[Image.fromBytes(png())])
        path, text = await plugin.submit(event, "来点", character + ", smile").future
        assert character in text and calls == [character + ", smile"]
        assert path.is_file()
        await plugin.terminate()

    asyncio.run(scenario())
