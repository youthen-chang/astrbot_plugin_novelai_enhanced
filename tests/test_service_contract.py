"""Exercise the real request builder against an in-memory HTTP transport."""

import asyncio
import io
import json
import os
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image as PILImage

if not os.environ.get("ASTRBOT_SOURCE_DIR"):
    pytest.skip(
        "Set ASTRBOT_SOURCE_DIR for framework integration", allow_module_level=True
    )

from enhanced_test.generation import Engine, Rejected, Source
from enhanced_test.presets import PresetStore
from enhanced_test.service import NovelAIService
from enhanced_test.settings import DEFAULTS


def test_actual_api_payload_proxy_v4_negatives_and_metadata(tmp_path, monkeypatch):
    async def scenario():
        calls = []
        output = io.BytesIO()
        PILImage.new("RGB", (32, 48), "pink").save(output, "PNG")

        def handle(request):
            assert str(request.url) == "https://proxy.test/ai/generate-image"
            assert request.headers["Authorization"] == "Bearer dummy-test-token"
            calls.append(json.loads(request.content))
            return httpx.Response(200, content=output.getvalue())

        cfg = DEFAULTS | {
            "API_TOKEN": "dummy-test-token",
            "API_PROXY": "https://proxy.test",
            "NEGATIVE_PROMPT": "lowres",
        }
        service = NovelAIService(cfg, PresetStore(tmp_path), None)
        monkeypatch.setattr(
            service,
            "_build_client_config",
            lambda: {"transport": httpx.MockTransport(handle), "trust_env": False},
        )
        engine = Engine(tmp_path / "jobs", lambda: service.cfg)
        source = Source("test:group_1", "group", "1", "10", "test:user_10")

        async def draw():
            return await service._call_txt2img("1girl", negative_prompt="hat")

        data, info = await engine.submit(source, "1", draw).future
        payload = calls[0]
        assert (
            payload["action"] == "generate" and payload["parameters"]["n_samples"] == 1
        )
        assert payload["parameters"]["v4_prompt"]["caption"]["base_caption"] == "1girl"
        negatives = payload["parameters"]["negative_prompt"]
        assert all(value in negatives for value in ("lowres", "nsfw", "hat"))
        assert (
            payload["parameters"]["v4_negative_prompt"]["caption"]["base_caption"]
            == negatives
        )
        assert (
            info["seed"] == payload["parameters"]["seed"] and data == output.getvalue()
        )
        cfg["ENABLE_R18"] = True
        assert "nsfw" not in service._get_negative_prompt()
        assert "lowres" in service._get_negative_prompt()
        await engine.stop()

    asyncio.run(scenario())


def test_translation_uses_astrbot_provider_and_blocks_failed_translation(tmp_path):
    async def scenario():
        calls = []

        async def translate(**kw):
            calls.append(kw)
            return SimpleNamespace(completion_text="white dress")

        cfg = DEFAULTS | {
            "TRANSLATE_PROVIDER_ID": "test-provider",
            "TRANSLATE_MODEL_NAME": "test-model",
        }
        context = SimpleNamespace(llm_generate=translate)
        service = NovelAIService(cfg, PresetStore(tmp_path), context)
        assert await service._translate_prompt("1girl，白裙") == "1girl, white dress"
        assert len(calls) == 1 and calls[0]["chat_provider_id"] == "test-provider"
        assert calls[0]["model"] == "test-model"

        async def failed(**kw):
            raise RuntimeError("private provider detail")

        context.llm_generate = failed
        with pytest.raises(Rejected, match="翻译失败") as exc:
            await service._translate_prompt("白裙")
        assert "private" not in str(exc.value)

    asyncio.run(scenario())
