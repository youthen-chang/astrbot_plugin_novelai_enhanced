import asyncio
import json

from enhanced_test.classification import classify
from enhanced_test.gallery_outbox import Outbox


def test_failed_gallery_upload_recovers_without_drawing_again(tmp_path):
    async def scenario():
        attempts = []
        image = b"\x89PNG\r\n\x1a\n" + b"test-outbox-bytes"
        config = {
            "GALLERY_UPLOAD_URL": "https://gallery.test/api/upload",
            "GALLERY_UPLOAD_KEY": "test-only",
        }

        async def sender(data, meta, url, key, **kwargs):
            attempts.append((data, meta, kwargs["name"]))
            return len(attempts) > 1, "HTTP 502: private response body"

        clock = [100]
        box = Outbox(
            tmp_path, sender, lambda: config, clock=lambda: clock[0], retry_base=1
        )

        async def report():
            pass

        box.report = report
        box.enqueue(image, {"source": "AstrBot", "raw_prompt": "test"}, "astrbot_test")
        box.enqueue(image, {"source": "AstrBot"}, "duplicate")
        assert box.snapshot()["counts"] == {"queued": 1}
        await box.process_once()
        assert box.snapshot()["counts"] == {"retry": 1}
        clock[0] += 2
        recovered = Outbox(tmp_path, sender, lambda: config, clock=lambda: clock[0])
        recovered.report = report
        await recovered.process_once()
        assert recovered.snapshot()["counts"] == {"uploaded": 1}
        assert len(attempts) == 2 and attempts[0][0] == attempts[1][0]
        assert attempts[1][2] == "astrbot_test"
        assert not list(tmp_path.glob("*.png"))
        assert "private response" not in json.dumps(recovered.snapshot())

    asyncio.run(scenario())


def test_negative_tags_do_not_mark_sfw_image_sensitive():
    result = classify(
        {"prompt": "1girl, white dress", "negative_prompt": "nsfw, nude, nipples"}
    )
    assert result["rating"] == "unknown"
    assert result["tags"]["people"] == ["1girl"]
