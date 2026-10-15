"""AstrBot identity, literal command matching and bounded reference downloads."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
from PIL import Image as PILImage

from .generation import Rejected, Source


def source_from_event(event):
    """Build quota identity from trusted platform fields.

    Args:
        event: AstrBot platform event.

    Returns:
        Identity with separate group/private and adapter/bot scopes.

    Raises:
        Rejected: A required platform identity is missing.
    """
    user = str(event.get_sender_id() or "")
    group = str(event.get_group_id() or "")
    adapter = str(event.get_platform_id() or "")
    bot = str(event.get_self_id() or "")
    if not user or not adapter or not bot:
        raise Rejected("无法确定请求来自哪个会话或用户，未提交生图。")
    kind, channel = ("group", group) if group else ("private", user)
    namespace = f"{adapter}:{bot}"
    return Source(
        f"{namespace}:{kind}_{channel}", kind, channel, user, f"{namespace}:user_{user}"
    )


RESERVED = {
    "额度",
    "来点",
    "来点nai",
    "laidian",
    "看参数",
    "反推",
    "查看参数",
    "naimeta",
    "添加人物",
    "删除人物",
    "人物列表",
    "添加风格",
    "删除风格",
    "风格列表",
    "拉黑",
    "画图拉黑",
    "解除拉黑",
    "移出黑名单",
    "黑名单",
    "黑名单列表",
    "生成画图密钥",
    "生成密钥",
    "增加画图次数",
    "密钥加额",
    "画图密钥列表",
    "删除画图密钥",
    "绘图帮助",
}


def validate_triggers(draw, redraw):
    for value in (draw, redraw):
        if (
            not isinstance(value, str)
            or not value.strip("/")
            or len(value) > 40
            or any(c.isspace() for c in value)
        ):
            raise Rejected("画图和重画触发词不能为空、不能含空白，最长 40 字。")
        if value.lstrip("/") in RESERVED:
            raise Rejected("触发词不能与插件其他指令重名。")
    if draw == redraw:
        raise Rejected("画图和重画触发词不能相同。")


def match_command(text, config):
    """Match literal triggers without executing user-supplied regex.

    Args:
        text: Current message text.
        config: Effective plugin configuration.

    Returns:
        A (command, argument) tuple, or None.
    """
    text = str(text or "").strip()
    draw, redraw = config.DRAW_TRIGGER, config.REDRAW_TRIGGER
    validate_triggers(draw, redraw)
    triggers = [("draw", draw), ("redraw", redraw)]
    if config.KEEP_LEGACY_DRAW_COMMANDS:
        triggers += [("draw", "/" + n) for n in ("nai", "画图", "nai画图", "novelai")]
        triggers += [("redraw", "/重画"), ("redraw", "/redraw")]
    # Bare custom triggers can directly join the prompt. Slash commands require a boundary.
    for kind, prefix in sorted(triggers, key=lambda item: len(item[1]), reverse=True):
        if text.startswith(prefix):
            tail = text[len(prefix) :]
            if (
                prefix.startswith("/")
                and tail
                and not tail[0].isspace()
                and tail[0] not in "+＋:：,，"
            ):
                continue
            return kind, tail.lstrip(" \t\r\n+＋:：,，")
    if text.startswith("/"):
        parts = text[1:].split(None, 1)
        if parts and parts[0] in RESERVED:
            return parts[0], parts[1].strip() if len(parts) > 1 else ""
    return None


async def read_reference(component, normalize=True):
    """Load an image from a trusted AstrBot image component with size limits.

    Args:
        component: Image component resolved from the platform event.
        normalize: Convert to metadata-free PNG for image-to-image requests.

    Returns:
        A normalized PNG without embedded metadata.

    Raises:
        Rejected: The image is missing, invalid or too large.
    """
    limit = 32 * 1024 * 1024
    location = component.url or component.file or component.path or ""
    try:
        if location.startswith("base64://"):
            if len(location) > limit * 4 // 3 + 20:
                raise ValueError("Image too large")
            data = base64.b64decode(location[9:], validate=True)
        elif location.startswith(("https://", "http://")):
            async with httpx.AsyncClient(
                timeout=30, trust_env=False, follow_redirects=True
            ) as client:
                async with client.stream("GET", location) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Image too large")
                        chunks.append(chunk)
                    data = b"".join(chunks)
        else:
            # Platform file URI handling is provided by AstrBot, including Windows paths.
            path = Path(await component.convert_to_file_path())
            if path.stat().st_size > limit:
                raise ValueError("Image too large")
            data = path.read_bytes()
        if len(data) > limit:
            raise ValueError("Image too large")
        with PILImage.open(io.BytesIO(data)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("Image dimensions too large")
            if not normalize:
                image.verify()
                return data
            image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        raise Rejected(
            "无法读取参考图片，或图片超过 32 MB / 3200 万像素，请重新发送图片。"
        ) from None
