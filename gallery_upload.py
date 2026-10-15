"""Upload generated PNGs and metadata to the gallery using httpx."""

from __future__ import annotations

import base64
import random
import time

import httpx

GALLERY_NAME_PREFIX = "astrbot_"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def build_name(prefix: str = GALLERY_NAME_PREFIX) -> str:
    return f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999):03d}"


async def upload_to_gallery(
    image_bytes: bytes,
    meta: dict,
    url: str,
    key: str,
    name: str | None = None,
    timeout: float = 300.0,
) -> tuple[bool, str]:
    """上传图片 + 生成参数到图库，返回 (是否成功, 文件名或错误信息)。"""
    if not image_bytes:
        return False, "图片为空"
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return False, "图片超过图库大小上限"
    if not image_bytes.startswith(PNG_MAGIC):
        return False, "只支持 PNG 图片"
    name = name or build_name()
    payload = {
        "name": name,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "meta": dict(meta or {}),
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                url, json=payload, headers={"X-Upload-Key": key}
            )
    except Exception as exc:
        return False, f"请求失败: {exc}"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.text[:120]}"
    try:
        result = response.json()
    except Exception:
        return False, "响应不是 JSON"
    if not isinstance(result, dict) or not result.get("ok"):
        detail = result.get("error") if isinstance(result, dict) else None
        return False, str(detail or "图库返回失败")
    return True, str(result.get("name") or name)
