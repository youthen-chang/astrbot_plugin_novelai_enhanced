"""Bounded silent 429 retry with no credential/body leakage."""

import asyncio
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from .generation import Rejected


def retry_seconds(response, default):
    raw = response.headers.get("Retry-After", "")
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        try:
            delay = (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            delay = default
    return max(0, min(600, max(default, delay)))


async def generate_request(
    client,
    url,
    payload,
    headers,
    retries=3,
    wait=15,
    rotate=None,
    sleep=asyncio.sleep,
    log=None,
):
    # Same payload, including seed, is retained across explicit 429 responses.
    attempts = 0
    rotations = set()
    request_headers = headers()
    while True:
        try:
            response = await client.post(url, headers=request_headers, json=payload)
        except Exception as exc:
            # A network timeout can leave a generation running remotely.
            # Do not automatically regenerate an uncertain result.
            import httpx

            if isinstance(exc, httpx.RequestError):
                raise Rejected(
                    "NovelAI 网络请求中断，结果不确定，未扣除插件额度。"
                ) from None
            raise
        if response.status_code == 200:
            return response
        if response.status_code == 429:
            if attempts >= retries:
                raise Rejected("NovelAI 请求受限，后台重试后仍未成功，未扣除额度。")
            delay = retry_seconds(response, wait)
            attempts += 1
            if log:
                log.warning(
                    f"NovelAI HTTP 429; waiting {delay:g}s, retry {attempts}/{retries}"
                )
            await sleep(delay)
            continue
        if response.status_code in (401, 402):
            token = request_headers.get("Authorization", "")
            if token not in rotations and rotate:
                rotations.add(token)
                if rotate(f"HTTP {response.status_code}"):
                    request_headers = headers()
                    if request_headers.get("Authorization", "") not in rotations:
                        continue
            reason = "密钥无效" if response.status_code == 401 else "余额或配额不足"
            raise Rejected(f"NovelAI {reason}，请检查插件配置。")
        raise Rejected(f"NovelAI 请求失败（HTTP {response.status_code}），未扣除额度。")
