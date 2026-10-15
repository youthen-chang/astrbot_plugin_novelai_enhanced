"""Separate explicit extra-negative text without interpreting it as positive tags."""

import re

from .generation import Rejected

NEGATIVE_MARKER = re.compile(r"(?<!-)-n[:：][ \t]*", re.IGNORECASE)
# Do not interpret the decimal point in weighted tags such as 1.5 as the end.
NEGATIVE_END = re.compile(r"。|(?<!\d)\.|\.(?!\d)")


def split_extra_negative(text):
    text = str(text or "").strip()
    matches = list(NEGATIVE_MARKER.finditer(text))
    if not matches:
        return text, "", False
    if len(matches) > 1:
        raise Rejected("额外负面提示词请只写一段：-n:内容。")
    marker = matches[0]
    end = NEGATIVE_END.search(text, marker.end())
    if end is None:
        raise Rejected("额外负面提示词须以句号 . 或 。结束，例如 -n:glasses, hat。")
    before = text[: marker.start()].strip(" ,，\n\r\t")
    after = text[end.end() :].strip(" ,，\n\r\t")
    positive = ", ".join(part for part in (before, after) if part)
    negative = text[marker.end() : end.start()].strip()
    if len(negative) > 4000:
        raise Rejected("额外负面提示词过长，请控制在 4000 字以内。")
    return positive, negative, True
