"""Small, explainable prompt rules. Negative prompts never imply R18."""

import re

ADULT_TAGS = frozenset(
    {
        "nsfw",
        "r18",
        "r-18",
        "nude",
        "naked",
        "completely nude",
        "sex",
        "sexual intercourse",
        "penetration",
        "vaginal",
        "anal",
        "penis",
        "pussy",
        "cum",
        "nipples",
    }
)


def tokens(prompt):
    result = []
    for piece in re.split(r"[,\n]", prompt.lower()):
        piece = re.sub(r"\d+(?:\.\d+)?::", "", piece).replace("::", "")
        piece = piece.strip(" {}[]")
        piece = re.sub(r":\d+(?:\.\d+)?\)$", "", piece).lstrip("(")
        if piece:
            result.append(piece.replace("\\(", "(").replace("\\)", ")"))
    return result


def classify(meta, catalog=None):
    prompt = meta.get("prompt", "")
    raw = meta.get("raw_prompt", "")
    prompt = prompt if isinstance(prompt, str) else ""
    raw = raw if isinstance(raw, str) else ""
    positive = set(tokens(prompt))
    matched = sorted(positive & ADULT_TAGS)
    # Explicit bot metadata is accepted only through the authenticated uploader.
    if meta.get("r18") is True:
        matched.append("机器人R18标记")
    rating = "sensitive" if matched else "unknown"
    counts = {}
    for n, gender in re.findall(r"\b(\d+)(girls?|boys?)\b", prompt.lower()):
        kind = "girl" if gender.startswith("girl") else "boy"
        counts[kind] = max(counts.get(kind, 0), min(int(n), 99))
    people = [f"{n}{kind}{'s' if n != 1 else ''}" for kind, n in counts.items() if n]
    characters, styles = [], []
    catalog = catalog or {}
    for name, value in catalog.get("characters", {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        # Matching a character's canonical series tag is more reliable than
        # generic clothing/quality tags shared by character presets.
        canonical = [t for t in tokens(value) if "(" in t and ")" in t]
        if name in raw or any(t in positive for t in canonical):
            characters.append(name)
    for name in catalog.get("styles", {}):
        if isinstance(name, str) and re.search(re.escape(name) + r"(?!\d)", raw):
            styles.append(name)
    for number in re.findall(r"风格\s*(\d{1,4})(?!\d)", raw):
        label = "风格" + number
        if label not in styles:
            styles.append(label)
    supplied = meta.get("gallery_tags", {})
    if isinstance(supplied, dict):
        for kind, values in (("characters", characters), ("styles", styles)):
            for value in (
                supplied.get(kind, []) if isinstance(supplied.get(kind), list) else []
            ):
                if (
                    isinstance(value, str)
                    and 0 < len(value) <= 80
                    and value not in values
                ):
                    values.append(value)
    return {
        "rating": rating,
        "reason": "正向标签：" + "、".join(matched)
        if matched
        else "提示词未命中明确R18规则",
        "tags": {
            "people": people[:4],
            "characters": characters[:20],
            "styles": styles[:10],
        },
    }
