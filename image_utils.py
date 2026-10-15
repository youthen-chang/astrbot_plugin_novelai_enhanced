"""NovelAI image parsing and size handling, derived from luoxiQAQ."""

import base64
import io
import json
import re
import zipfile

NAI_API_BASE = "https://image.novelai.net"

V4_STYLE_MODELS = {
    "nai-diffusion-4",
    "nai-diffusion-4-curated-preview",
    "nai-diffusion-4.5-full",
    "nai-diffusion-4-5-full",
    "nai-diffusion-4-5-curated",
    "nai-diffusion-5-full",
    "nai-diffusion-5-curated",
}

NSFW_NEGATIVE_TAGS = "nsfw, nude, naked, nipples, pussy, penis, sex, vaginal, anal, oral, cum, ejaculation, pubic hair, genitals, exposed, uncensored"

TRANSLATE_SYSTEM_PROMPT = "You are a prompt translator for NovelAI image generation. Translate the user's description into English danbooru-style tags.\nRules:\n1. Output ONLY comma-separated English tags, no explanations\n2. Use danbooru tag format (lowercase, underscores for multi-word tags)\n3. Always start with quality tags: masterpiece, best quality, very aesthetic\n4. Include character count tags (1girl, 1boy, 2girls, etc.)\n5. Translate clothing, pose, expression, background into appropriate tags\n6. Keep any English tags the user already provided as-is\n7. If input is already English tags, return with quality tags prepended\n8. Do NOT add NSFW tags unless explicitly requested"

RATIO_PRESETS = {
    "竖": (832, 1216),
    "portrait": (832, 1216),
    "2:3": (832, 1216),
    "方": (1024, 1024),
    "square": (1024, 1024),
    "1:1": (1024, 1024),
    "横": (1216, 832),
    "landscape": (1216, 832),
    "3:2": (1216, 832),
}


def _is_v4_style(model: str) -> bool:
    return model.lower() in V4_STYLE_MODELS


def _model_label(model: str) -> str:
    ml = model.lower()
    if "5-full" in ml or "5-curated" in ml:
        return "v5"
    if "4-5" in ml or "4.5" in ml:
        return "v4.5"
    if "4" in ml:
        return "v4"
    return "v3"


def _create_v4_prompt(prompt: str) -> dict:
    return {
        "caption": {
            "base_caption": prompt,
            "char_captions": [],
            "scenery_captions": [],
        },
        "use_coords": False,
        "use_order": True,
    }


def _strip_png_metadata(data: bytes) -> bytes:
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as img:
            img.info.clear()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return data


def _extract_image(response_data: bytes) -> bytes | None:
    try:
        from PIL import Image as PILImage

        try:
            img = PILImage.open(io.BytesIO(response_data))
            img.close()
            return response_data
        except Exception:
            pass
    except ImportError:
        if response_data[:4] == b"\x89PNG" or response_data[:2] == b"\xff\xd8":
            return response_data
    try:
        with zipfile.ZipFile(io.BytesIO(response_data)) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    return zf.read(name)
    except Exception:
        pass
    try:
        data = json.loads(response_data.decode("utf-8"))
        if "image" in data:
            return base64.b64decode(data["image"])
    except Exception:
        pass
    return None


def _has_chinese(text: str) -> bool:
    return bool(re.search("[\\u4e00-\\u9fff]", text))


def _parse_size(size_str: str) -> tuple:
    if not size_str or size_str == "auto":
        return (0, 0)
    s = size_str.strip().lower()
    if s in RATIO_PRESETS:
        return RATIO_PRESETS[s]
    m = re.match("(\\d+)\\s*[xX\\u00d7]\\s*(\\d+)", size_str)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _extract_draw_params(prompt: str) -> tuple:
    """从 prompt 中提取尺寸关键词（竖/方/横），返回 (clean_prompt, width, height)。"""
    width, height = (0, 0)
    _delims = set(" ,，、\t\n")
    for keyword in ("竖", "横", "方"):
        best = -1
        start = 0
        while True:
            idx = prompt.find(keyword, start)
            if idx < 0:
                break
            before_ok = idx == 0 or prompt[idx - 1] in _delims
            after_idx = idx + len(keyword)
            after_ok = after_idx >= len(prompt) or prompt[after_idx] in _delims
            if before_ok and after_ok:
                best = idx
                break
            start = idx + 1
        if best >= 0:
            width, height = _parse_size(keyword)
            prompt = prompt[:best] + prompt[best + len(keyword) :]
            break
    return (prompt.strip(" ,，"), width, height)


def _extract_png_metadata(data: bytes) -> str:
    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(data))
        info = {}
        if hasattr(img, "text"):
            info.update(img.text)
        for k, v in img.info.items():
            if isinstance(v, str) and k not in info:
                info[k] = v
        lines = []
        if "Description" in info:
            lines.append(f"📝 正面提示词:\n{info['Description']}")
        if "Comment" in info:
            try:
                comment = json.loads(info["Comment"])
                if "uc" in comment:
                    lines.append(f"🚫 负面提示词:\n{comment['uc']}")
                params = []
                for key in (
                    "steps",
                    "scale",
                    "sampler",
                    "seed",
                    "width",
                    "height",
                    "noise_schedule",
                    "cfg_rescale",
                    "sm",
                    "sm_dyn",
                    "strength",
                    "noise",
                ):
                    if key in comment:
                        params.append(f"{key}: {comment[key]}")
                if params:
                    lines.append("⚙️ 参数: " + " | ".join(params))
            except (json.JSONDecodeError, TypeError):
                lines.append(f"💬 Comment: {info['Comment'][:500]}")
        if "Software" in info:
            lines.append(f"🔧 Software: {info['Software']}")
        if "Source" in info:
            lines.append(f"🏷️ Model: {info['Source']}")
        if not lines and info:
            for k, v in info.items():
                if isinstance(v, str) and len(v) < 2000:
                    lines.append(f"{k}: {v[:500]}")
        if not lines:
            stealth = _try_stealth_pnginfo(data)
            if stealth:
                lines.append(f"📝 隐写元数据:\n{stealth[:2000]}")
        if not lines:
            return ""
        return "\n\n".join(lines)
    except ImportError:
        return ""
    except Exception:
        return ""


def _try_stealth_pnginfo(data: bytes) -> str:
    try:
        import gzip

        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(data))
        if img.mode != "RGBA":
            return ""
        pixels = np.array(img)
        alpha = pixels[:, :, 3].flatten()
        bits = alpha & 1
        byte_count = len(bits) // 8
        raw_bytes = np.packbits(bits[: byte_count * 8])
        raw = bytes(raw_bytes)
        for magic in (b"stealth_pnginfo", b"stealth_pngcomp"):
            idx = raw.find(magic)
            if idx >= 0:
                payload = raw[idx + len(magic) :]
                if magic == b"stealth_pngcomp":
                    try:
                        payload = gzip.decompress(payload)
                    except Exception:
                        pass
                text = payload.decode("utf-8", errors="ignore")
                end = text.find("\x00")
                return text[:end] if end > 0 else text[:2000]
        return ""
    except Exception:
        return ""
