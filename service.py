"""NovelAI service adapted for AstrBot; no framework global configuration."""

import asyncio
import random
import re
from contextvars import ContextVar
from types import SimpleNamespace

import httpx
from astrbot.api import logger

from .generation import CURRENT_JOB, Rejected
from .image_utils import (
    NAI_API_BASE,
    NSFW_NEGATIVE_TAGS,
    TRANSLATE_SYSTEM_PROMPT,
    _create_v4_prompt,
    _extract_image,
    _has_chinese,
    _is_v4_style,
)
from .presets import PRESET_KIND_LABELS
from .request_retry import generate_request
from .settings import DEFAULTS


class NovelAIService:
    def __init__(self, config, presets, context):
        self.config, self.presets, self.context = config, presets, context
        self.token_index = 0
        self.snapshot = ContextVar("novelai_astrbot_config_snapshot", default=None)

    @property
    def cfg(self):
        return self.snapshot.get() or SimpleNamespace(**(DEFAULTS | dict(self.config)))

    async def _translate_prompt(self, prompt):
        provider_id = self.cfg.TRANSLATE_PROVIDER_ID
        if not provider_id or not _has_chinese(prompt):
            return prompt
        parts = [p.strip() for p in re.split(r"[,，]", prompt) if p.strip()]
        system = self._build_translate_system_prompt(self._find_preset_hints(prompt))
        for i, part in enumerate(parts):
            if not _has_chinese(part):
                continue
            try:
                options = (
                    {"model": self.cfg.TRANSLATE_MODEL_NAME}
                    if self.cfg.TRANSLATE_MODEL_NAME
                    else {}
                )
                response = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt="Translate only this segment to comma-separated danbooru tags; do not add quality tags: "
                        + part,
                        system_prompt=system,
                        max_tokens=500,
                        temperature=0.3,
                        **options,
                    ),
                    timeout=45,
                )
                translated = response.completion_text.strip()
                if not translated:
                    raise ValueError("Empty translation")
                parts[i] = translated
            except Exception:
                raise Rejected(
                    "提示词翻译失败，请检查翻译模型配置，或改用英文标签。未提交生图。"
                ) from None
        return ", ".join(parts)

    def _expand_preset_prompt(self, prompt: str) -> str:
        character_names: list[str] = []
        style_names: list[str] = []

        def collect(kind: str, names: list[str], name: str) -> str:
            name = name.strip()
            if name and name not in names:
                names.append(name)
            return ""

        prompt = re.sub(
            "人物\\s*=\\s*([^\\s#@=，,]+)",
            lambda match: collect("characters", character_names, match.group(1)),
            prompt,
        )
        prompt = re.sub(
            "风格\\s*=\\s*([^\\s#@=，,]+)",
            lambda match: collect("styles", style_names, match.group(1)),
            prompt,
        )
        prompt = re.sub(
            "@([^\\s#@=，,]+)",
            lambda match: collect("characters", character_names, match.group(1)),
            prompt,
        )
        prompt = re.sub(
            "#([^\\s@=，,]+)",
            lambda match: collect("styles", style_names, match.group(1)),
            prompt,
        )
        all_presets = self.presets.all()
        _delimiters = set(" ,，、。！？\t\n")
        for kind, names_list in (
            ("styles", style_names),
            ("characters", character_names),
        ):
            known = sorted(all_presets[kind].keys(), key=len, reverse=True)
            for name in known:
                if name in names_list:
                    continue
                idx = prompt.find(name)
                if idx < 0:
                    continue
                before_ok = idx == 0 or prompt[idx - 1] in _delimiters
                after_idx = idx + len(name)
                after_ok = after_idx >= len(prompt) or prompt[after_idx] in _delimiters
                if before_ok and after_ok:
                    names_list.append(name)
                    prompt = prompt[:idx] + prompt[after_idx:]
        missing = []
        expanded = []
        for kind, names in (("characters", character_names), ("styles", style_names)):
            for name in names:
                value = self.presets.get(kind, name)
                if value is None:
                    missing.append(f"{PRESET_KIND_LABELS[kind]}预设「{name}」")
                else:
                    expanded.append(value)
        if missing:
            raise ValueError("未找到" + "、".join(missing))
        prompt = re.sub("\\s+", " ", prompt).strip(" ,，")
        return ", ".join(expanded + ([prompt] if prompt else []))

    def _get_tokens(self) -> list:
        raw = self.cfg.API_TOKEN or ""
        return [t.strip() for t in re.split("[,;\\n]+", raw) if t.strip()]

    def _get_current_token(self) -> str:
        tokens = self._get_tokens()
        if not tokens:
            raise ValueError("NovelAI Token 未配置，请在插件设置中填写 API_TOKEN。")
        self.token_index = self.token_index % len(tokens)
        return tokens[self.token_index]

    def _rotate_token(self, reason: str) -> bool:
        tokens = self._get_tokens()
        if len(tokens) <= 1:
            return False
        self.token_index = (self.token_index + 1) % len(tokens)
        logger.warning(
            f"NovelAI token rotated after {reason}; index {self.token_index + 1}/{len(tokens)}"
        )
        return True

    def _get_api_url(self, endpoint: str = "generate-image") -> str:
        if self.cfg.API_PROXY:
            return f"{self.cfg.API_PROXY.rstrip('/')}/ai/{endpoint}"
        return f"{NAI_API_BASE}/ai/{endpoint}"

    def _parse_resolution(self) -> tuple:
        res = self.cfg.DEFAULT_RESOLUTION or "832x1216"
        m = re.match("(\\d+)[xX×](\\d+)", res)
        return (int(m.group(1)), int(m.group(2))) if m else (832, 1216)

    def _build_client_config(self) -> dict:
        cfg: dict = {
            "trust_env": False,
            "timeout": httpx.Timeout(90.0, connect=30.0),
            "limits": httpx.Limits(max_keepalive_connections=2, max_connections=3),
        }
        if self.cfg.HTTP_PROXY:
            cfg["proxy"] = self.cfg.HTTP_PROXY
        return cfg

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_current_token()}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Connection": "close",
        }

    def _get_negative_prompt(self, extra: str = "") -> str:
        parts = []
        if self.cfg.NEGATIVE_PROMPT:
            parts.append(self.cfg.NEGATIVE_PROMPT)
        if not self.cfg.ENABLE_R18:
            parts.append(NSFW_NEGATIVE_TAGS)
        if extra:
            parts.append(extra)
        return ", ".join(parts)

    def _build_parameters(
        self, prompt, width, height, steps, scale, sampler, model, negative_prompt=""
    ):
        neg = self._get_negative_prompt(negative_prompt)
        params = {
            "params_version": 3,
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "ucPreset": 0,
            "qualityToggle": True,
            "dynamic_thresholding": False,
            "controlnet_strength": 1,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": self.cfg.CFG_RESCALE,
            "noise_schedule": "karras",
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": None,
            "use_coords": False,
            "characterPrompts": [],
            "negative_prompt": neg,
            "seed": random.randint(0, 4294967295),
        }
        if _is_v4_style(model):
            params["v4_prompt"] = _create_v4_prompt(prompt)
            params["v4_negative_prompt"] = {
                "caption": {
                    "base_caption": neg,
                    "char_captions": [],
                    "scenery_captions": [],
                }
            }
        return params

    def _find_preset_hints(self, text):
        """Scan text for known character/style preset names, return [(name, tags)]."""
        all_presets = self.presets.all()
        hints = []
        for kind in ("characters", "styles"):
            known = sorted(all_presets[kind].keys(), key=len, reverse=True)
            for name in known:
                if name in text:
                    hints.append((name, all_presets[kind][name]))
        return hints

    def _build_translate_system_prompt(self, preset_hints=None):
        prompt = TRANSLATE_SYSTEM_PROMPT
        if preset_hints:
            hint_lines = "\n".join(
                (f"  {name} -> {tags}" for name, tags in preset_hints)
            )
            prompt += (
                "\n\nIMPORTANT - Character/style presets detected in user input. You MUST use the EXACT danbooru tags listed below for these names (extract only the character-specific tags like character_name_(series), do NOT duplicate quality tags, do NOT use solo when multiple characters, add correct count tags like 2girls):\n"
                + hint_lines
            )
        return prompt

    async def _call_txt2img(
        self,
        prompt,
        width=0,
        height=0,
        steps=0,
        scale=0,
        sampler="",
        model="",
        negative_prompt="",
    ):
        if CURRENT_JOB.get() is None:
            raise Rejected("生图必须通过统一队列提交。")
        model = model or self.cfg.DEFAULT_MODEL
        dw, dh = self._parse_resolution()
        width, height = (width or dw, height or dh)
        steps = steps or self.cfg.DEFAULT_STEPS
        scale = scale or self.cfg.DEFAULT_SCALE
        sampler = sampler or self.cfg.DEFAULT_SAMPLER
        raw_prompt = prompt
        raw_negative_prompt = negative_prompt
        prompt = await self._translate_prompt(self._expand_preset_prompt(prompt))
        if negative_prompt:
            negative_prompt = await self._translate_prompt(negative_prompt)
        params = self._build_parameters(
            prompt, width, height, steps, scale, sampler, model, negative_prompt
        )
        info = {
            "raw_prompt": raw_prompt,
            "prompt": prompt,
            "negative_prompt": self._get_negative_prompt(negative_prompt),
            "raw_negative_prompt": raw_negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "scale": scale,
            "sampler": sampler,
            "model": model,
            "seed": params.get("seed"),
            "image_type": "txt2img",
        }
        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": params,
        }
        api_url = self._get_api_url("generate-image")
        async with httpx.AsyncClient(**self._build_client_config()) as client:
            response = await generate_request(
                client,
                api_url,
                payload,
                self._build_headers,
                retries=self.cfg.RATE_LIMIT_RETRIES,
                wait=self.cfg.RATE_LIMIT_WAIT,
                rotate=self._rotate_token,
                log=logger,
            )
            img = _extract_image(response.content)
            if img is None:
                raise Rejected("NovelAI 返回的数据没有有效图片，未扣除额度。")
            return (img, info)

    async def _call_img2img(
        self,
        prompt,
        image_b64,
        width=0,
        height=0,
        steps=0,
        scale=0,
        strength=0,
        noise=-1,
        model="",
        negative_prompt="",
    ):
        if CURRENT_JOB.get() is None:
            raise Rejected("生图必须通过统一队列提交。")
        model = model or self.cfg.DEFAULT_MODEL
        dw, dh = self._parse_resolution()
        width, height = (width or dw, height or dh)
        steps = steps or self.cfg.DEFAULT_STEPS
        scale = scale or self.cfg.DEFAULT_SCALE
        strength = strength or self.cfg.IMG2IMG_STRENGTH
        noise = noise if noise >= 0 else self.cfg.IMG2IMG_NOISE
        raw_prompt = prompt
        raw_negative_prompt = negative_prompt
        prompt = await self._translate_prompt(self._expand_preset_prompt(prompt))
        if negative_prompt:
            negative_prompt = await self._translate_prompt(negative_prompt)
        params = self._build_parameters(
            prompt,
            width,
            height,
            steps,
            scale,
            self.cfg.DEFAULT_SAMPLER,
            model,
            negative_prompt,
        )
        params["image"] = image_b64
        params["strength"] = strength
        params["noise"] = noise
        info = {
            "raw_prompt": raw_prompt,
            "prompt": prompt,
            "negative_prompt": self._get_negative_prompt(negative_prompt),
            "raw_negative_prompt": raw_negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "scale": scale,
            "sampler": self.cfg.DEFAULT_SAMPLER,
            "model": model,
            "seed": params.get("seed"),
            "image_type": "img2img",
            "strength": strength,
            "noise": noise,
        }
        payload = {
            "input": prompt,
            "model": model,
            "action": "img2img",
            "parameters": params,
        }
        api_url = self._get_api_url("generate-image")
        async with httpx.AsyncClient(**self._build_client_config()) as client:
            response = await generate_request(
                client,
                api_url,
                payload,
                self._build_headers,
                retries=self.cfg.RATE_LIMIT_RETRIES,
                wait=self.cfg.RATE_LIMIT_WAIT,
                rotate=self._rotate_token,
                log=logger,
            )
            img = _extract_image(response.content)
            if img is None:
                raise Rejected("NovelAI 返回的数据没有有效图片，未扣除额度。")
            return (img, info)
