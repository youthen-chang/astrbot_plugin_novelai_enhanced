"""NovelAI Enhanced for AstrBot, based on luoxiQAQ's MIT licensed plugin."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import secrets
import time
import uuid
from datetime import datetime
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response, request
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

from .adapter import match_command, read_reference, source_from_event, validate_triggers
from .classification import classify
from .gallery_outbox import Outbox
from .gallery_upload import build_name, upload_to_gallery
from .generation import CURRENT_JOB, SAVED, SHANGHAI, Duplicate, Engine, Rejected
from .image_utils import (
    _extract_draw_params,
    _extract_png_metadata,
    _parse_size,
    _strip_png_metadata,
)
from .presets import PRESET_KIND_LABELS, PresetStore
from .prompts import split_extra_negative
from .service import NovelAIService

PLUGIN_NAME = "astrbot_plugin_novelai_enhanced"


class NovelAIEnhanced(Star):
    author = "Yasumi"
    name = PLUGIN_NAME

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.root = StarTools.get_data_dir(PLUGIN_NAME)
        self.presets = PresetStore(self.root)
        self.service = NovelAIService(config, self.presets, context)
        self.engine = Engine(self.root / "generation", lambda: self.service.cfg)
        self.outbox = Outbox(
            self.root / "gallery_outbox",
            upload_to_gallery,
            lambda: dict(self.config),
            lambda meta: classify(meta, self.presets.all())["tags"],
        )
        self.tasks = set()
        self.recent = {}
        self.last_path = self.root / "last_draw.json"
        try:
            self.last = json.loads(self.last_path.read_text(encoding="utf-8"))
            if not isinstance(self.last, dict):
                self.last = {}
        except (OSError, ValueError):
            self.last = {}
        self.closed = False
        for route, handler, methods in [
            ("presets", self.page_presets, ["GET"]),
            ("presets/save", self.page_save, ["POST"]),
            ("presets/delete", self.page_delete, ["POST"]),
            ("presets/import", self.page_import, ["POST"]),
            ("status", self.page_status, ["GET"]),
        ]:
            context.register_web_api(
                f"/{PLUGIN_NAME}/{route}", handler, methods, "NovelAI Enhanced"
            )

    async def initialize(self):
        validate_triggers(
            self.service.cfg.DRAW_TRIGGER, self.service.cfg.REDRAW_TRIGGER
        )
        self.outbox.start()
        logger.info(
            "NovelAI Enhanced initialized; quota and whitelist switches default to disabled"
        )

    async def terminate(self):
        self.closed = True
        await self.engine.stop()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.outbox.stop()
        self.recent.clear()
        # Remove only bound APIs owned by this instance on reload/uninstall.
        self.context.registered_web_apis[:] = [
            api
            for api in self.context.registered_web_apis
            if getattr(api[1], "__self__", None) is not self
        ]

    @filter.event_message_type(filter.EventMessageType.ALL, priority=80)
    async def on_message(self, event: AstrMessageEvent):
        if self.closed:
            return
        cfg = self.service.cfg
        if (
            cfg.ONLY_INSTANCE_ID
            and str(event.get_self_id()) != cfg.ONLY_INSTANCE_ID.strip()
        ):
            return
        try:
            source = source_from_event(event)
        except Rejected:
            return
        images = [part for part in event.get_messages() if isinstance(part, Image)]
        if images:
            # Cache component references, not downloaded image bytes, within the same conversation.
            self.recent[source.chat] = (time.time(), images[-1])
            if len(self.recent) > 200:
                self.recent.pop(next(iter(self.recent)))
        try:
            # AstrBot strips wake prefixes from message_str; original Plain segments preserve '/'.
            original_text = "".join(
                p.text for p in event.get_messages() if isinstance(p, Plain)
            )
            command = match_command(original_text or event.get_message_str(), cfg)
        except Rejected:
            return
        if not command:
            return
        event.stop_event()
        kind, argument = command
        if kind in ("draw", "redraw", "来点", "来点nai", "laidian"):
            try:
                ticket = self.submit(event, kind, argument)
            except Duplicate:
                return
            except (Rejected, ValueError) as exc:
                await event.send(MessageChain().message(str(exc)))
                return
            task = asyncio.create_task(self.deliver(event, ticket))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        else:
            try:
                text = await self.command(event, kind, argument)
            except (Rejected, ValueError) as exc:
                text = str(exc)
            await event.send(MessageChain().message(text))

    def submit(
        self,
        event,
        kind,
        prompt,
        size="",
        negative_prompt="",
        model="",
        strength=0,
        noise=-1,
    ):
        """Validate and reserve quota before expensive translation or provider requests.

        Args:
            event: Trusted AstrBot event shared by command and LLM paths.
            kind: Generation operation.
            prompt: User's input including optional preset and negative syntax.
            size: Optional supported resolution.
            negative_prompt: Additional negative tags provided by an LLM tool.
            model: Optional supported model.
            strength: Optional image-to-image strength.
            noise: Optional image-to-image noise.

        Returns:
            A queued ticket for exactly one generation.

        Raises:
            Rejected: Access, configuration, validation or quota checks fail.
        """
        if self.closed:
            raise Rejected("插件正在重新加载，请稍后再试。")
        cfg = self.service.cfg
        if (
            cfg.ONLY_INSTANCE_ID
            and str(event.get_self_id()) != cfg.ONLY_INSTANCE_ID.strip()
        ):
            raise Rejected("该功能仅限指定机器人实例使用。")
        source = source_from_event(event)
        self.engine.policy.check_access(source)
        self.service._get_current_token()
        positive, inline_negative, has_inline = split_extra_negative(prompt)
        if has_inline:
            negative_prompt = ", ".join(
                v for v in (negative_prompt.strip(), inline_negative) if v
            )
        clean, width, height = _extract_draw_params(positive)
        completion = "NovelAI 画图完成"
        if kind == "redraw":
            last = dict(self.last.get(source.chat) or {})
            if not last:
                raise Rejected("还没有画过图，请先画一张。")
            positive = (positive.strip() + "，" if positive.strip() else "") + last[
                "prompt"
            ]
            clean, width, height = _extract_draw_params(positive)
            width, height = (
                width or last.get("width", 0),
                height or last.get("height", 0),
            )
            if not has_inline and not negative_prompt:
                negative_prompt = last.get("extra_negative", "")
            completion = "NovelAI 重画完成"
        if not clean.strip():
            raise Rejected("请在画图触发词后提供描述。")
        if size and size != "auto":
            width, height = _parse_size(size)
            if not width or not height:
                raise Rejected("尺寸只支持竖、方、横或 832x1216、1216x832、1024x1024。")
        if width and (width, height) not in ((832, 1216), (1216, 832), (1024, 1024)):
            raise Rejected("尺寸只支持 832x1216、1216x832、1024x1024。")
        if model and model not in (
            "nai-diffusion-5-full",
            "nai-diffusion-5-curated",
            "nai-diffusion-4-5-full",
            "nai-diffusion-4-5-curated",
        ):
            raise Rejected("请使用配置页提供的 NovelAI 模型。")
        if strength and not 0.1 <= strength <= 0.9:
            raise Rejected("图生图强度必须在 0.1 到 0.9 之间。")
        if noise != -1 and not 0 <= noise <= 1:
            raise Rejected("图生图噪声必须在 0 到 1 之间。")
        swap = kind in ("来点", "来点nai", "laidian")
        if swap:
            catalog = self.presets.all()["characters"]
            names, remaining = [], clean
            for name in sorted(catalog, key=len, reverse=True):
                match = re.search(
                    r"(?:^|[\s,，、])@?" + re.escape(name) + r"(?=$|[\s,，、])",
                    remaining,
                )
                if match:
                    names.append(name)
                    remaining = (
                        remaining[: match.start()] + " " + remaining[match.end() :]
                    )
            if not names:
                raise Rejected(
                    "未找到人物预设，请使用 /人物列表 查询，或在预设页面添加。"
                )
            completion = "来点「" + "、".join(names) + "」完成"
        # Reject unknown explicit presets before reserving quota.
        try:
            self.service._expand_preset_prompt(clean)
        except ValueError as exc:
            raise Rejected(str(exc)) from None
        if not (
            1 <= cfg.DEFAULT_STEPS <= 50
            and 1 <= cfg.DEFAULT_SCALE <= 20
            and 0 <= cfg.CFG_RESCALE <= 1
        ):
            raise Rejected("请检查采样步数、CFG 权重和 CFG Rescale 配置范围。")
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        if not message_id:
            raise Rejected("消息缺少平台 ID，无法保证去重，未提交生图。")

        async def operation():
            reference = (
                None if kind == "redraw" else await self.reference(event, recent=swap)
            )
            if (swap or kind == "img2img") and reference is None:
                raise Rejected("请引用一张图片，或把图片与画图描述放在同一条消息中。")
            if reference is not None:
                data, info = await self.service._call_img2img(
                    clean,
                    base64.b64encode(reference).decode("ascii"),
                    width,
                    height,
                    strength=strength or (cfg.CHAR_SWAP_STRENGTH if swap else 0),
                    noise=noise
                    if noise != -1
                    else (cfg.CHAR_SWAP_NOISE if swap else -1),
                    model=model,
                    negative_prompt=negative_prompt,
                )
            else:
                data, info = await self.service._call_txt2img(
                    clean, width, height, model=model, negative_prompt=negative_prompt
                )
            # Save a PNG with searchable metadata; chat gets a separate metadata-free copy.
            info["source"] = "AstrBot"
            info["saved_at"] = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S")
            with PILImage.open(io.BytesIO(data)) as image:
                if image.width * image.height > 32_000_000:
                    raise Rejected("返回图片尺寸异常，无法保存，未扣除额度。")
                image.load()
                metadata = PngInfo()
                metadata.add_text("Description", info["prompt"])
                metadata.add_text(
                    "Comment",
                    json.dumps(
                        {
                            "uc": info["negative_prompt"],
                            **{
                                k: info[k]
                                for k in (
                                    "steps",
                                    "scale",
                                    "sampler",
                                    "seed",
                                    "width",
                                    "height",
                                    "model",
                                )
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
                metadata.add_text("Software", "NovelAI / AstrBot Enhanced")
                output = io.BytesIO()
                image.save(output, format="PNG", pnginfo=metadata)
                data = output.getvalue()
            directory = self.root / "generated"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("nai_" + uuid.uuid4().hex + ".png")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
            SAVED.set(True)
            # Quota charge is durable before gallery/chat delivery, even across a process crash.
            self.engine.finish(CURRENT_JOB.get(), "succeeded", saved=True)
            self.last[source.chat] = {
                "prompt": positive,
                "width": info["width"],
                "height": info["height"],
                "image_path": path.name,
                "extra_negative": negative_prompt,
            }
            try:
                last_tmp = self.last_path.with_suffix(".tmp")
                last_tmp.write_text(
                    json.dumps(self.last, ensure_ascii=False), encoding="utf-8"
                )
                last_tmp.replace(self.last_path)
            except OSError:
                logger.warning("NovelAI redraw state could not be persisted")
            try:
                if (
                    self.service.cfg.GALLERY_UPLOAD_URL
                    and self.service.cfg.GALLERY_UPLOAD_KEY
                ):
                    self.outbox.enqueue(data, info, build_name())
                    self.outbox.start()
            except Exception:
                logger.warning(
                    "NovelAI gallery enqueue failed; saved PNG remains available locally"
                )
            # Keep a short recent archive, preserving files referenced by redraw state.
            protected = {
                v.get("image_path") for v in self.last.values() if isinstance(v, dict)
            }
            for previous in directory.glob("nai_*.png"):
                try:
                    if (
                        previous.name not in protected
                        and time.time() - previous.stat().st_mtime > 86400
                    ):
                        previous.unlink(missing_ok=True)
                except OSError:
                    logger.warning("NovelAI old image cleanup failed")
            return path, completion

        async def snapshotted():
            token = self.service.snapshot.set(self.service.cfg)
            try:
                return await operation()
            finally:
                self.service.snapshot.reset(token)

        return self.engine.submit(source, "message:" + message_id, snapshotted)

    async def deliver(self, event, ticket, send=True):
        """Send one combined completion message without regenerating on delivery failure.

        Args:
            event: Original request's platform event.
            ticket: Queued generation ticket.
            send: Whether to send completion to the chat.

        Returns:
            Saved image path or a safe failure message.
        """
        if send and self.service.cfg.GENERATING_TEXT:
            try:
                await event.send(
                    MessageChain().message(self.service.cfg.GENERATING_TEXT)
                )
            except Exception:
                logger.warning("NovelAI progress message delivery failed")
        try:
            path, text = await asyncio.shield(ticket.future)
        except Rejected as exc:
            if send and not self.closed:
                try:
                    await event.send(MessageChain().message(str(exc)))
                except Exception:
                    logger.warning("NovelAI error message delivery failed")
            return str(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("NovelAI generation failed (%s)", type(exc).__name__)
            if send and not self.closed:
                try:
                    await event.send(
                        MessageChain().message(
                            "生图失败，请检查插件配置和日志。未成功保存的图片不扣插件额度。"
                        )
                    )
                except Exception:
                    logger.warning("NovelAI error message delivery failed")
            return "生图失败，请勿自动重试。"
        if send:
            try:
                await event.send(
                    MessageChain(
                        chain=[
                            Plain(text),
                            Image.fromBytes(_strip_png_metadata(path.read_bytes())),
                        ]
                    )
                )
            except Exception:
                logger.warning(
                    "NovelAI image was saved but chat delivery failed; generation will not be repeated"
                )
                try:
                    await event.send(
                        MessageChain().message(
                            "图片已生成并保存，但发送失败；不会重复生图或重复扣额度。可在画廊查看或联系管理员。"
                        )
                    )
                except Exception:
                    pass
                return "图片已保存但发送失败，请勿再次生成。"
        return str(path)

    async def reference(self, event, recent=False, raw=False):
        """Resolve attached, replied or cached images without cross-session fallback.

        Args:
            event: Original request event.
            recent: Allow the most recent image from this same conversation.
            raw: Preserve original metadata for parameter inspection.

        Returns:
            Image bytes, or None if there is no image reference.
        """
        components = event.get_messages()
        images = [p for p in components if isinstance(p, Image)]
        for reply in (p for p in components if isinstance(p, Reply)):
            images.extend(p for p in (reply.chain or []) if isinstance(p, Image))
            if not images and hasattr(event, "bot"):
                try:
                    response = await event.bot.get_msg(message_id=int(reply.id))
                    for segment in response.get("message", []):
                        if segment.get("type") == "image":
                            item = segment["data"]
                            images.append(
                                Image(
                                    file=item.get("file", ""), url=item.get("url", "")
                                )
                            )
                except Exception:
                    raise Rejected(
                        "无法读取被引用的图片，请把图片和指令一起重新发送。"
                    ) from None
            if not images:
                raise Rejected("被引用的消息中没有可读取的图片。")
        if not images and recent:
            source = source_from_event(event)
            entry = self.recent.get(source.chat)
            if entry and time.time() - entry[0] < 86400:
                images = [entry[1]]
        if not images:
            return None
        return await read_reference(images[0], normalize=not raw)

    @filter.llm_tool(name="novelai_generate")
    async def novelai_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "auto",
        negative_prompt: str = "",
        model: str = "",
        send_to_chat: bool = True,
    ):
        """Generate one NovelAI image only when the user explicitly requests drawing. Do not call twice for the same message.

        Args:
            prompt(string): Positive description, supports @character, #style and -n:extra negatives.
            size(string): auto, portrait, landscape or square.
            negative_prompt(string): Extra negative tags appended to global defaults.
            model(string): Optional model ID; leave empty for plugin defaults.
            send_to_chat(boolean): Send image and completion text to this chat, normally true.
        """
        return await self.agent_submit(
            event, "draw", prompt, size, negative_prompt, model, send_to_chat
        )

    @filter.llm_tool(name="novelai_img2img")
    async def novelai_img2img(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "auto",
        negative_prompt: str = "",
        model: str = "",
        strength: float = 0.5,
        noise: float = 0.2,
        send_to_chat: bool = True,
    ):
        """Transform an attached or replied image using NovelAI only upon an explicit user request. Never repeat a submitted generation.

        Args:
            prompt(string): Requested description with optional preset references.
            size(string): auto, portrait, landscape or square.
            negative_prompt(string): Additional negative tags.
            model(string): Optional model ID, empty for default.
            strength(number): Transformation strength between 0.1 and 0.9.
            noise(number): Image-to-image noise between 0 and 1.
            send_to_chat(boolean): Send the resulting image to the chat, normally true.
        """
        return await self.agent_submit(
            event,
            "img2img",
            prompt,
            size,
            negative_prompt,
            model,
            send_to_chat,
            strength,
            noise,
        )

    async def agent_submit(
        self,
        event,
        kind,
        prompt,
        size,
        negative_prompt,
        model,
        send,
        strength=0,
        noise=-1,
    ):
        try:
            ticket = self.submit(
                event, kind, prompt, size, negative_prompt, model, strength, noise
            )
        except (Rejected, ValueError) as exc:
            return str(exc) + " 不要重复调用生图工具。"
        if not send:
            return await self.deliver(event, ticket, send=False)
        # Return acceptance immediately so the framework's tool timeout does not cancel queued requests.
        task = asyncio.create_task(self.deliver(event, ticket))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return "生图任务已受理，插件会自行发送图片。不要再次调用生图工具，不要声称图片已经完成。"

    async def command(self, event, kind, argument):
        source = source_from_event(event)
        if kind == "额度":
            return self.engine.quota_text(source)
        if kind == "绘图帮助":
            cfg = self.service.cfg
            return (
                f"画图：{cfg.DRAW_TRIGGER} 描述 [竖/横/方] [-n:额外负面词。]\n"
                f"重画：{cfg.REDRAW_TRIGGER} [追加描述]\n"
                "/额度、/来点 人物名、/看参数、/人物列表、/风格列表\n"
                "图片+画图指令，或回复图片+画图指令，可使用图生图。"
            )
        if kind in ("人物列表", "风格列表"):
            category = "characters" if kind == "人物列表" else "styles"
            values = list(self.presets.all()[category])
            try:
                page = max(1, int(argument or 1))
            except ValueError:
                raise Rejected("用法：/" + kind + " [页码]") from None
            pages = max(1, (len(values) + 49) // 50)
            page = min(page, pages)
            return f"{kind}（第 {page}/{pages} 页，共 {len(values)} 个）\n" + "、".join(
                values[(page - 1) * 50 : page * 50]
            )
        if kind in ("看参数", "反推", "查看参数", "naimeta"):
            data = await self.reference(event, raw=True)
            if data is None:
                last = self.last.get(source.chat, {})
                name = str(last.get("image_path", ""))
                path = self.root / "generated" / name
                if name and Path(name).name == name and path.is_file():
                    data = path.read_bytes()
                else:
                    data = await self.reference(event, recent=True, raw=True)
            return (
                (_extract_png_metadata(data) or "未能从图片中提取到 NovelAI 参数。")
                if data
                else "请引用图片，或在画图后使用 /看参数。"
            )
        if not self.engine.policy.is_admin(source):
            raise Rejected("仅插件管理员可使用此管理指令，请在后台配置 PLUGIN_ADMINS。")
        if kind in ("添加人物", "添加风格", "删除人物", "删除风格"):
            category = "characters" if "人物" in kind else "styles"
            if kind.startswith("添加"):
                parts = argument.split(None, 1)
                if len(parts) != 2:
                    raise Rejected(f"用法：/{kind} 名称 提示词")
                self.presets.set(category, *parts)
                return "预设已保存。"
            return (
                "预设已删除。"
                if self.presets.delete(category, argument)
                else "未找到此预设。"
            )
        if kind in (
            "拉黑",
            "画图拉黑",
            "解除拉黑",
            "移出黑名单",
            "黑名单",
            "黑名单列表",
        ):
            entries = {}
            for line in self.config.get("DRAW_BLACKLIST", "").splitlines():
                parts = line.split("#", 1)[0].strip().split(None, 1)
                if parts:
                    entries[parts[0]] = parts[1] if len(parts) > 1 else ""
            if kind in ("黑名单", "黑名单列表"):
                return (
                    "画图黑名单：\n"
                    + "\n".join(k + (" " + v if v else "") for k, v in entries.items())
                    if entries
                    else "画图黑名单为空。"
                )
            parts = argument.split(None, 1)
            number = parts[0] if parts else source.channel
            if not number.isdigit():
                raise Rejected("请提供群号或用户 QQ 号。")
            if kind in ("拉黑", "画图拉黑"):
                entries[number] = parts[1] if len(parts) > 1 else ""
            else:
                entries.pop(number, None)
            self.config["DRAW_BLACKLIST"] = "\n".join(
                k + (" " + v if v else "") for k, v in entries.items()
            )
            self.config.save_config()
            return "画图黑名单已更新。"
        if kind in (
            "生成画图密钥",
            "生成密钥",
            "增加画图次数",
            "密钥加额",
            "画图密钥列表",
            "删除画图密钥",
        ):
            return self.manage_gallery_keys(kind, argument)
        return "未知指令。"

    def manage_gallery_keys(self, kind, argument):
        path = self.root / "gallery_draw_keys.json"
        try:
            keys = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            keys = []
        if kind == "画图密钥列表":
            return (
                "\n".join(
                    f"{k['key']}：{max(0, k['limit'] - k['used'])} / {k['limit']}"
                    for k in keys
                )
                or "画图密钥为空。"
            )
        parts = argument.split()
        if kind in ("生成画图密钥", "生成密钥"):
            quota = int(parts[0]) if parts else 100
            if quota < 1:
                raise Rejected("用法：/生成画图密钥 [NAI5次数>=1]")
            created = {
                "kid": secrets.token_hex(4),
                "key": "lx" + secrets.token_hex(6),
                "limit": quota,
                "used": 0,
                "created": datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M"),
            }
            keys.append(created)
            text = f"画图密钥：{created['key']}\nNAI5 额度：{quota}\n仅供兼容原画廊的共享密钥文件使用；不属于本站上传密钥。"
        elif kind in ("密钥加额", "增加画图次数"):
            if len(parts) != 2 or not parts[1].isdigit():
                raise Rejected("用法：/密钥加额 密钥 数量")
            match = next((k for k in keys if k["key"] == parts[0]), None)
            if not match:
                raise Rejected("未找到密钥。")
            match["limit"] += int(parts[1])
            text = "密钥额度已增加。"
        else:
            keys = [k for k in keys if k["key"] != argument.strip()]
            text = "密钥已删除。"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return text

    async def page_presets(self):
        return json_response(self.presets.all())

    async def page_save(self):
        try:
            data = await request.json(default={})
            self.presets.set(data["kind"], data["name"], data["prompt"])
            return json_response({"ok": True})
        except (ValueError, KeyError, TypeError, AttributeError):
            return error_response("预设格式不正确，请检查名称、分类和提示词。")

    async def page_delete(self):
        try:
            data = await request.json(default={})
            return json_response(
                {"ok": self.presets.delete(data["kind"], data["name"])}
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            return error_response("预设格式不正确。")

    async def page_import(self):
        try:
            if len(await request.body()) > 4 * 1024 * 1024:
                return error_response("预设文件不能超过 4 MB。")
            data = await request.json(default={})
            if not isinstance(data, dict) or not any(
                k in data for k in PRESET_KIND_LABELS
            ):
                return error_response("需要含 characters 或 styles 对象的 JSON。")
            if any(
                k in data and not isinstance(data[k], dict) for k in PRESET_KIND_LABELS
            ):
                return error_response("characters 和 styles 必须是对象。")
            counts = {
                k: self.presets.set_many(k, data.get(k, {})) for k in PRESET_KIND_LABELS
            }
            return json_response({"ok": True, "imported": counts})
        except (ValueError, TypeError):
            return error_response("预设文件格式不正确。")

    async def page_status(self):
        return json_response(
            {
                "gallery": self.outbox.snapshot(),
                "pending": len(self.engine.pending),
                "token_configured": bool(self.service.cfg.API_TOKEN),
                "quota_enabled": self.service.cfg.DAILY_QUOTA_ENABLED,
            }
        )
