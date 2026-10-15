"""Persistent preset catalog; bundled presets originate from luoxiQAQ."""

import json
import re
import threading
from pathlib import Path
from typing import Any

from .preset_defaults import seed_presets

PRESET_KIND_LABELS = {"characters": "人物", "styles": "风格"}


class PresetStore:
    def __init__(self, root) -> None:
        self.path = Path(root) / "presets.json"
        seed_presets(self.path, Path(__file__).with_name("presets.json"))
        self._lock = threading.RLock()
        self._sig = self._signature()
        self._data = self._load()

    def _signature(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_if_changed(self) -> None:
        signature = self._signature()
        if signature == self._sig:
            return
        self._data = self._load()
        self._sig = signature

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            data = {}
        return {
            "characters": self._clean_map(data.get("characters", {})),
            "styles": self._clean_map(data.get("styles", {})),
        }

    @staticmethod
    def _clean_map(value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        return {
            str(name).strip(): str(prompt).strip()
            for name, prompt in value.items()
            if str(name).strip() and str(prompt).strip()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)
        self._sig = self._signature()

    @staticmethod
    def _check_kind(kind: str) -> str:
        if kind not in PRESET_KIND_LABELS:
            raise ValueError("预设类型必须是 characters 或 styles")
        return kind

    @staticmethod
    def _check_values(name: str, prompt: str) -> tuple[str, str]:
        name = name.strip()
        prompt = prompt.strip()
        if not name or not prompt:
            raise ValueError("预设名称和提示词都不能为空")
        if re.search("[\\s@#=<>/'\\\"]", name):
            raise ValueError("预设名称不能包含空格、@、#、=、尖括号、斜杠或引号")
        if len(name) > 40:
            raise ValueError("预设名称不能超过 40 个字符")
        if len(prompt) > 4000:
            raise ValueError("预设提示词不能超过 4000 个字符")
        return (name, prompt)

    def all(self) -> dict:
        with self._lock:
            self._reload_if_changed()
            return {kind: dict(values) for kind, values in self._data.items()}

    def get(self, kind: str, name: str) -> str | None:
        kind = self._check_kind(kind)
        with self._lock:
            self._reload_if_changed()
            return self._data[kind].get(name.strip())

    def set(self, kind: str, name: str, prompt: str) -> None:
        kind = self._check_kind(kind)
        name, prompt = self._check_values(name, prompt)
        with self._lock:
            self._reload_if_changed()
            self._data[kind][name] = prompt
            self._save()

    def set_many(self, kind: str, items: dict) -> int:
        kind = self._check_kind(kind)
        count = 0
        with self._lock:
            self._reload_if_changed()
            for name, prompt in items.items():
                try:
                    clean_name, clean_prompt = self._check_values(
                        str(name), str(prompt)
                    )
                except ValueError:
                    continue
                self._data[kind][clean_name] = clean_prompt
                count += 1
            if count:
                self._save()
        return count

    def delete(self, kind: str, name: str) -> bool:
        kind = self._check_kind(kind)
        with self._lock:
            self._reload_if_changed()
            if name.strip() not in self._data[kind]:
                return False
            del self._data[kind][name.strip()]
            self._save()
            return True
