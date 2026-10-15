"""Install bundled defaults once without overwriting user-managed presets."""

import json
from pathlib import Path


def seed_presets(destination: Path, bundled: Path) -> bool:
    if destination.exists():
        return False
    content = bundled.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict) or any(
        not isinstance(data.get(kind), dict) for kind in ("characters", "styles")
    ):
        raise ValueError("Bundled presets must contain character and style maps")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = destination.open("xb")
    except FileExistsError:
        return False
    with stream:
        stream.write(content)
    return True
