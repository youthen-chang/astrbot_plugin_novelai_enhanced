"""Build an installable archive using an explicit source allowlist."""

import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "dist"
destination.mkdir(exist_ok=True)
archive = destination / "astrbot_plugin_novelai_enhanced-1.0.0.zip"
prefix = "astrbot_plugin_novelai_enhanced"
files = list(root.glob("*.py"))
files += [
    root / name
    for name in (
        "_conf_schema.json",
        "metadata.yaml",
        "requirements.txt",
        "presets.json",
        "README.md",
        "LICENSE",
        "PRESET_SOURCE.md",
        "pyproject.toml",
        ".gitignore",
    )
]
files += list((root / "pages").rglob("*"))
files += list((root / "tests").glob("*.py"))
files.append(Path(__file__))
files = sorted({file for file in files if file.is_file()})
with zipfile.ZipFile(
    archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
) as output:
    for file in files:
        output.write(file, str(Path(prefix) / file.relative_to(root)))
with zipfile.ZipFile(archive) as output:
    names = output.namelist()
    assert output.testzip() is None
    assert all(name.startswith(prefix + "/") for name in names)
    assert not any(
        part in name
        for name in names
        for part in (".venv/", "__pycache__/", "generation.sqlite3", "last_draw.json")
    )
    catalog = json.loads(output.read(prefix + "/presets.json"))
    assert len(catalog["characters"]) == 1062 and len(catalog["styles"]) == 61
    schema = json.loads(output.read(prefix + "/_conf_schema.json"))
    assert (
        schema["API_TOKEN"]["default"] == schema["GALLERY_UPLOAD_KEY"]["default"] == ""
    )
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(
    f"{checksum}  {archive.name}\n", encoding="ascii"
)
print(
    f"{archive}\n{len(files)} files; {archive.stat().st_size:,} bytes\nSHA256 {checksum}"
)
