import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

# Framework integration uses an isolated root and never reads the running bot's data.
runtime = tempfile.TemporaryDirectory(
    prefix="astrbot-novelai-tests-", ignore_cleanup_errors=True
)
os.environ["ASTRBOT_ROOT"] = runtime.name
if source := os.environ.get("ASTRBOT_SOURCE_DIR"):
    sys.path.insert(0, source)
    sys.path.insert(1, str(Path(source) / "venv/Lib/site-packages"))

# Import pure modules without starting a bot.
root = Path(__file__).resolve().parents[1]
package = types.ModuleType("enhanced_test")
package.__path__ = [str(root)]
sys.modules["enhanced_test"] = package

for name in ["generation", "request_retry", "adapter", "prompts"]:
    spec = importlib.util.spec_from_file_location(
        "enhanced_test." + name, root / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
