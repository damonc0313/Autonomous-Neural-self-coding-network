# Ensure project root is on PYTHONPATH
import importlib
import os
import sys
from pathlib import Path
import pytest

# Add repository root to path for module resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_NAMES = [
    "digital_crucible",
    "graphformic_coder",
    "autonomous_evolution_engine",
    "core.neural_processor",
]

@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name):
    """Ensure that each critical top-level module can be imported successfully."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(f"Failed to import {module_name}: {exc}")