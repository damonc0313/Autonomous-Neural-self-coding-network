"""Utilities and interfaces for GPT-OSS red teaming framework.

This package consolidates model interfaces, vulnerability tests, and schema helpers
that assist the red teaming orchestrator. Submodules are lazily imported to keep
startup time fast.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

__all__ = [
    "GPTOSSInterface",
    "FindingsManager",
    "ComprehensiveTestRunner",
]

# Lazy attribute loader -------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "GPTOSSInterface":
        module: ModuleType = import_module("tools.model_interface")
        return getattr(module, "GPTOSSInterface")
    if name == "FindingsManager":
        module: ModuleType = import_module("tools.findings_schema")
        return getattr(module, "FindingsManager")
    if name == "ComprehensiveTestRunner":
        module: ModuleType = import_module("tools.vulnerability_tests")
        return getattr(module, "ComprehensiveTestRunner")
    raise AttributeError(name)