"""CPU-only checks of the exl3cuda plugin (no GPU, no kernels built): the module imports, the vLLM entry point is
declared, env-switched features default off, and the weight-keeper loader is resolved from AllBroker first.

    python -m pytest -q exl3cuda/tests/test_cpu_plugin.py      (from the repository root)
"""
import importlib
import importlib.util
import os
import sys
import tomllib
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def plugin():
    if "exl3cuda" not in sys.modules or not sys.modules["exl3cuda"].__file__.startswith(ROOT):
        spec = importlib.util.spec_from_file_location("exl3cuda", os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        m = importlib.util.module_from_spec(spec)
        sys.modules["exl3cuda"] = m
        spec.loader.exec_module(m)
        sys.modules.pop("exl3cuda.vllm_plugin", None)
    return importlib.import_module("exl3cuda.vllm_plugin")


def test_entry_point_declared():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["project"]["entry-points"]["vllm.general_plugins"]["exl3cuda"] == "exl3cuda.vllm_plugin:register"
    assert cfg["project"]["license"]["text"] == "MIT"


def test_plugin_imports_without_gpu(plugin):
    assert callable(plugin.register)
    assert callable(plugin.whx_registry) and callable(plugin.whx_finish)


@pytest.mark.parametrize("have", [["llmt_kv.whx_loader", "exl3cuda.whx_loader", "whx_loader"],
                                  ["exl3cuda.whx_loader", "whx_loader"], ["whx_loader"]])
def test_whx_loader_order(plugin, monkeypatch, have):
    monkeypatch.delenv("EXL3_WHX_MODULE", raising=False)
    real = importlib.import_module

    def fake(name, *a, **k):
        if name in ("llmt_kv.whx_loader", "exl3cuda.whx_loader", "whx_loader"):
            if name not in have:
                raise ImportError(name)
            return types.SimpleNamespace(name=name)
        return real(name, *a, **k)
    monkeypatch.setattr(importlib, "import_module", fake)
    assert plugin._import_whx_loader().name == have[0]


def test_whx_loader_env_override(plugin, monkeypatch):
    mod = types.ModuleType("my_keeper_loader")
    monkeypatch.setitem(sys.modules, "my_keeper_loader", mod)
    monkeypatch.setenv("EXL3_WHX_MODULE", "my_keeper_loader")
    assert plugin._import_whx_loader() is mod


def test_whx_loader_missing_raises(plugin, monkeypatch):
    monkeypatch.setenv("EXL3_WHX_MODULE", "no_such_module_xyz")
    with pytest.raises(ImportError):
        plugin._import_whx_loader()


def test_whx_registry_empty_model(plugin):
    import torch
    assert plugin.whx_registry(torch.nn.Linear(2, 2)) == {}


def test_lazyvis_default_off():
    spec = importlib.util.spec_from_file_location("llmt_lazyvis_t", os.path.join(ROOT, "llmt_lazyvis.py"))
    m = importlib.util.module_from_spec(spec)
    old = os.environ.pop("EXL3_LAZY_VISION", None)
    try:
        spec.loader.exec_module(m)
    finally:
        if old is not None:
            os.environ["EXL3_LAZY_VISION"] = old
    assert m.ON is False and m.MODE == "auto" and m.TTL == 120.0
