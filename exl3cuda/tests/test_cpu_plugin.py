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


def _src(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def test_kernel_defaults_match_build_outputs():
    """Every '<package>/lib/<x>.so' default names a file a csrc/*/build.py writes; the VA4 op default is the
    TORCH_LIBRARY namespace of the shipped VA4 kernel."""
    import glob
    import re
    py = "".join(_src(os.path.basename(p)) for p in glob.glob(os.path.join(ROOT, "*.py")))
    builds = "".join(_src(os.path.relpath(p, ROOT)) for p in glob.glob(os.path.join(ROOT, "csrc", "*", "build.py")))
    so = set(re.findall(r'"lib", "([A-Za-z0-9_]+\.so)"', py))
    assert so >= {"va5090.so", "va4c.so", "kvw4.so", "gdn_chunk_mag.so"}
    for name in so:
        assert f'"{name}"' in builds, name
    op = set(re.findall(r'os\.environ\.get\("EXL3_VA4_OP", "([a-z0-9_]+)"\)', py))
    assert op == set(re.findall(r"TORCH_LIBRARY\(([a-z0-9_]+),", _src("csrc", "va4", "va4c_kernel.cu")))


@pytest.mark.parametrize("name", ["int4.env", "int8.env"])
def test_config_env_files(name):
    """configs/*.env: plain VAR=value lines (docker --env-file and `set -a; .` both read them) and only variables
    the package reads."""
    import glob
    import re
    py = "".join(_src(os.path.basename(p)) for p in glob.glob(os.path.join(ROOT, "*.py")))
    seen = []
    for line in _src("configs", name).splitlines():
        m = re.match(r"#?(EXL3_[A-Z0-9_]+)=(.*)$", line)
        if m:
            assert re.fullmatch(r"[^\s#'\"]*", m.group(2)), line
            assert f'"{m.group(1)}"' in py, m.group(1)
            seen.append(m.group(1))
        else:
            assert line == "" or line.startswith("#"), line
    assert len(seen) == len(set(seen)) and "EXL3_VA" in seen
