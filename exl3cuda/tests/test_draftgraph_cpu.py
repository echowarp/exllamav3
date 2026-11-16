"""CPU checks of DRAFT-GRAPH (exl3cuda/draftgraph.py, no GPU): vLLM's propose() is rewritten around the draft loop, the
loop runs eagerly whenever the graph is off or the batch is not graphable, and install() never raises.

    python -m pytest -q exl3cuda/tests/test_draftgraph_cpu.py      (from the repository root, in an image with vLLM)
"""
import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("exl3cuda_draftgraph", os.path.join(ROOT, "draftgraph.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_default_off_and_eager_fallback():
    dg = _load()
    assert dg._ON[0] is False                       # EXL3_DRAFT_GRAPH unset: off
    assert dg._loop(None, {}, lambda L: ("ids", "probs")) == ("ids", "probs")


def test_why_reasons():
    dg = _load()
    sm = types.SimpleNamespace(all_greedy=True, all_random=False)
    P = types.SimpleNamespace(method="mtp", use_heterogeneous_vocab=False, constant_draft_positions=False,
                              eplb_state=None, num_speculative_tokens=7, _enable_probabilistic_draft_probs=False)
    assert dg._why(P, {"sampling_metadata": sm, "batch_size": 2}) is None
    assert dg._why(P, {"sampling_metadata": sm, "batch_size": dg._BMAX + 1}) == "bmax"
    P.num_speculative_tokens = 1
    assert dg._why(P, {"sampling_metadata": sm, "batch_size": 2}) == "passes"
    P.num_speculative_tokens = 7
    mixed = types.SimpleNamespace(all_greedy=False, all_random=False)
    assert dg._why(P, {"sampling_metadata": mixed, "batch_size": 2}) == "sampling"


def test_install_rewrites_propose_and_is_safe():
    pytest.importorskip("vllm")
    dg = _load()
    import vllm.v1.spec_decode.llm_base_proposer as m
    P = m.SpecDecodeBaseProposer
    orig = P.propose
    try:
        assert dg._install_graph() is True
        assert P.propose is not orig
        assert P.propose.__code__.co_filename == "<llmt_dg_propose>"
        assert callable(m.__dict__["_llmt_dg_stock"]) and m.__dict__["_llmt_dg_loop"] is dg._loop
    finally:
        P.propose = orig
    dg.install()                                      # without a GPU the runner import may fail: must not raise
