"""DRAFT-GRAPH for CUDA (EXL3_DRAFT_GRAPH=1, default off): MTP draft passes 2..k of SpecDecodeBaseProposer.propose as ONE
captured CUDA graph per (batch, padded batch, passes, greedy, DYN on, far-head pass index), replayed with one launch instead
of ~7 piecewise-graph launches + eager glue per pass (va4 attention dispatch, eagle step update, MTP embedding, DYN-ROW
logits, draft sampling). The idea follows exl3xpu's draftgraph.py (Intel Arc), ported to CUDA.

Exactness: the captured body IS vLLM's own loop (the source of propose() between the loop markers, exec'd as a function of
propose()'s locals), run on static copies of its inputs with cudagraph_runtime_mode NONE, so the replay launches the same
kernels on the same values as the eager (piecewise) loop. The plugin's DYN draft rows switch themselves off under stream
capture; here the capture enables them (graph keyed on whether the batch's DYN map is active), reading the plugin's
persistent per-batch tables, exactly what the eager pass reads.
Not graphed (eager stock loop): mixed greedy / random batches, heterogeneous vocab, constant draft positions, EPLB, passes
< 1, batches above EXL3_DRAFT_GRAPH_BMAX, capture failure for that key (logged once; that key stays eager), more than
EXL3_DRAFT_GRAPH_MAX graphs. Any failure to install leaves vLLM's loop untouched.

Settings: EXL3_DRAFT_GRAPH=1; EXL3_DRAFT_GRAPH_MAX (16 graphs), EXL3_DRAFT_GRAPH_BMAX (4, the largest batch graphed: graph
pools are torch memory outside the KV budget), EXL3_DRAFT_GRAPH_CHECK=N (the first N greedy replays also run the eager loop
on the same inputs and compare the drafts; log DRAFT-GRAPH-CHECK), EXL3_DRAFT_GRAPH_CTL=<json file> (lab: {"graph": 0|1}
flips graph / eager in a running engine for same-engine A/Bs). Loaded by exl3cuda.vllm_plugin.register()."""
from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import textwrap
import threading
import time

import torch

_ON = [os.environ.get("EXL3_DRAFT_GRAPH", "0") == "1"]
_MAX = int(os.environ.get("EXL3_DRAFT_GRAPH_MAX", "16"))
_BMAX = int(os.environ.get("EXL3_DRAFT_GRAPH_BMAX", "4"))    # graph pools are torch memory outside the KV budget
_CHECK = [int(os.environ.get("EXL3_DRAFT_GRAPH_CHECK", "0"))]
_ST: dict = {"caps": {}, "failed": set(), "stats": {"replay": 0, "stock": 0, "n": 0, "why": {}}, "chk": [0, 0]}
_LOOP_START = "        for token_index in range(self.num_speculative_tokens - 1):\n"
_LOOP_END = "        # [batch_size, num_speculative_tokens]\n"
_CALL = "        draft_token_ids_list, draft_probs_list = _llmt_dg_loop(self, locals(), _llmt_dg_stock)\n"
_NAMES = ["self", "draft_token_ids_list", "draft_probs_list", "hidden_states", "positions", "common_attn_metadata",
          "batch_size", "input_batch_size", "cudagraph_runtime_mode", "batch_size_across_dp", "block_size",
          "sampling_metadata", "per_layer_attn_metadata"]
log = logging.getLogger("exl3cuda.draftgraph")
_plugin = {"mod": None}


def _stock_fn(loop_src, g):
    body = "def _llmt_dg_stock(L):\n" + "".join(f"    {n} = L[{n!r}]\n" for n in _NAMES)
    body += textwrap.indent(textwrap.dedent(loop_src), "    ")
    body += "    return draft_token_ids_list, draft_probs_list\n"
    ns: dict = {}
    exec(compile(body, "<llmt_dg_stock>", "exec"), g, ns)
    return ns["_llmt_dg_stock"]


def _dyn_active(self, B):
    m = _plugin["mod"]
    if m is None:
        return False
    mgr = (getattr(m, "_DYN", None) or {}).get("mgr")
    return bool(mgr is not None and not getattr(mgr, "disabled", False) and mgr.any and mgr.B == B)


def _why(self, L):
    sm = L["sampling_metadata"]
    for name, ok in (("method", self.method == "mtp"),
                     ("sampling", sm.all_greedy or sm.all_random),   # draft sampling ignores generators (upstream too)
                     ("hetvocab", not self.use_heterogeneous_vocab), ("constpos", not self.constant_draft_positions),
                     ("eplb", self.eplb_state is None), ("passes", self.num_speculative_tokens >= 2),
                     ("bmax", L["batch_size"] <= _BMAX),
                     ("probs", sm.all_greedy or self._enable_probabilistic_draft_probs)):
        if not ok:
            return name
    return None


def _count(kind, why=None):
    st = _ST["stats"]
    st[kind] += 1; st["n"] += 1
    if why:
        st["why"][why] = st["why"].get(why, 0) + 1
    if st["n"] % 2000 == 0:
        log.warning("DRAFT-GRAPH last 2000 loops: %d replayed, %d eager %s; %d graphs, %d failed keys; check %d/%d bad",
                    st["replay"], st["stock"], st["why"], len(_ST["caps"]), len(_ST["failed"]), _ST["chk"][1], _ST["chk"][0])
        st.update(replay=0, stock=0, why={})


class _Cap:
    pass


def _fill(c, L):
    B = c.B
    c.ids.copy_(L["draft_token_ids_list"][-1][:B])
    c.hid.copy_(L["hidden_states"][:B])
    c.pos.copy_(L["positions"])
    cm = L["common_attn_metadata"]
    c.seq.copy_(cm.seq_lens[:B])
    c.bt.copy_(cm.block_table_tensor[:B])
    sm = L["sampling_metadata"]
    if c.temp is not None:
        c.temp.copy_(sm.temperature[:B])
    if c.tk is not None:
        c.tk.copy_(sm.top_k[:B])
    if c.tp is not None:
        c.tp.copy_(sm.top_p[:B])


def _static_L(self, L, c):
    from vllm.config import CUDAGraphMode
    cm = L["common_attn_metadata"]
    scm = dataclasses.replace(cm, seq_lens=c.seq, block_table_tensor=c.bt, max_seq_len=self.max_model_len)
    for f in ("_seq_lens_cpu", "_num_computed_tokens_cpu"):
        if hasattr(scm, f):
            setattr(scm, f, None)
    sm = L["sampling_metadata"]
    rep = {k: v for k, v in (("temperature", c.temp), ("top_k", c.tk), ("top_p", c.tp)) if v is not None}
    ssm = dataclasses.replace(sm, **rep) if rep else sm
    return dict(L, draft_token_ids_list=[c.ids], draft_probs_list=None if L["draft_probs_list"] is None else [],
                hidden_states=c.hid, positions=c.pos, common_attn_metadata=scm, sampling_metadata=ssm,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)


def _capture(self, L, key, stock):
    B, IB, steps, greedy, dyn, k0, _, _ = key
    cm = L["common_attn_metadata"]
    c = _Cap()
    c.B = B
    c.ids = L["draft_token_ids_list"][-1][:B].clone()
    c.hid = L["hidden_states"][:B].clone()
    c.pos = L["positions"].clone()
    c.seq = cm.seq_lens[:B].clone()
    c.bt = cm.block_table_tensor[:B].clone()
    sm = L["sampling_metadata"]
    c.temp = None if (greedy or sm.temperature is None) else sm.temperature[:B].clone()
    c.tk = None if (greedy or sm.top_k is None) else sm.top_k[:B].clone()
    c.tp = None if (greedy or sm.top_p is None) else sm.top_p[:B].clone()
    m = _plugin["mod"]
    orig_cap = torch.cuda.is_current_stream_capturing

    def body():
        self.model._exl3_pass = k0
        SL = _static_L(self, L, c)
        if dyn and m is not None:      # the plugin's DYN row skips under capture; the eager pass would use it
            torch.cuda.is_current_stream_capturing = lambda: False
        try:
            ids, probs = stock(SL)
        finally:
            torch.cuda.is_current_stream_capturing = orig_cap
        return ids[1:], probs

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):      # warm-up: Triton / Inductor first calls, allocator
        _fill(c, L)
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    _fill(c, L)
    g = torch.cuda.CUDAGraph()
    pool = _ST.setdefault("pool", torch.cuda.graph_pool_handle())
    with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
        c.outs, c.probs = body()
    c.g = g
    c.k_after = self.model._exl3_pass
    torch.cuda.synchronize()
    free = torch.cuda.mem_get_info()[0] / 2**20
    log.warning("DRAFT-GRAPH captured B=%d IB=%d passes=%d %s dyn=%s k0=%d (%d graphs); torch reserved %.0f MiB, device free %.0f MiB",
                B, IB, steps, "greedy" if greedy else "random", dyn, k0, len(_ST["caps"]) + 1,
                torch.cuda.memory_reserved() / 2**20, free)
    return c


def _loop(self, L, stock):
    if not _ON[0]:
        return stock(L)
    w = _why(self, L)
    if w:
        _count("stock", w)
        return stock(L)
    B, IB = L["batch_size"], L["input_batch_size"]
    steps = self.num_speculative_tokens - 1
    sm = L["sampling_metadata"]
    greedy = bool(sm.all_greedy)
    k0 = getattr(self.model, "_exl3_pass", 0)
    key = (B, IB, steps, greedy, _dyn_active(self, B), k0, sm.top_k is None, sm.top_p is None)
    c = _ST["caps"].get(key)
    if c is None:
        if key in _ST["failed"] or len(_ST["caps"]) >= _MAX:
            _count("stock", "nokey")
            return stock(L)
        try:
            _ST["caps"][key] = _capture(self, L, key, stock)
        except Exception as e:  # noqa: BLE001
            _ST["failed"].add(key)
            log.warning("DRAFT-GRAPH capture failed for %s (%r): eager loop for this key", key, e)
        torch.cuda.synchronize()
        self.model._exl3_pass = k0
        _count("stock", "capture")
        return stock(L)       # the capture step itself runs eagerly (inputs untouched: the capture used copies)
    cm = L["common_attn_metadata"]
    chk = greedy and _CHECK[0] > 0
    if chk:
        seq0 = cm.seq_lens[:B].clone()
    _fill(c, L)
    c.g.replay()
    self.model._exl3_pass = c.k_after
    _count("replay")
    if chk:
        _CHECK[0] -= 1
        got = torch.stack([o[:B] for o in c.outs], 1).clone()
        cm.seq_lens[:B].copy_(seq0)
        self.model._exl3_pass = k0
        ref, _ = stock(dict(L, draft_token_ids_list=list(L["draft_token_ids_list"]), draft_probs_list=None))
        self.model._exl3_pass = c.k_after
        refm = torch.stack([r[:B] for r in ref[1:]], 1)
        bad = int((refm != got).any().item())
        _ST["chk"][0] += 1; _ST["chk"][1] += bad
        if bad or _ST["chk"][0] % 20 == 0:
            log.warning("DRAFT-GRAPH-CHECK key %s: %d checks, %d differing; last ref %s graph %s", key, _ST["chk"][0],
                        _ST["chk"][1], refm.tolist()[:2], got.tolist()[:2])
    # caller-visible state as the eager loop leaves it
    cm.seq_lens[:B].copy_(c.seq)
    cm.slot_mapping = self._slot_mapping_buffer[:B]
    cm.max_seq_len = min(cm.max_seq_len + steps, self.max_model_len)
    for f in ("_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
        v = getattr(cm, f, None)
        if v is not None:
            setattr(cm, f, v + steps)
    ids = L["draft_token_ids_list"] + [o[:B] for o in c.outs]
    dpl = L["draft_probs_list"]
    if dpl is not None and c.probs is not None:
        dpl = dpl + list(c.probs)
    return ids, dpl


def _orig_propose(P):
    """vLLM's own propose() and the closure cell holding it (None: it is P.propose itself). Wrappers (the plugin's
    graded-vocab reset) stay in place: only the innermost function is swapped."""
    seen, todo = set(), [(P.propose, None)]
    while todo:
        f, cell = todo.pop()
        if id(f) in seen or not callable(f):
            continue
        seen.add(id(f))
        code = getattr(f, "__code__", None)
        if code is not None and code.co_filename.endswith("llm_base_proposer.py") and f.__name__ == "propose":
            return f, cell
        for cl in (getattr(f, "__closure__", None) or ()):
            try:
                todo.append((cl.cell_contents, cl))
            except ValueError:
                pass
    return None, None


def _install_graph():
    import vllm.v1.spec_decode.llm_base_proposer as m
    P = m.SpecDecodeBaseProposer
    f, cell = _orig_propose(P)
    if f is None:
        log.warning("DRAFT-GRAPH: vLLM propose() not found: not installed"); return False
    src = textwrap.indent(textwrap.dedent(inspect.getsource(f)), "    ")
    i, j = src.find(_LOOP_START), src.find(_LOOP_END)
    if i < 0 or j < i:
        log.warning("DRAFT-GRAPH: loop markers not found: not installed"); return False
    g = f.__globals__          # the module dict itself: later module-level patches stay visible
    g["_llmt_dg_stock"] = _stock_fn(_LOOP_START + src[i + len(_LOOP_START):j], g)
    g["_llmt_dg_loop"] = _loop
    ns: dict = {}
    exec(compile("class _X:\n" + src[:i] + _CALL + src[j:], "<llmt_dg_propose>", "exec"), g, ns)
    new = ns["_X"].propose
    if cell is None:
        P.propose = new
    else:
        cell.cell_contents = new
    return True


def _control(path):
    last = None
    while True:
        time.sleep(0.5)
        try:
            mt = os.stat(path).st_mtime
        except OSError:
            continue
        if mt == last:
            continue
        last = mt
        try:
            c = json.load(open(path))
        except (OSError, ValueError):
            continue
        if "graph" in c:
            _ON[0] = bool(c["graph"])
            log.warning("DRAFT-GRAPH ctl: %s", "graph" if _ON[0] else "eager")
        if "check" in c:
            _CHECK[0] = int(c["check"])


def install():
    """Called by exl3cuda.vllm_plugin.register() when EXL3_DRAFT_GRAPH=1. Installs lazily (after the plugin patched the
    proposer); any failure leaves vLLM's loop in place."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("DRAFT-GRAPH not installed (%r)", e)
        return
    try:
        import exl3cuda.vllm_plugin as pm
        _plugin["mod"] = pm
    except Exception:  # noqa: BLE001
        pass
    done = {"v": False}
    orig = GPUModelRunner.propose_draft_token_ids

    def propose_draft_token_ids(self, *a, **k):
        if not done["v"]:
            done["v"] = True
            try:
                ok = _install_graph()
            except Exception as e:  # noqa: BLE001
                ok = False
                log.warning("DRAFT-GRAPH install failed (%r): vLLM's loop stays", e)
            log.warning("DRAFT-GRAPH %s (max %d graphs, batch <= %d, DYN plugin %s)", "installed" if ok else "NOT installed",
                        _MAX, _BMAX, _plugin["mod"] is not None)
        return orig(self, *a, **k)
    GPUModelRunner.propose_draft_token_ids = propose_draft_token_ids
    ctl = os.environ.get("EXL3_DRAFT_GRAPH_CTL")
    if ctl:
        threading.Thread(target=_control, args=(ctl,), daemon=True, name="draftgraph-ctl").start()
