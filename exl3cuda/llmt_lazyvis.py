"""Lazy vision encoder for vLLM Qwen3-VL-style towers.

EXL3_LAZY_VISION=1: the vision tower's weights live in pinned host memory, not on the GPU.
  - Offloaded right after weight loading, inside the window where vLLM measures the model's weight memory, so the KV
    cache is sized without the tower: its bytes go to KV tokens.
  - The startup profile run streams the tower (below), so the KV budget never contains the tower either.
  - On an image request (visual.forward), the tower is either
      resident: copied in whole (one H2D pass), kept while images keep coming, released after EXL3_LAZY_VISION_TTL
                seconds without an image (default 120) or as soon as free memory falls below the activation reserve
                (memory pressure: the activation reserve no longer fits; KV always wins). Allowed only when free + cached-unused device memory covers
                tower + the profiled activation peak + EXL3_LAZY_VISION_MARGIN_MB (default 256), so it never eats into
                what the engine already counted on.
      streamed: block by block (non-block parts + one 27-block slice at a time, freed after use) when there is no
                room for the whole tower. Needs only ~one block (~30 MB) beyond the profiled activations.
  - EXL3_LAZY_VISION_MODE=auto (default) | stream (always stream) | resident (always load whole; tests only: the
    KV gain then has no guaranteed room).
  - Text-only requests never call visual.forward, so they never load anything.
Logs counts/timings only (no request data)."""
import os
import threading
import time

import torch

# device-neutral: CUDA (RTX 5090) as well as Intel XPU (Arc)
_ACC = torch.cuda if torch.cuda.is_available() else torch.xpu

_ENV = os.environ.get
ON = _ENV("EXL3_LAZY_VISION", "0") == "1"
TTL = float(_ENV("EXL3_LAZY_VISION_TTL", "120"))
MARGIN = int(float(_ENV("EXL3_LAZY_VISION_MARGIN_MB", "256")) * 2**20)
MODE = _ENV("EXL3_LAZY_VISION_MODE", "auto")
_STATE = {"ready": False, "act_reserve": None, "towers": []}


def _log(msg, *a):
    try:
        from vllm.logger import init_logger
        init_logger("vllm.lazyvis").info("LAZYVIS: " + msg, *a)
    except Exception:  # noqa: BLE001
        print("LAZYVIS: " + (msg % a), flush=True)


def _sync():
    try:
        torch.accelerator.synchronize()
    except Exception:  # noqa: BLE001
        _ACC.synchronize()


class _Unit:
    """One group of tensors (params + buffers) that moves together: pinned host master copies, device copies on demand."""

    def __init__(self, mod_tensors, copy=True):
        self.items = []   # (module, name, kind, host tensor)
        for m, name, kind, t in mod_tensors:
            host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            if copy:
                host.copy_(t.detach())
            self.items.append((m, name, kind, host))
        self.bytes = sum(h.numel() * h.element_size() for *_, h in self.items)
        self.on_dev = True

    def _set(self, m, name, kind, t):
        if kind == "p":
            m._parameters[name].data = t
        else:
            m._buffers[name] = t

    def to_host(self):
        for m, name, kind, host in self.items:
            self._set(m, name, kind, host)
        self.on_dev = False

    def to_dev(self, dev):
        for m, name, kind, host in self.items:
            d = torch.empty(host.shape, dtype=host.dtype, device=dev)
            d.copy_(host, non_blocking=True)
            self._set(m, name, kind, d)
        self.on_dev = True


class LazyVision:
    def __init__(self, vis, copy=True):
        self.vis, self.lock = vis, threading.RLock()
        self.dev = next(vis.parameters()).device
        blocks = getattr(vis, "blocks", None)
        block_mods = set()
        self.block_units = []
        for blk in (blocks or []):
            ts = self._tensors(blk)
            block_mods.update(id(m) for m in blk.modules())
            self.block_units.append(_Unit(ts, copy))
        head = [t for t in self._tensors(vis) if id(t[0]) not in block_mods]
        self.head = _Unit(head, copy)
        self.bytes = self.head.bytes + sum(u.bytes for u in self.block_units)
        stray = [(type(m).__name__, k) for m in vis.modules() for k, v in vars(m).items()
                 if isinstance(v, torch.Tensor) and v.device.type != "cpu" and k not in ("_parameters", "_buffers")]
        for u in [self.head] + self.block_units:
            u.to_host()
        self.resident, self.streaming, self.last_use, self.n_img = False, False, 0.0, 0
        self.stats = {"resident_loads": 0, "streamed": 0, "releases": 0}
        if blocks is not None:
            for blk, u in zip(blocks, self.block_units):
                blk.register_forward_pre_hook(self._blk_pre(u))
                blk.register_forward_hook(self._blk_post(u))
        orig = vis.forward

        def forward(*a, **k):
            with self.lock:
                self._enter()
                try:
                    return orig(*a, **k)
                finally:
                    self._exit()
        vis.forward = forward
        vis._lazyvis = self
        _log("tower " + ("offloaded to" if copy else "placed in (before weight loading)") + " pinned host memory: %.3f GiB (%d blocks, head %.1f MiB)%s; mode %s, ttl %.0f s, "
             "margin %d MiB", self.bytes / 2**30, len(self.block_units), self.head.bytes / 2**20,
             (" stray device tensors %s" % stray) if stray else "", MODE, TTL, MARGIN >> 20)

    @staticmethod
    def _tensors(mod):
        out = []
        for m in mod.modules():
            for n, p in m._parameters.items():
                if p is not None:
                    out.append((m, n, "p", p))
            for n, b in m._buffers.items():
                if b is not None and b.device.type != "cpu":
                    out.append((m, n, "b", b))
        return out

    # --- block streaming hooks (active only while self.streaming) ---
    def _blk_pre(self, u):
        def pre(_m, _a):
            if self.streaming:
                u.to_dev(self.dev)
        return pre

    def _blk_post(self, u):
        def post(_m, _a, _o):
            if self.streaming:
                u.to_host()
        return post

    def _room(self):
        """Device memory left after the engine's activation reserve (profiled peak), counting what is free on the
        device, cached unused in torch, and activations in flight right now (they come out of that reserve)."""
        free, _total = _ACC.mem_get_info(self.dev)
        alloc = _ACC.memory_allocated(self.dev)
        cached = _ACC.memory_reserved(self.dev) - alloc
        transient = 0
        if _STATE.get("base_alloc") is not None:
            transient = max(0, alloc - _STATE["base_alloc"] - (self.bytes if self.resident else 0))
        return free + cached + transient - (_STATE["act_reserve"] or 0)

    def _enter(self):
        now = time.time()
        self.last_use, self.n_img = now, self.n_img + 1
        if self.resident:
            return
        want = MODE == "resident" or (MODE == "auto" and _STATE["ready"] and _STATE["act_reserve"] is not None
                                      and self._room() >= self.bytes + MARGIN)
        if want:
            t0 = time.perf_counter()
            try:
                for u in [self.head] + self.block_units:
                    u.to_dev(self.dev)
                _sync()
                self.resident = True
                self.stats["resident_loads"] += 1
                _log("tower loaded resident: %.3f GiB in %.1f ms (load %d)", self.bytes / 2**30,
                     (time.perf_counter() - t0) * 1e3, self.stats["resident_loads"])
                return
            except torch.OutOfMemoryError:
                for u in [self.head] + self.block_units:
                    u.to_host()
                _log("resident load hit OOM: streaming instead")
        self.streaming = True
        self.head.to_dev(self.dev)
        self.stats["streamed"] += 1
        if self.stats["streamed"] <= 3 or self.stats["streamed"] % 100 == 0:
            _log("streaming the tower block by block (call %d; ready %s, room %s)", self.stats["streamed"],
                 _STATE["ready"], None if _STATE["act_reserve"] is None else "%.2f GiB" % (self._room() / 2**30))

    def _exit(self):
        if self.streaming:
            self.head.to_host()
            for u in self.block_units:
                if u.on_dev:
                    u.to_host()
            self.streaming = False

    def maybe_release(self):
        with self.lock:
            if not self.resident:
                return
            idle = time.time() - self.last_use
            why = None
            if idle >= TTL:
                why = "idle %.0f s" % idle
            elif MODE == "auto" and self._room() < 0:   # the activation reserve is no longer guaranteed
                why = "memory pressure"
            if why:
                for u in [self.head] + self.block_units:
                    u.to_host()
                self.resident = False
                self.stats["releases"] += 1
                _log("tower released (%s): %.3f GiB back to the device pool", why, self.bytes / 2**30)


def _reaper():
    while True:
        time.sleep(min(5.0, max(0.5, TTL / 4)))
        for lv in list(_STATE["towers"]):
            try:
                lv.maybe_release()
            except Exception as e:  # noqa: BLE001
                _log("release check failed: %r", e)


def _offload(model, early):
    vis = getattr(model, "visual", None)
    if isinstance(vis, torch.nn.Module) and hasattr(vis, "_lazyvis") and not early:
        # loaded into the pinned masters; process_weights_after_loading's device_loading_context may have swapped in
        # plain CPU copies: point every tensor back at its pinned master
        lv = vis._lazyvis
        with lv.lock:
            for u in [lv.head] + lv.block_units:
                u.to_host()
        return
    if isinstance(vis, torch.nn.Module) and not hasattr(vis, "_lazyvis") and any(True for _ in vis.parameters()):
        try:
            lv = LazyVision(vis, copy=True)   # always copy: buffers (rotary inv_freq, ...) are set at construction, not loaded
        except Exception as e:  # noqa: BLE001
            _log("not enabled (%r): tower stays resident", e)
            return
        # hand the freed blocks back to the device: vLLM's profiler counts torch-cached-but-unallocated memory as
        # "non-torch" usage, which would take the tower's bytes away from the KV cache again
        _sync()
        _ACC.empty_cache()
        _STATE["towers"].append(lv)
        if len(_STATE["towers"]) == 1:
            threading.Thread(target=_reaper, name="lazyvis-reaper", daemon=True).start()


def _allbroker_lazyvis():
    """True when AllBroker's llmt_kv vLLM plugin is installed (and not switched off): its llmt_kv.lazyvis then manages
    the tower for EXL3_LAZY_VISION=1, and this standalone copy must not wrap the same loader points a second time."""
    if _ENV("LLMT_KV_PLUGIN", "1") == "0":
        return False
    try:
        from importlib.metadata import entry_points
        return any(ep.value.startswith("llmt_kv.") for ep in entry_points(group="vllm.general_plugins"))
    except Exception:  # noqa: BLE001
        return False


def install():
    """Called from the patched vllm/model_executor/models/qwen3_vl.py at import (worker process, before weights load)."""
    if not ON or _STATE.get("installed"):
        return
    _STATE["installed"] = True
    if _allbroker_lazyvis():
        _log("AllBroker's llmt_kv.lazyvis is installed and manages the tower; standalone copy not installed")
        return
    import vllm.model_executor.model_loader.base_loader as bl
    orig_pwal = bl.process_weights_after_loading

    def pwal(model, *a, **k):
        r = orig_pwal(model, *a, **k)
        _offload(model, early=False)
        return r
    bl.process_weights_after_loading = pwal

    # earliest point: right after the model is built, before load_weights (base_loader calls log_model_inspection in
    # between). The tower's (still uninitialized) params move to pinned host buffers and the checkpoint is read straight
    # into them, so the tower never holds device memory that the driver might keep pooled afterwards.
    orig_inspect = bl.log_model_inspection

    def inspect(model, *a, **k):
        r = orig_inspect(model, *a, **k)
        _offload(model, early=True)
        return r
    bl.log_model_inspection = inspect

    from vllm.v1.worker import gpu_worker as gw
    orig_dam = gw.Worker.determine_available_memory

    def dam(self, *a, **k):
        r = orig_dam(self, *a, **k)
        _STATE["act_reserve"] = int(getattr(self, "peak_activation_memory", 0) or 0)
        _log("activation reserve from the profile run: %.2f GiB", _STATE["act_reserve"] / 2**30)
        return r
    gw.Worker.determine_available_memory = dam

    orig_warm = gw.Worker.compile_or_warm_up_model

    def warm(self, *a, **k):
        r = orig_warm(self, *a, **k)
        _STATE["ready"] = True
        for lv in _STATE["towers"]:   # whatever warm-up loaded goes back out: text-only traffic never holds the tower
            with lv.lock:
                if lv.resident:
                    for u in [lv.head] + lv.block_units:
                        u.to_host()
                    lv.resident = False
        _sync()
        _STATE["base_alloc"] = _ACC.memory_allocated()   # idle engine: weights + KV + persistent buffers
        return r
    gw.Worker.compile_or_warm_up_model = warm
    _log("installed (mode %s, ttl %.0f s)", MODE, TTL)
