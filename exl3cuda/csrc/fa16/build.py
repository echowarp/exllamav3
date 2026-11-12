"""Build one FA16 variant: python3 build.py NAME FA16_PV FA16_QK  (production variant f: `build.py fa_f 1 2`).

Needs ./srcmod and ./fa from fetch_and_patch.sh (or FA_SRC / CUTLASS_INC). Output: <package>/lib/NAME/NAME.so; the
plugin imports module fa_<EXL3_FA16> from /fa16ext/fa_<v> or from sys.path, so put <package>/lib/fa_f on PYTHONPATH.
Variants: fa_b = 1 0 (P.V fp16 per 64-key block), fa_f = 1 2 (fa_b + Q.K fp16 per 64 dims). sm_120 only.
One nvcc needs ~12 GB RAM: MAX_JOBS=1 is the default. EXL3_EXTRA_INC: extra include dirs (':'-separated), e.g. for
CUDA library headers (cuSPARSE / cuSOLVER / cuFFT) that torch's headers include but a runtime image may lack.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
name, pv, qk = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("MAX_JOBS", "1")
from torch.utils.cpp_extension import load  # noqa: E402

src = os.environ.get("FA_SRC", os.path.join(HERE, "srcmod"))
cutlass = os.environ.get("CUTLASS_INC", os.path.join(HERE, "fa", "csrc", "cutlass", "include"))
extra_inc = [p for p in os.environ.get("EXL3_EXTRA_INC", "").split(":") if p]
bd = os.path.join(HERE, "build", name)
os.makedirs(bd, exist_ok=True)
m = load(name=name, sources=[os.path.join(HERE, "fa16.cu")], build_directory=bd, verbose=False,
         extra_include_paths=[src, cutlass] + extra_inc,
         extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                            "-gencode=arch=compute_120,code=sm_120", f"-DFA16_PV={pv}", f"-DFA16_QK={qk}",
                            "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_ALIBI", "-DFLASHATTENTION_DISABLE_SOFTCAP",
                            "-DFLASHATTENTION_DISABLE_UNEVEN_K", "-DFLASHATTENTION_DISABLE_LOCAL",
                            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                            "-Xptxas=-v", "-lineinfo", "-static-global-template-stub=false"],
         extra_cflags=["-O2", "-std=c++17"])
out = os.path.join(LIB, name)
os.makedirs(out, exist_ok=True)
shutil.copy(m.__file__, os.path.join(out, f"{name}.so"))
print("built", os.path.normpath(os.path.join(out, f"{name}.so")))
