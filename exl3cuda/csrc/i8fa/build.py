"""Build the I8FA kernels:
  python3 build.py i8kv                 gather-dequant + LSE merge (i8kv.cu; built without fast math)
  python3 build.py i8fa_fp 1 2          paged in-kernel-dequant FA2 with FA16 f numerics (used with EXL3_FA16=f)
  python3 build.py i8fa_a 0 0           the same with stock FA2 numerics (used when EXL3_FA16 is unset)
The i8fa_* kernels include the patched FA2 tree: run ../fa16/fetch_and_patch.sh first (or set FA_SRC / CUTLASS_INC).
Output: <package>/lib/NAME/NAME.so (the plugin looks in EXL3_I8FA_DIR/NAME/, default <package>/lib). sm_120 only.
Extra args after QK are passed to nvcc. EXL3_EXTRA_INC: extra include dirs (':'-separated), as in ../fa16/build.py.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
FA16 = os.path.join(HERE, "..", "fa16")
name = sys.argv[1]
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("MAX_JOBS", "1")
from torch.utils.cpp_extension import load  # noqa: E402

extra_inc = [p for p in os.environ.get("EXL3_EXTRA_INC", "").split(":") if p]
bd = os.path.join(HERE, "build", name)
os.makedirs(bd, exist_ok=True)
if name == "i8kv":
    m = load(name=name, sources=[os.path.join(HERE, "i8kv.cu")], build_directory=bd, verbose=False,
             extra_include_paths=extra_inc,
             extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120", "-static-global-template-stub=false",
                                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
             extra_cflags=["-O2", "-std=c++17"])
else:
    pv, qk = sys.argv[2], sys.argv[3]
    extra = sys.argv[4:]
    src = os.environ.get("FA_SRC", os.path.join(FA16, "srcmod"))
    cutlass = os.environ.get("CUTLASS_INC", os.path.join(FA16, "fa", "csrc", "cutlass", "include"))
    m = load(name=name, sources=[os.path.join(HERE, "i8fa.cu")], build_directory=bd, verbose=False,
             extra_include_paths=[src, cutlass, HERE] + extra_inc,
             extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                                "-gencode=arch=compute_120,code=sm_120", f"-DFA16_PV={pv}", f"-DFA16_QK={qk}",
                                "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_ALIBI", "-DFLASHATTENTION_DISABLE_SOFTCAP",
                                "-DFLASHATTENTION_DISABLE_UNEVEN_K", "-DFLASHATTENTION_DISABLE_LOCAL",
                                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                                "-Xptxas=-v", "-lineinfo", "-static-global-template-stub=false"] + extra,
             extra_cflags=["-O2", "-std=c++17"])
out = os.path.join(LIB, name)
os.makedirs(out, exist_ok=True)
shutil.copy(m.__file__, os.path.join(out, f"{name}.so"))
print("built", os.path.normpath(os.path.join(out, f"{name}.so")))
