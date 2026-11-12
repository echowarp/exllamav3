"""Build kvw4_kernel.cu as a torch.ops library: python3 build.py [extra nvcc flags]
Output: <package>/lib/kvw4.so (the plugin's default; override with the EXL3_*_SO variable). sm_120 (RTX 5090) only."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
from torch.utils.cpp_extension import load  # noqa: E402

bd = os.path.join(HERE, "build")
os.makedirs(bd, exist_ok=True)
load(name="kvw4", sources=[os.path.join(HERE, "kvw4_kernel.cu")], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + sys.argv[1:],
     is_python_module=False, verbose=False, build_directory=bd)
os.makedirs(LIB, exist_ok=True)
shutil.copy(os.path.join(bd, "kvw4.so"), os.path.join(LIB, "kvw4.so"))
print("built", os.path.normpath(os.path.join(LIB, "kvw4.so")))
