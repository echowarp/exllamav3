"""Build va_kernel.cu as a torch.ops library: python3 build.py [extra nvcc flags]
Output: <package>/lib/va5090.so (the plugin's default; override with the EXL3_*_SO variable). sm_120 (RTX 5090) only."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
from torch.utils.cpp_extension import load  # noqa: E402

bd = os.path.join(HERE, "build")
os.makedirs(bd, exist_ok=True)
load(name="va5090", sources=[os.path.join(HERE, "va_kernel.cu")], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + sys.argv[1:],
     is_python_module=False, verbose=False, build_directory=bd)
os.makedirs(LIB, exist_ok=True)
shutil.copy(os.path.join(bd, "va5090.so"), os.path.join(LIB, "va5090.so"))
print("built", os.path.normpath(os.path.join(LIB, "va5090.so")))
