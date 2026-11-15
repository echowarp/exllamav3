"""Build va4c_kernel.cu as a torch.ops library: python3 build.py [nvcc flags]
Default flags -DHPV -DLZT=0.f = the reference build (fp16-accumulate P.V, eager rescale; the measured numbers in
configs/int4.env). Flags given on the command line replace them, e.g. `build.py -DLZT=8.f` = fp32 P.V, lazy rescale.
Output: <package>/lib/va4c.so (the plugin's default; override with the EXL3_*_SO variable). sm_120 (RTX 5090) only."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
from torch.utils.cpp_extension import load  # noqa: E402

flags = sys.argv[1:] or ["-DHPV", "-DLZT=0.f"]
bd = os.path.join(HERE, "build")
os.makedirs(bd, exist_ok=True)
load(name="va4c", sources=[os.path.join(HERE, "va4c_kernel.cu")], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + flags,
     is_python_module=False, verbose=False, build_directory=bd)
os.makedirs(LIB, exist_ok=True)
shutil.copy(os.path.join(bd, "va4c.so"), os.path.join(LIB, "va4c.so"))
print("built", os.path.normpath(os.path.join(LIB, "va4c.so")))
