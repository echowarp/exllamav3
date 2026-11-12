"""Build/load the Magnitude GDN chunk port (gdn_chunk.cu) as a torch extension, sm_120 (RTX 5090)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_chunk.cu")
BUILD = os.path.join(HERE, "build")


def load_ext(verbose=False):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "10")
    from torch.utils.cpp_extension import load
    os.makedirs(BUILD, exist_ok=True)
    return load(
        name="gdn_chunk_mag",
        sources=[SRC],
        build_directory=BUILD,
        # -fmad=false: Magnitude forms its CUDA kernels with --fmad=false (fused multiply-adds only where the source
        # writes __fmaf_rn), which its bit-exactness claims rely on.
        extra_cuda_cflags=["-O3", "-fmad=false", "-lineinfo", "-std=c++17", "-Xptxas=-v"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
