"""Compile the GDN chunk port (no GPU needed): python3 build.py
Output: <package>/lib/gdn_chunk_mag.so (gdn_port.py's default; override with EXL3_GDN_PORT_SO). sm_120 only."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "lib")
sys.path.insert(0, HERE)
import gdnext  # noqa: E402

ext = gdnext.load_ext(verbose=True)
os.makedirs(LIB, exist_ok=True)
shutil.copy(ext.__file__, os.path.join(LIB, "gdn_chunk_mag.so"))
print("built", os.path.normpath(os.path.join(LIB, "gdn_chunk_mag.so")))
print("NK NV W C =", ext.NK, ext.NV, ext.W, ext.C, " PIECE_BYTES =", ext.PIECE_BYTES)
