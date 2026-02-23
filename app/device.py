"""
device.py – auto-detect GPU/CPU and expose helpers.

Set env var FORCE_CPU=1 to force CPU mode for testing.
"""
import os
import torch


def get_device() -> str:
    """Return 'cuda' if a usable GPU is available, otherwise 'cpu'."""
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def has_gpu() -> bool:
    return get_device() == "cuda"


def gpu_info() -> dict:
    """Return a summary dict about the current compute device."""
    dev = get_device()
    info = {"device": dev}
    if dev == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e6)
        info["vram_free_mb"] = round(torch.cuda.mem_get_info(0)[0] / 1e6)
    else:
        import multiprocessing
        info["cpu_cores"] = multiprocessing.cpu_count()
    return info


# ── print once at import ──
_dev = get_device()
if _dev == "cuda":
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("💻 Running on CPU mode")
