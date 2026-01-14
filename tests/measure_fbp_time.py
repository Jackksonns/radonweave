# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.lib.format import open_memmap
from tqdm import tqdm

from src.data import get_standard_dataset
from src.data.fbp_dataset import FBPDataset


def _get_test_len(dataset):
    if hasattr(dataset, "get_len"):
        return int(dataset.get_len("test"))
    if hasattr(dataset, "get_test_len"):
        return int(dataset.get_test_len())
    raise AttributeError("dataset has no test length accessor")


def _hardware_info():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def main():
    parser = argparse.ArgumentParser(description="Measure test FBP cache generation time.")
    parser.add_argument("--config", default=None, help="Optional training config yaml.")
    parser.add_argument("--impl", default=None, help="Ray transform impl (astra_cuda/skimage/astra_cpu).")
    parser.add_argument("--filter-type", default=None, help="FBP filter type.")
    parser.add_argument("--frequency-scaling", type=float, default=None, help="FBP frequency scaling.")
    parser.add_argument("--fbp-cache-path", default=None, help="Path to cache_lodopab_test_fbp.npy.")
    parser.add_argument("--output", default=None, help="Output fbp_time.json path.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of test samples.")
    parser.add_argument("--flush-interval", type=int, default=200, help="Memmap flush interval.")
    parser.add_argument("--only-time", action="store_true",
                        help="Only measure time without writing cache file.")
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    impl = args.impl or (cfg.get("data", {}).get("impl") if cfg else None) or "astra_cuda"
    filter_type = args.filter_type or (cfg.get("reconstructor", {}).get("params", {}).get("filter_type")
                                       if cfg else None) or "Hann"
    frequency_scaling = args.frequency_scaling
    if frequency_scaling is None and cfg:
        frequency_scaling = cfg.get("reconstructor", {}).get("params", {}).get("frequency_scaling")
    if frequency_scaling is None:
        frequency_scaling = 1.0

    project_root = Path(__file__).resolve().parents[1]
    cache_path = Path(args.fbp_cache_path) if args.fbp_cache_path else project_root / "cache_lodopab_test_fbp.npy"
    output_path = Path(args.output) if args.output else project_root / "fbp_time.json"

    dataset = get_standard_dataset("lodopab", impl=impl)
    ray_trafo = dataset.get_ray_trafo(impl=impl)
    fbp_dataset = FBPDataset(dataset, ray_trafo, filter_type=filter_type, frequency_scaling=frequency_scaling)

    total_len = _get_test_len(dataset)
    num_samples = min(total_len, int(args.num_samples)) if args.num_samples is not None else total_len

    fbp_shape = fbp_dataset.space[0].shape
    fbp_dtype = fbp_dataset.space[0].dtype
    os.makedirs(cache_path.parent, exist_ok=True)
    cache_written = False
    memmap = None
    if not args.only_time:
        if cache_path.exists():
            print(f"[WARN] cache exists and will be overwritten: {cache_path}")
        os.makedirs(cache_path.parent, exist_ok=True)
        memmap = open_memmap(str(cache_path), mode="w+", dtype=fbp_dtype, shape=(num_samples,) + fbp_shape)

    times = []
    t_total_start = time.perf_counter()
    for idx in tqdm(range(num_samples), desc="Generating FBP cache", ncols=100):
        t0 = time.perf_counter()
        fbp, _ = fbp_dataset.get_sample(idx, part="test")
        if memmap is not None:
            memmap[idx] = np.asarray(fbp)
        times.append(time.perf_counter() - t0)
        if memmap is not None and args.flush_interval > 0 and (idx + 1) % args.flush_interval == 0:
            memmap.flush()
    if memmap is not None:
        del memmap
        cache_written = True
    total_time = time.perf_counter() - t_total_start

    stats = {
        "mean": float(np.mean(times)) if times else None,
        "std": float(np.std(times)) if times else None,
        "p95": float(np.percentile(times, 95)) if times else None,
        "num_samples": num_samples,
        "total_time": float(total_time),
        "cache_path": str(cache_path),
        "cache_written": cache_written,
        "only_time": bool(args.only_time),
        "impl": impl,
        "filter_type": filter_type,
        "frequency_scaling": float(frequency_scaling),
        "hardware": _hardware_info(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if cache_written:
        print(f"[DONE] wrote cache: {cache_path}")
    else:
        print("[DONE] cache not written (only timing).")
    print(f"[DONE] wrote timing: {output_path}")


if __name__ == "__main__":
    main()
