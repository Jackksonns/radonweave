# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
Quick checker for FBP cache files used by FBP-based reconstructors.

Usage
-----
python -m src.check_fbp_cache --config configs/train_fbp_fbpunet.yaml

What it does
------------
Reads the cache file paths from the given YAML config (`data.cache_files`),
normalizes them to (fbp_path, None) tuples, and for each existing file prints:
  - existence
  - file size on disk
  - numpy shape/dtype (mmap read)
If a file is missing or unreadable, it reports the error.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml
import numpy as np


def _normalize_cache_files(cache_files: dict) -> dict:
    """Ensure each part has a (fbp_path, None) tuple."""
    normalized = {}
    for part, files in (cache_files or {}).items():
        if isinstance(files, str):
            files = (files, None)
        elif isinstance(files, list):
            if len(files) == 1:
                files = (files[0], None)
            else:
                files = tuple(files[:2])
        elif isinstance(files, tuple):
            if len(files) == 1:
                files = (files[0], None)
        normalized[part] = files
    return normalized


def _pretty_size(bytes_val: int) -> str:
    gb = bytes_val / (1024 ** 3)
    mb = bytes_val / (1024 ** 2)
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


def check_cache(config_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    # If the cache is actually placed in the src directory at the project root, an additional base directory can be specified
    src_dir = project_root / "src"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})

    cache_files_cfg = data_cfg.get("cache_files") or {}
    cache_files = _normalize_cache_files(cache_files_cfg)

    print(f"[CONFIG] {config_path}")
    for part, files in cache_files.items():
        if files is None:
            continue
        fbp_path = files[0]
        if not fbp_path:
            continue
        fbp_path = Path(fbp_path)
        if not fbp_path.is_absolute():
            # First, parse from the relative project root directory; if it doesn't exist, then try the src directory
            candidate = (project_root / fbp_path).resolve()
            if candidate.exists():
                fbp_path = candidate
            else:
                fbp_path = (src_dir / fbp_path).resolve()
        print(f"\n[PART] {part}")
        print(f"  fbp_path : {fbp_path}")
        if not fbp_path.exists():
            print("  status   : MISSING")
            continue
        try:
            size = fbp_path.stat().st_size
            print(f"  size     : {size} bytes ({_pretty_size(size)})")
        except OSError as e:
            print(f"  size     : error ({e})")
        try:
            arr = np.load(fbp_path, mmap_mode="r")
            print(f"  shape    : {arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"  read     : failed ({e})")


def main():
    parser = argparse.ArgumentParser(description="Check FBP cache files from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_fbp_fbpunet.yaml"),
        help="Path to YAML config (default: configs/train_fbp_fbpunet.yaml)",
    )
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    check_cache(config_path)


if __name__ == "__main__":
    main()
