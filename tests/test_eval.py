# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

try:
    import lpips as lpips_lib
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

from src.data import get_standard_dataset
from src.data.fbp_dataset import FBPDataset, generate_fbp_cache_files, get_cached_fbp_dataset
from src.utils.registry import build
from src.utils.torch_utility import load_state_dict_convert_data_parallel


def compute_psnr(pred, target, data_range=1.0):
    mse = np.mean((pred - target) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(pred, target, data_range=1.0):
    if not SKIMAGE_OK:
        return None
    try:
        return float(ssim_fn(pred, target, data_range=data_range))
    except Exception:
        return None


def compute_rmse(pred, target):
    mse = np.mean((pred - target) ** 2, dtype=np.float64)
    return float(np.sqrt(mse))

def _to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return arr


def _save_image(arr, path):
    arr = _to_2d(arr)
    if arr.size == 0:
        return
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if not np.isfinite(arr_min) or not np.isfinite(arr_max):
        return
    if abs(arr_max - arr_min) < 1e-8:
        arr_uint8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_uint8 = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), arr_uint8)
    except Exception:
        np.save(str(path).replace(".png", ".npy"), arr)


def _pick_save_indices(total_len: int):
    if total_len <= 0:
        return []
    if total_len <= 5:
        return list(range(total_len))
    mid = total_len // 2
    first_2 = max(1, mid // 2)
    second_1 = mid + max(1, (total_len - 1 - mid) // 2)
    indices = [0, first_2, mid, second_1, total_len - 1]
    indices = sorted({i for i in indices if 0 <= i < total_len})
    if len(indices) < 5:
        for i in range(total_len):
            if i not in indices:
                indices.append(i)
                if len(indices) == 5:
                    break
        indices = sorted(indices)
    return indices


def _is_fbp_model(model_name):
    fbp_models = {
        "redcnn",
        "fedm2ct",
        "dugan",
    }
    return model_name.startswith("fbp_") or model_name in fbp_models


def _is_sist_model(model_name):
    return model_name == "sist"


def _get_test_len(dataset):
    if hasattr(dataset, "get_len"):
        return int(dataset.get_len("test"))
    if hasattr(dataset, "get_test_len"):
        return int(dataset.get_test_len())
    raise AttributeError("dataset has no test length accessor")


def _prepare_test_fbp_cache(dataset, ray_trafo, cache_path,
                            generate_cache, filter_type, frequency_scaling):
    cache_path = Path(cache_path)
    cache_files = {"test": (str(cache_path), None)}
    if generate_cache or (not cache_path.is_file()):
        generate_fbp_cache_files(
            dataset,
            ray_trafo,
            cache_files,
            size={"test": _get_test_len(dataset)},
            filter_type=filter_type,
            frequency_scaling=frequency_scaling,
        )
    if not cache_path.is_file():
        raise FileNotFoundError(f"FBP cache not found: {cache_path}")
    return get_cached_fbp_dataset(
        dataset,
        ray_trafo,
        cache_files,
        size={"test": _get_test_len(dataset)},
        filter_type=filter_type,
        frequency_scaling=frequency_scaling,
    )


def compute_nps(noise):
    fft = np.fft.fft2(noise)
    psd = np.abs(fft) ** 2 / float(noise.size)
    return np.fft.fftshift(psd)


def radial_profile(data):
    h, w = data.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)




def init_lpips(device, net="alex"):
    if not LPIPS_OK:
        raise RuntimeError("lpips is not available. Install with `pip install lpips`.")
    model = lpips_lib.LPIPS(net=net)
    model = model.to(device)
    model.eval()
    return model


def lpips_score(lpips_model, pred, target, device):
    pred_t = torch.from_numpy(pred).float().to(device)
    tgt_t = torch.from_numpy(target).float().to(device)
    if pred_t.ndim == 2:
        pred_t = pred_t[None, None]
        tgt_t = tgt_t[None, None]
    pred_t = pred_t * 2.0 - 1.0
    tgt_t = tgt_t * 2.0 - 1.0
    pred_t = pred_t.repeat(1, 3, 1, 1)
    tgt_t = tgt_t.repeat(1, 3, 1, 1)
    with torch.no_grad():
        val = lpips_model(pred_t, tgt_t)
    return float(val.item())


def build_reconstructor(cfg, dataset, impl):
    rc_cfg = cfg["reconstructor"]
    params = rc_cfg.get("params", {})
    ray_trafo = dataset.get_ray_trafo(impl=impl)
    params["op"] = ray_trafo
    params["ray_trafo"] = ray_trafo
    recon = build("reconstructor", rc_cfg["name"], **params)
    model = None
    if hasattr(recon, "build_model"):
        model = recon.build_model(cfg)
    if model is None:
        model = getattr(recon, "model", None)
    if model is None and hasattr(recon, "get_network"):
        model = recon.get_network()
        recon.model = model
    if model is None:
        raise RuntimeError("Failed to build model for reconstructor.")
    return recon


def infer_reconstruction(recon, model_name, observation, device, input_is_fbp=False):
    if isinstance(observation, (list, tuple)) and len(observation) == 2:
        sinogram, image_ld = observation
        sino_np = np.asarray(sinogram)
        img_np = np.asarray(image_ld)
        sino_t = torch.from_numpy(sino_np).float().to(device)
        img_t = torch.from_numpy(img_np).float().to(device)
        if sino_t.ndim == 2:
            sino_t = sino_t.unsqueeze(0)
        elif sino_t.ndim == 4 and sino_t.size(1) == 1:
            sino_t = sino_t.squeeze(1)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.ndim == 3:
            img_t = img_t.unsqueeze(1)
        with torch.no_grad():
            out = recon.model((sino_t, img_t))
        return out[0, 0].detach().cpu().numpy()
    if model_name.startswith("fbp_") and not input_is_fbp:
        if not hasattr(recon, "fbp_op"):
            raise RuntimeError("fbp_op not initialized; build_model must be called before inference.")
        fbp = recon.fbp_op(observation)
        inp = np.asarray(fbp)
    else:
        inp = np.asarray(observation)
    x = torch.from_numpy(inp).float().to(device)
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[:, None]
    with torch.no_grad():
        out = recon.model(x)
    return out[0, 0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained weights on LoDoPaB test set.")
    parser.add_argument("--config", required=True, help="Path to training config yaml.")
    parser.add_argument("--weights", required=True, help="Path to saved .pt weights.")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of test samples.")
    parser.add_argument("--output_dir", default=None, help="Output directory for metrics.")
    parser.add_argument("--lpips_net", default="alex", help="LPIPS backbone: alex, vgg, squeeze.")
    parser.add_argument("--use-fbp-cache", action="store_true", help="Use cached test FBPs for fbp_* models.")
    parser.add_argument("--generate-fbp-cache", action="store_true",
                        help="Generate test FBP cache if missing.")
    parser.add_argument("--fbp-cache-path", default=None,
                        help="Path to cache_lodopab_test_fbp.npy (default: project root).")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    impl = cfg.get("data", {}).get("impl", "astra_cuda")
    base_dataset = get_standard_dataset("lodopab", impl=impl)
    ray_trafo = base_dataset.get_ray_trafo(impl=impl)

    model_name = cfg.get("reconstructor", {}).get("name", "model")
    is_fbp = _is_fbp_model(model_name)
    is_sist = _is_sist_model(model_name)
    filter_type = cfg.get("reconstructor", {}).get("params", {}).get("filter_type", "Hann")
    frequency_scaling = cfg.get("reconstructor", {}).get("params", {}).get("frequency_scaling", 1.0)
    if args.use_fbp_cache and not (is_fbp or is_sist):
        print("[WARN] --use-fbp-cache ignored for non-fbp models.")
    use_fbp_cache = args.use_fbp_cache and is_fbp
    use_fbp_cache_sist = args.use_fbp_cache and is_sist
    eval_dataset = base_dataset
    if use_fbp_cache:
        default_cache = Path(__file__).resolve().parents[1] / "cache_lodopab_test_fbp.npy"
        cache_path = args.fbp_cache_path or str(default_cache)
        eval_dataset = _prepare_test_fbp_cache(
            base_dataset,
            ray_trafo,
            cache_path,
            args.generate_fbp_cache,
            filter_type,
            frequency_scaling,
        )
        print(f"[INFO] using test FBP cache: {cache_path}")
    fbp_dataset_for_save = None
    if is_fbp and not use_fbp_cache:
        fbp_dataset_for_save = FBPDataset(
            base_dataset,
            ray_trafo,
            filter_type=filter_type,
            frequency_scaling=frequency_scaling,
        )
    fbp_aux_dataset = None
    if is_sist:
        if use_fbp_cache_sist:
            default_cache = Path(__file__).resolve().parents[1] / "cache_lodopab_test_fbp.npy"
            cache_path = args.fbp_cache_path or str(default_cache)
            fbp_aux_dataset = _prepare_test_fbp_cache(
                base_dataset,
                ray_trafo,
                cache_path,
                args.generate_fbp_cache,
                filter_type,
                frequency_scaling,
            )
            print(f"[INFO] using test FBP cache (SIST): {cache_path}")
        else:
            fbp_aux_dataset = FBPDataset(
                base_dataset,
                ray_trafo,
                filter_type=filter_type,
                frequency_scaling=frequency_scaling,
            )
    out_dir = Path(args.output_dir) if args.output_dir else Path("./test_results") / model_name
    os.makedirs(out_dir, exist_ok=True)
    images_dir = out_dir / "images"
    os.makedirs(images_dir, exist_ok=True)

    recon = build_reconstructor(cfg, base_dataset, impl)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    recon.device = device
    recon.model = recon.model.to(device)

    try:
        state = torch.load(args.weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location=device)
    load_state_dict_convert_data_parallel(recon.model, state)
    recon.model.eval()

    if not LPIPS_OK:
        raise RuntimeError("lpips is not available. Install with `pip install lpips`.")
    lpips_model = init_lpips(device, args.lpips_net)

    total_len = _get_test_len(eval_dataset)
    if args.num_samples is not None:
        total_len = min(total_len, int(args.num_samples))
    save_indices = set(_pick_save_indices(total_len))

    psnrs, ssims, rmses, lpips_vals = [], [], [], []
    nps_sum = None
    per_sample = []
    inference_times = []

    for idx in tqdm(range(total_len), desc="Evaluating", ncols=100):
        observation, ground_truth = eval_dataset.get_sample(idx, part="test")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if is_sist:
            if fbp_aux_dataset is None:
                raise RuntimeError("SIST requires auxiliary FBP dataset for image_ld.")
            fbp_img, _ = fbp_aux_dataset.get_sample(idx, part="test")
            recon_np = infer_reconstruction(
                recon,
                model_name,
                (observation, fbp_img),
                device,
                input_is_fbp=use_fbp_cache,
            )
        else:
            recon_np = infer_reconstruction(recon, model_name, observation, device, input_is_fbp=use_fbp_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)
        gt_np = np.asarray(ground_truth)

        if idx in save_indices:
            sample_dir = images_dir / f"sample_{idx:04d}"
            os.makedirs(sample_dir, exist_ok=True)
            if is_fbp:
                sino_obs, _ = base_dataset.get_sample(idx, part="test")
                _save_image(sino_obs, sample_dir / "sinogram.png")
                fbp_img = observation if use_fbp_cache else None
                if fbp_img is None and fbp_dataset_for_save is not None:
                    fbp_img, _ = fbp_dataset_for_save.get_sample(idx, part="test")
                if fbp_img is not None:
                    _save_image(fbp_img, sample_dir / "fbp.png")
            _save_image(observation, sample_dir / "observation.png")
            _save_image(recon_np, sample_dir / "reconstruction.png")
            if ground_truth is not None:
                _save_image(gt_np, sample_dir / "ground_truth.png")

        psnr = compute_psnr(recon_np, gt_np, data_range=1.0)
        ssim = compute_ssim(recon_np, gt_np, data_range=1.0)
        rmse = compute_rmse(recon_np, gt_np)
        psnrs.append(psnr)
        if ssim is not None:
            ssims.append(ssim)
        rmses.append(rmse)

        noise = recon_np - gt_np
        nps = compute_nps(noise)
        nps_sum = nps if nps_sum is None else nps_sum + nps

        lpips_val = None
        if lpips_model is not None:
            lpips_val = lpips_score(lpips_model, recon_np, gt_np, device)
            lpips_vals.append(lpips_val)

        per_sample.append(
            {
                "index": idx,
                "psnr": float(psnr),
                "ssim": float(ssim) if ssim is not None else None,
                "rmse": float(rmse),
                "lpips": float(lpips_val) if lpips_val is not None else None,
            }
        )

    nps_mean = nps_sum / float(total_len) if nps_sum is not None else None
    if nps_mean is not None:
        np.save(out_dir / "nps_mean.npy", nps_mean)
        np.save(out_dir / "nps_radial.npy", radial_profile(nps_mean))

    summary = {
        "num_samples": total_len,
        "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
        "psnr_std": float(np.std(psnrs)) if psnrs else None,
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
        "ssim_std": float(np.std(ssims)) if ssims else None,
        "rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rmse_std": float(np.std(rmses)) if rmses else None,
        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
        "lpips_std": float(np.std(lpips_vals)) if lpips_vals else None,
        "nps_mean_value": float(np.mean(nps_mean)) if nps_mean is not None else None,
    }

    t_net_mean = float(np.mean(inference_times)) if inference_times else None
    t_net_std = float(np.std(inference_times)) if inference_times else None
    t_net_p95 = float(np.percentile(inference_times, 95)) if inference_times else None

    fbp_time_path = cfg.get("fbp_time_path") or cfg.get("data", {}).get("fbp_time_path")
    t_pre_mean = None
    if use_fbp_cache and fbp_time_path:
        fbp_time_path = Path(fbp_time_path)
        if not fbp_time_path.is_absolute():
            fbp_time_path = Path(args.config).resolve().parent / fbp_time_path
        if fbp_time_path.is_file():
            try:
                with open(fbp_time_path, "r", encoding="utf-8") as f:
                    fbp_time = json.load(f)
                t_pre_mean = float(fbp_time.get("mean")) if fbp_time.get("mean") is not None else None
            except Exception:
                t_pre_mean = None
        else:
            print(f"[WARN] fbp_time_path not found: {fbp_time_path}")
    elif use_fbp_cache and not fbp_time_path:
        print("[WARN] use_fbp_cache is on but fbp_time_path is not set in config.")

    t_total_mean = (t_net_mean + t_pre_mean) if (t_net_mean is not None and t_pre_mean is not None) else t_net_mean

    time_report = [
        f"num_samples: {total_len}",
        f"device: {device.type}",
        f"t_net_mean: {t_net_mean}",
        f"t_net_std: {t_net_std}",
        f"t_net_p95: {t_net_p95}",
        f"fbp_time_path: {str(fbp_time_path) if fbp_time_path else None}",
        f"t_pre_mean: {t_pre_mean}",
        f"t_total_mean: {t_total_mean}",
    ]
    with open(out_dir / f"inference_time_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(time_report) + "\n")

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
