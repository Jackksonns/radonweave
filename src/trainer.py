# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
# src/trainer.py
import os, time, math, json, signal
import sys
import inspect
from pathlib import Path

# Add the project root directory (the parent directory of src) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
    _PLOT_OK = True
except Exception:
    _PLOT_OK = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): return it  # fallback


try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

from src.utils.registry import build
from src.evaluator import save_image
from src.utils.torch_utility import load_state_dict_convert_data_parallel

import src.models.reconstructors as _  # noqa


def _to_np_img(t):
    """[B,1,H,W]或[1,H,W] → [H,W] np.float32"""
    if t.ndim == 4: t = t[0, 0]
    elif t.ndim == 3: t = t[0]
    return t.detach().float().cpu().numpy()


def _psnr_np(pred, target, data_range=1.0):
    mse = np.mean((pred - target) ** 2, dtype=np.float64)
    if mse <= 1e-12: return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _move_to_device(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device, non_blocking=non_blocking) for x in obj)
    return obj


def _batch_size(obj):
    if torch.is_tensor(obj):
        return int(obj.size(0)) if obj.ndim >= 1 else 1
    if isinstance(obj, (list, tuple)) and obj:
        return _batch_size(obj[0])
    return None


class _Channelify(nn.Module):
    """During training, it is adapted to be [B,1,H,W]. When saving, only the internal network weights are saved."""
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x):
        if isinstance(x, (list, tuple)) and not getattr(self.inner, "accepts_tuple", False):
            x = x[0]
        if isinstance(x, (list, tuple)):
            return self.inner(x)
        if not torch.is_tensor(x): x = torch.as_tensor(x)
        if x.ndim == 2:      x = x.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        elif x.ndim == 3:    x = x.unsqueeze(1)               # [B,H,W] -> [B,1,H,W]
        return self.inner(x)


def _save_params_pair(reconstructor, base_path: str):
    """
    Stored in pairs (with .pt, _hyper_params.json, and _meta.json)
    """
    pt_path = base_path + '.pt'
    hp_path = base_path + '.pt_hyper_params.json'
    meta_path = base_path + '.pt_meta.json'

    state = (reconstructor.model.state_dict()
             if hasattr(reconstructor, 'model') and reconstructor.model is not None
             else None)
    if state is None:
        raise RuntimeError("reconstructor.model is None. Check the init/build_model process.")

    torch.save(state, pt_path)

    with open(hp_path, 'w', encoding='utf-8') as f:
        json.dump(getattr(reconstructor, 'hyper_params', {}), f, indent=2, ensure_ascii=False)

    meta = {
        'torch': torch.__version__,
        'cuda': getattr(torch.version, 'cuda', None),
    }
    
    try:
        import dival
        meta['dival'] = getattr(dival, '__version__', None)
    except Exception:
        meta['dival'] = None
    try:
        import odl
        meta['odl'] = getattr(odl, '__version__', None)
    except Exception:
        meta['odl'] = None

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] wrote:\n  - {pt_path}\n  - {hp_path}\n  - {meta_path}")


def _save_train_images(train_model, train_loader, dataset, device, save_dir, num_samples=5):
    """
    After the training is completed, save the restored images of the model on the training set 
    Parameters
    train_model : nn.Module
    Packaged training model (Channelify) train_loader : DataLoader
    Training data loader device : torch.device
    Computing equipment save_dir : str
    Save directory (usually a folder with the same name as the logs) num_samples : int
    The number of samples to be saved, by default 5.
    """
    import os
    
    
    img_dir = os.path.join(save_dir, "train_images")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"\n[SAVE_TRAIN_IMAGES] Save the training set and restore the images to {img_dir}...")
    
    train_model.eval()
    saved_count = 0

    def _dataset_len(ds, part=None):
        if hasattr(ds, "get_len"):
            try:
                return int(ds.get_len(part))
            except TypeError:
                return int(ds.get_len())
        if hasattr(ds, "__len__"):
            return len(ds)
        return 0

    def _dataset_sample(ds, idx, part=None):
        if hasattr(ds, "get_sample"):
            try:
                return ds.get_sample(idx, part=part)
            except TypeError:
                return ds.get_sample(idx)
        return ds[idx]

    def _as_tensor(arr):
        if torch.is_tensor(arr):
            return arr
        return torch.from_numpy(np.asarray(arr))

    def _ensure_batch(t):
        if t.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        if t.ndim == 3:
            return t.unsqueeze(0)
        return t

    def _dataset_len(ds, part=None):
        if hasattr(ds, "get_len"):
            try:
                return int(ds.get_len(part))
            except TypeError:
                return int(ds.get_len())
        if hasattr(ds, "__len__"):
            return len(ds)
        return 0

    def _dataset_sample(ds, idx, part=None):
        if hasattr(ds, "get_sample"):
            try:
                return ds.get_sample(idx, part=part)
            except TypeError:
                return ds.get_sample(idx)
        return ds[idx]

    def _as_tensor(arr):
        if torch.is_tensor(arr):
            return arr
        return torch.from_numpy(np.asarray(arr))

    def _ensure_batch(t):
        if t.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        if t.ndim == 3:
            return t.unsqueeze(0)
        return t
    
    
    total_len = _dataset_len(dataset, "train")
    indices = list(range(min(num_samples, total_len)))

    with torch.no_grad():
        for idx in indices:
           
            x_arr, y_arr = _dataset_sample(dataset, idx, part='train')
            if isinstance(x_arr, (list, tuple)):
                x_sample = tuple(_ensure_batch(_as_tensor(a).float()).to(device) for a in x_arr)
                x_np = _to_np_img(x_sample[0])
                x_aux_np = _to_np_img(x_sample[1]) if len(x_sample) > 1 else None
            else:
                x_sample = _ensure_batch(_as_tensor(x_arr).float()).to(device)
                x_np = _to_np_img(x_sample)
                x_aux_np = None
            y_sample = _ensure_batch(_as_tensor(y_arr).float()).to(device)

            
            with torch.amp.autocast('cuda', enabled=True):
                y_hat = train_model(x_sample)

            
            y_hat_np = _to_np_img(y_hat)
            y_np = _to_np_img(y_sample)

           
            psnr = _psnr_np(y_hat_np, y_np, data_range=1.0)

            sample_dir = os.path.join(img_dir, f'sample_{saved_count:03d}')
            os.makedirs(sample_dir, exist_ok=True)

            try:
                save_image(x_np, os.path.join(sample_dir, 'observation.png'), cmap='gray')
                if x_aux_np is not None:
                    save_image(x_aux_np, os.path.join(sample_dir, 'fbp.png'), cmap='gray')
                save_image(y_hat_np, os.path.join(sample_dir, 'reconstruction.png'), cmap='gray')
                save_image(y_np, os.path.join(sample_dir, 'ground_truth.png'), cmap='gray')

                print(f"  ✓ sample_{saved_count:03d} saved (PSNR: {psnr:.2f})")
            except Exception as e:
                print(f"  ✗ sample_{saved_count:03d} 保存失败: {e}")

            saved_count += 1

    print(f"[SAVE_TRAIN_IMAGES] 共保存 {saved_count} 个样本的图像到 {img_dir}")


def _save_val_images(train_model, val_loader, dataset, device, save_dir, num_samples=5):
    """
    After the training is completed, save the model's restored images on the validation set. 
    Parameters
    train_model : nn.Module
    Packaged training model (Channelify) val_loader : DataLoader
    Verification data loader device : torch.device
    Computing equipment save_dir : str
    Save directory (usually a folder with the same name as the logs) num_samples : int
    The number of samples to be saved, by default 5.
    """
    import os
    
    # 创建保存目录
    img_dir = os.path.join(save_dir, "val_images")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"\n[SAVE_VAL_IMAGES] 保存验证集恢复图像到 {img_dir}...")
    
    train_model.eval()
    saved_count = 0
    
    def _dataset_len(ds, part=None):
        if hasattr(ds, "get_len"):
            try:
                return int(ds.get_len(part))
            except TypeError:
                return int(ds.get_len())
        if hasattr(ds, "__len__"):
            return len(ds)
        return 0

    def _dataset_sample(ds, idx, part=None):
        if hasattr(ds, "get_sample"):
            try:
                return ds.get_sample(idx, part=part)
            except TypeError:
                return ds.get_sample(idx)
        return ds[idx]

    def _as_tensor(arr):
        if torch.is_tensor(arr):
            return arr
        return torch.from_numpy(np.asarray(arr))

    def _ensure_batch(t):
        if t.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        if t.ndim == 3:
            return t.unsqueeze(0)
        return t

    total_len = _dataset_len(dataset, "validation")
    indices = list(range(min(num_samples, total_len)))

    with torch.no_grad():
        for idx in indices:
            x_arr, y_arr = _dataset_sample(dataset, idx, part='validation')
            if isinstance(x_arr, (list, tuple)):
                x_sample = tuple(_ensure_batch(_as_tensor(a).float()).to(device) for a in x_arr)
                x_np = _to_np_img(x_sample[0])
                x_aux_np = _to_np_img(x_sample[1]) if len(x_sample) > 1 else None
            else:
                x_sample = _ensure_batch(_as_tensor(x_arr).float()).to(device)
                x_np = _to_np_img(x_sample)
                x_aux_np = None
            y_sample = _ensure_batch(_as_tensor(y_arr).float()).to(device)

            with torch.amp.autocast('cuda', enabled=True):
                y_hat = train_model(x_sample)

            y_hat_np = _to_np_img(y_hat)
            y_np = _to_np_img(y_sample)

            psnr = _psnr_np(y_hat_np, y_np, data_range=1.0)

            sample_dir = os.path.join(img_dir, f'sample_{saved_count:03d}')
            os.makedirs(sample_dir, exist_ok=True)

            try:
                save_image(x_np, os.path.join(sample_dir, 'observation.png'), cmap='gray')
                if x_aux_np is not None:
                    save_image(x_aux_np, os.path.join(sample_dir, 'fbp.png'), cmap='gray')
                save_image(y_hat_np, os.path.join(sample_dir, 'reconstruction.png'), cmap='gray')
                save_image(y_np, os.path.join(sample_dir, 'ground_truth.png'), cmap='gray')

                print(f"  ✓ sample_{saved_count:03d} saved (PSNR: {psnr:.2f})")
            except Exception as e:
                print(f"  ✗ sample_{saved_count:03d} 保存失败: {e}")

            saved_count += 1

    print(f"[SAVE_VAL_IMAGES] A total of {saved_count} samples' images have been saved to {img_dir}")


class Trainer:
    """
    Unified Trainer:
    - Build the reconstructor using Registry based on cfg['reconstructor'];
    - Utilize tqdm, AMP, CosineAnnealingLR, PSNR/SSIM verification, and save the best model pairs.
    """
    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg["trainer"].get("device", "cpu"))
        self.config_path = config_path

    def fit(self):
        rc_cfg = self.cfg["reconstructor"]
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["trainer"]
        optim_cfg = self.cfg["optim"]
        model_name = rc_cfg.get("name", "model")
        epochs = int(train_cfg.get("epochs", 60))
        start_epoch = int(train_cfg.get("resume_epoch", 1) or 1)
        if start_epoch < 1:
            start_epoch = 1
        if start_epoch > epochs:
            print(f"[RESUME] resume_epoch={start_epoch} > epochs={epochs}, set to {epochs}.")
            start_epoch = epochs
        gan_mode = bool(train_cfg.get("gan_mode", False))

        from src.data import get_standard_dataset
        impl = data_cfg.get("impl", "astra_cuda")
        dataset = get_standard_dataset('lodopab', impl=impl)
        ray_trafo = dataset.get_ray_trafo(impl=impl)
        
        params = rc_cfg.get("params", {})
        # Simultaneously pass op and ray_trafo to ensure compatibility with different reconstructor implementations
        params['op'] = ray_trafo
        params['ray_trafo'] = ray_trafo
        recon = build("reconstructor", rc_cfg["name"], **params)
        gan_mode = gan_mode or getattr(recon, "gan_mode", False)

        sig = inspect.signature(recon.build_dataloaders)
        if 'dataset' in sig.parameters or 'ray_trafo' in sig.parameters:
            train_loader, val_loader = recon.build_dataloaders(
                data_cfg, dataset=dataset, ray_trafo=ray_trafo)
        else:
            train_loader, val_loader = recon.build_dataloaders(data_cfg)

        model = recon.build_model(self.cfg)
        if model is None:
            raise RuntimeError("The build_model() function must return an nn.Module and configure the recon.model.")
        model = model.to(self.device)

        resume_path = train_cfg.get("resume_from")
        if resume_path:
            candidates = []
            if os.path.isabs(resume_path):
                candidates = [resume_path]
            else:
                candidates = [
                    resume_path,
                    os.path.join(project_root, resume_path),
                    os.path.join(current_dir, resume_path),
                ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resume_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f"resume_from not found: {resume_path}\n"
                    f"Tried: {', '.join(candidates)}"
                )
            state = torch.load(resume_path, map_location=self.device)
            state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
            load_state_dict_convert_data_parallel(model, state_dict)
            print(f"[RESUME] loaded weights from {resume_path}")
        optim_params = optim_cfg.get("params", {})
        base_lr = float(optim_params.get("lr", 2e-4))
        weight_decay = float(optim_params.get("weight_decay", 0.0))
        optimizer = None

        if gan_mode and hasattr(recon, "gan_setup"):
            recon.gan_setup(self.device, optim_cfg)
            train_model = _Channelify(recon.get_eval_model(use_ema=True)).to(self.device)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
            train_model = _Channelify(model).to(self.device)

        # 5) 损失、AMP、Cosine LR
        if getattr(recon, "loss_function", None) is not None:
            criterion = recon.loss_function
        elif hasattr(recon, "_get_loss_fn"):
            criterion = recon._get_loss_fn()
        elif hasattr(recon, "get_loss_fn"):
            criterion = recon.get_loss_fn()
        else:
            criterion = nn.MSELoss() # If you need to change it to something else, you can put it in recon or cfg['loss']
        use_amp = bool(train_cfg.get("amp", True))  
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp) 

        steps_per_epoch = len(train_loader)
        max_train_steps = train_cfg.get("max_train_steps_per_epoch")
        if max_train_steps:
            steps_per_epoch = min(steps_per_epoch, int(max_train_steps))
        lr_min = float(optim_params.get("lr_min", 1e-5))
        scheduler = None
        scheduler_name = str(optim_cfg.get("scheduler", "cosine")).lower()
        if (not gan_mode) and scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr_min
            )

        es_cfg = train_cfg.get("early_stopping", {})
        es_patience = int(es_cfg.get("patience", 0) or 0)
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        es_min_rel = float(es_cfg.get("min_relative_delta", 5e-4))
        es_monitor = str(es_cfg.get("monitor", "val_loss"))
        es_wait = 0
        early_stop_triggered = False
        stop_epoch = None
        best_ckpt_path = train_cfg.get("save_best_base") or str(Path("./results") / model_name / "best_loss")
        best_loss_seen = float('inf')
        best_loss_for_es = float('inf')
        if best_ckpt_path:
            os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
        val_loss_history = []
        metrics_history = []
        resume_history_path = train_cfg.get("resume_metrics_path")
        if resume_history_path:
            candidates = []
            if os.path.isabs(resume_history_path):
                candidates = [resume_history_path]
            else:
                candidates = [
                    resume_history_path,
                    os.path.join(project_root, resume_history_path),
                    os.path.join(current_dir, resume_history_path),
                ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resume_history_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f"resume_metrics_path not found: {resume_history_path}\n"
                    f"Tried: {', '.join(candidates)}"
                )
            try:
                with open(resume_history_path, "r", encoding="utf-8") as f:
                    metrics_history = json.load(f) or []
                for item in metrics_history:
                    if isinstance(item, dict) and "epoch" in item and "val_loss" in item:
                        val_loss_history.append((int(item["epoch"]), float(item["val_loss"])))
                if val_loss_history:
                    ordered = sorted(val_loss_history, key=lambda x: x[0])
                    best_loss_seen = float("inf")
                    best_loss_for_es = float("inf")
                    es_wait = 0
                    for _, v in ordered:
                        if not np.isfinite(v):
                            continue
                        if best_loss_seen == float("inf"):
                            best_loss_seen = v
                            best_loss_for_es = best_loss_seen
                            es_wait = 0
                            continue
                        rel_improve = (best_loss_seen - v) / max(abs(best_loss_seen), 1e-12)
                        if (v < best_loss_seen and rel_improve >= es_min_rel) or (v < best_loss_seen - es_min_delta):
                            best_loss_seen = v
                            best_loss_for_es = best_loss_seen
                            es_wait = 0
                        else:
                            es_wait += 1
                print(f"[RESUME] loaded metrics history from {resume_history_path}")
            except Exception as exc:
                print(f"[RESUME] failed to load resume_metrics_path: {exc}")

        print(f"Device: {self.device} | AMP: {use_amp} | Steps/epoch: {steps_per_epoch} | "
              f"Cosine(eta_min={lr_min}) | SSIM: {'on' if SKIMAGE_OK else 'off'}")

        global_step = 0
        interrupted = False
        current_epoch = None

        def _signal_handler(signum, frame):
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            pass

        try:
            for epoch in range(start_epoch, epochs + 1):
                current_epoch = epoch
                
                train_model.train()
                running = 0.0
                pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), ncols=100,
                            desc=f"Train epoch {epoch}/{epochs}")
                for it, (x, y) in pbar:
                    if max_train_steps and it > max_train_steps:
                        break
                    t0 = time.time()
                    if gan_mode:
                        loss_dict = recon.gan_training_step((x, y), global_step + 1, scaler, use_amp)
                        loss_val = loss_dict.get("loss_g_total", torch.tensor(0.0))
                        if torch.is_tensor(loss_val):
                            loss_val = float(loss_val.detach().cpu())
                        running += loss_val
                        global_step += 1
                        iter_t = time.time() - t0
                        bs = _batch_size(x)
                        speed = (bs / iter_t) if (iter_t > 0 and bs is not None) else 0.0
                        pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{(running/it):.4f}",
                                         it_s=f"{speed:.2f}")
                    else:
                        x = _move_to_device(x, self.device, non_blocking=True)
                        y = _move_to_device(y, self.device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)

                        with torch.amp.autocast('cuda', enabled=use_amp):
                            y_hat = train_model(x)
                            y_hat = torch.nan_to_num(y_hat)
                            y = torch.nan_to_num(y)
                            loss = criterion(y_hat, y)

                        scaler.scale(loss).backward()
                        step_before = getattr(optimizer, "_step_count", 0)
                        scaler.step(optimizer)
                        scaler.update()
                        if scheduler is not None and getattr(optimizer, "_step_count", 0) > step_before:
                            scheduler.step()

                        running += loss.item()
                        global_step += 1

                        iter_t = time.time() - t0
                        lr_now = optimizer.param_groups[0]['lr']
                        bs = _batch_size(x)
                        speed = (bs / iter_t) if (iter_t > 0 and bs is not None) else 0.0
                        pbar.set_postfix(loss=f"{loss.item():.4f}",
                                         avg=f"{(running/it):.4f}",
                                         lr=f"{lr_now:.1e}",
                                         it_s=f"{speed:.2f}")

                used_train = min(len(train_loader), max_train_steps or len(train_loader))
                train_loss = running / used_train if used_train > 0 else 0.0

                
                train_model.eval()
                val_running = 0.0
                psnrs, ssims = [], []
                max_val_steps = train_cfg.get("max_val_steps")
                used_val = 0
                with torch.no_grad():
                    vpbar = tqdm(enumerate(val_loader, 1), total=len(val_loader), ncols=100,
                                 desc=f"Valid epoch {epoch}/{epochs}")
                    for vit, (x, y) in vpbar:
                        if max_val_steps and vit > max_val_steps:
                            break
                        x = _move_to_device(x, self.device, non_blocking=True)
                        y = _move_to_device(y, self.device, non_blocking=True)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            if gan_mode:
                                y_hat = recon.gan_infer(x, use_ema=True)
                            else:
                                y_hat = train_model(x)
                            y_hat = torch.nan_to_num(y_hat)
                            y = torch.nan_to_num(y)
                            vloss = criterion(y_hat, y)
                        val_running += vloss.item()
                        used_val += 1

                        y_np    = _to_np_img(y)
                        yhat_np = _to_np_img(y_hat)
                        psnrs.append(_psnr_np(yhat_np, y_np, data_range=1.0))   
                        if SKIMAGE_OK:
                            try:
                                ssims.append(ssim_fn(y_np, yhat_np, data_range=1.0))
                            except Exception:
                                pass
                val_loss = val_running / max(1, used_val)
                val_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
                val_ssim = float(np.mean(ssims)) if ssims else float('nan')
                val_loss_history.append((epoch, val_loss))
                train_avg = running / max(1, steps_per_epoch)
                metrics_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(train_avg),
                        "val_loss": float(val_loss),
                        "val_psnr": float(val_psnr) if np.isfinite(val_psnr) else None,
                        "val_ssim": float(val_ssim) if np.isfinite(val_ssim) else None,
                    }
                )
                print(f"[Epoch {epoch}/{epochs}] "
                      f"train_avg={train_avg:.6f}  "
                      f"val_avg={val_loss:.6f}  val_PSNR={val_psnr:.2f}  val_SSIM={val_ssim:.3f}")

                #Early Stopping
                improved = False
                if es_monitor.lower() == "val_loss":
                    # Relative Improvement Percentage
                    if np.isfinite(val_loss) and np.isfinite(best_loss_seen):
                        rel_improve = (best_loss_seen - val_loss) / max(abs(best_loss_seen), 1e-12)
                    else:
                        rel_improve = float('inf')
                    # First time or meeting the relative improvement threshold / absolute min_delta
                    if (best_loss_seen == float('inf')) or (val_loss < best_loss_seen and rel_improve >= es_min_rel) or (val_loss < best_loss_seen - es_min_delta):
                        improved = True
                        best_loss_seen = val_loss
                        best_loss_for_es = best_loss_seen
                        es_wait = 0
                        if best_ckpt_path:
                            _save_params_pair(recon, best_ckpt_path)
                        print(f"[EarlyStop] New best val_loss={val_loss:.6f} (rel_improve={rel_improve:.6e}), saved to {best_ckpt_path}")
                    else:
                        es_wait += 1
                else:
                    es_wait += 1

                if scheduler is not None:
                    scheduler.step()

                if es_patience > 0 and es_wait >= es_patience:
                    print(f"[EARLY STOP] monitor={es_monitor} | best={best_loss_for_es:.6f} "
                          f"| no improvement in {es_patience} epochs (min_delta={es_min_delta})")
                    early_stop_triggered = True
                    stop_epoch = epoch
                    break
        except KeyboardInterrupt:
            interrupted = True
            stop_epoch = current_epoch or start_epoch
            print(f"\n[INTERRUPT] Training interrupted at epoch {stop_epoch}. Saving current state...")

        # If the early stopping has not been triggered, record the last epoch as the termination point
        if stop_epoch is None:
            stop_epoch = epochs

        #Training completed. The termination point saves the current model and images.
        if interrupted:
            save_dir = Path("./results") / f"{model_name}_interrupt{stop_epoch}"
            final_base = save_dir / "interrupt"
        else:
            save_dir = Path("./results") / f"{model_name}_stop{stop_epoch}"
            final_base = save_dir / "final"
        os.makedirs(save_dir, exist_ok=True)
        _save_params_pair(recon, str(final_base))
        final_info = {
            "stop_epoch": stop_epoch,
            "early_stop": early_stop_triggered,
            "interrupted": interrupted,
            "monitor": es_monitor,
            "best_loss_for_es": best_loss_for_es,
        }
        if metrics_history:
            final_info["final_metrics"] = metrics_history[-1]

        # Save the training/evaluation metrics for each epoch
        try:
            metrics_path = save_dir / "metrics_history.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ! failed to write metrics_history.json: {e}")

        num_train_samples = train_cfg.get("num_train_images_to_save", 5)
        save_train_dataset = getattr(getattr(train_loader, "dataset", None), "dataset", None) or getattr(train_loader, "dataset", dataset)
        _save_train_images(train_model, train_loader, save_train_dataset, self.device, str(save_dir), num_samples=num_train_samples)
        
        num_val_samples = train_cfg.get("num_val_images_to_save", 5)
        save_val_dataset = getattr(getattr(val_loader, "dataset", None), "dataset", None) or getattr(val_loader, "dataset", dataset)
        _save_val_images(train_model, val_loader, save_val_dataset, self.device, str(save_dir), num_samples=num_val_samples)

        if _PLOT_OK and val_loss_history:
            plot_dir = Path("./plot") / model_name
            os.makedirs(plot_dir, exist_ok=True)
            epochs_list, vloss_list = zip(*val_loss_history)
            plt.figure()
            plt.plot(epochs_list, vloss_list, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Val Loss")
            plt.title(f"Val Loss Curve - {model_name}")
            plt.grid(True)
            plt.tight_layout()
            out_path = plot_dir / "val_loss.png"
            try:
                plt.savefig(out_path)
                print(f"[PLOT] saved val_loss curve to {out_path}")
            except Exception as e:
                print(f"[PLOT] failed to save val_loss curve: {e}")
            plt.close()
        elif not _PLOT_OK:
            print("[PLOT] matplotlib not available; skipped val_loss curve.")
        try:
            final_info_path = save_dir / "final_info.json"
            with open(final_info_path, "w", encoding="utf-8") as f:
                json.dump(final_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ! failed to write final_info.json: {e}")


import yaml

if __name__ == "__main__":
    config_path = os.path.join(project_root, "configs/train_sist.yaml")
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    Trainer(cfg, config_path=config_path).fit()
