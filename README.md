# Radonweave

Research codebase for low-dose CT reconstruction/denoising on LoDoPaB-CT. It unifies FBP-input, sinogram-input, and GAN branches with shared training/eval tooling.

## Maintainer

- [Jackson ZHOU](https://github.com/Jackksonns)

## Features

- Integrated models: `fbp_unet`, `fbp_edcnn`, `redcnn`, `sist`, `iradonmap`, `attrdn`, `fedm2ct`, `dugan`.
- Unified trainer (`src/trainer.py`) and evaluator (`tests/test_eval.py`) with outputs under `results/` and `test_results/`.
- FBP cache and timing helpers (`src/data/fbp_dataset.py`, `tests/measure_fbp_time.py`).
- Metrics plotting (`tests/plot_metrics.py`).

## Project Structure

- `configs/`: training configs (`reconstructor` / `data` / `optim` / `trainer`)
- `src/`: training, model, and dataset implementations
- `tests/`: evaluation and utility scripts
- `test_results/`: evaluation outputs

## Setup

```bash
pip install -r requirements.txt
pip install dival lpips
````

* `astra-toolbox` + CUDA is used by the `astra_cuda` backend (fallbacks: `astra_cpu` / `skimage`).
* `lpips` is required only when computing LPIPS metrics.

## Acknowledgements

This project uses utilities from the [DIVAL](https://github.com/jleuschn/dival) project for dataset loading and some reconstructor logic. The overall design, models, and features in this repository are independently implemented. Thanks to the DIVAL authors and contributors for their work.

## Dataset (LoDoPaB-CT)

* The data path is configured in `~/.dival/config.json` under `lodopab_dataset/data_path`.
* Linux helper script:

```bash
bash src/get_lodopab.sh /path/to/lodopab
```

* Alternatively, use `LoDoPaBDataset` and follow the interactive download prompt.

## Quick Start

### Train

`src/trainer.py` currently hardcodes `config_path` at the bottom of the file. Update it to your desired `configs/*.yaml` first:

```bash
python src/trainer.py
```

### Evaluate

```bash
python -m tests.test_eval --config configs/train_fbp_fbpunet.yaml --weights /path/to/weights.pt
```

FBP cache and timing details are documented in the Notes section below.

### Plot Metrics

```bash
python -m tests.plot_metrics --metrics results/sist_stop18/metrics_history.json
```

## Model Coverage

* FBP-input: `fbp_unet`, `fbp_edcnn`, `redcnn`, `fedm2ct`, `dugan`
* Sinogram-input: `sist`, `iradonmap`, `attrdn`
* Experimental: `src/models/networks/dnst.py` defines DnST without a default config.

## Integrating a New Model

1. Add the network in `src/models/networks/` (or reuse an existing one).
2. Add a reconstructor in `src/models/reconstructors/` and register it with `@register("reconstructor", "<name>")`.
3. Export the reconstructor from `src/models/reconstructors/__init__.py`.
4. Create `configs/train_<name>.yaml` and set `reconstructor.name` plus parameters.
5. If the model needs special inputs (multi-input / FBP auxiliary inputs), extend inference logic in `tests/test_eval.py`.

# Radonweave Notes

## Test-Split Evaluation

Run `tests/test_eval.py` with model weights to evaluate on the test split (requires downloading the LoDoPaB test set and installing `lpips`).

* For `fbp_*` models that want to use the test-split FBP cache:

  * Add `--use-fbp-cache`
  * For the first run, you may add `--generate-fbp-cache`
* For non-`fbp_*` models (sinogram-input models), FBP cache is not needed:

  * Do not use `--use-fbp-cache` / `--generate-fbp-cache`

Examples (with an existing weights path):

```bash
python -m tests.test_eval --config configs/train_fbp_fbpunet.yaml --weights src/results/fbp_unet_stop61/final.pt
```

Example (using FBP cache):

```bash
python -m tests.test_eval --config configs/train_fbp_fbpunet.yaml --weights src/results/fbp_unet_stop61/final.pt --use-fbp-cache
```

Example (`fbp_edcnn`, verified):

```bash
python -m tests.test_eval --config configs/train_fbp_edcnn.yaml --weights src/results/fbp_edcnn_stop80/final.pt --use-fbp-cache
```

Example (`dugan`, verified):

```bash
python -m tests.test_eval --config configs/train_dugan.yaml \
  --weights src/results/dugan/best_loss.pt \
  --use-fbp-cache
```

### When to Use FBP Cache

* `fbp_*` models take **FBP-reconstructed images** as input:

  * If the cache exists and parameters match (`impl` / `filter_type` / `frequency_scaling`), use `--use-fbp-cache`.
  * If the cache is missing or you are unsure whether parameters match, it is recommended to delete the old cache and rebuild with `--generate-fbp-cache` (most reliable).
* Non-`fbp_*` models take **sinograms** as input:

  * Do not use `--use-fbp-cache` / `--generate-fbp-cache`.

## Inference Time Logging

`tests.test_eval` logs per-sample inference time and writes it to:

* `test_results/<model_name>/inference_time_<model_name>.txt` (or the directory set by `--output_dir`)

It includes `t_net_mean/std/p95`. If FBP cache is used, it also includes `t_pre_mean` and `t_total_mean`.

### FBP Cache Timing (t_pre)

For `fbp_*` models, first generate a timing profile:

```bash
python -m tests.measure_fbp_time --config configs/train_fbp_fbpunet.yaml
```

This produces:

* `cache_lodopab_test_fbp.npy`
* `fbp_time.json`

By default, both are written under the project root. Then set the following in the YAML:

```yaml
fbp_time_path: ./fbp_time.json
```

When evaluating with cache:

```bash
python -m tests.test_eval --config configs/train_fbp_fbpunet.yaml --weights <your_weights>.pt --use-fbp-cache
```

If you only want to measure time and do not want to overwrite the cache:

```bash
python -m tests.measure_fbp_time --config configs/train_fbp_fbpunet.yaml --only-time
```

### Reuse Conditions for fbp_time.json

`fbp_time.json` can be reused only if all of the following are the same:

* Same machine
* Same `impl` (`astra_cuda` / `skimage` / `astra_cpu`)
* Same `filter_type` and `frequency_scaling`
* The model is FBP-input (e.g., `fbp_unet`, `fbp_swinir`, `fbp_edcnn`)

## Workflow Summary

1. General test-split evaluation: run `tests/test_eval.py` with weights. Results are written to `test_results/<model>/`, including:

   * `metrics_summary.json`
   * `per_sample_metrics.json`
   * `inference_time_<model>.txt`

2. Timing for FBP-input models: run `tests.measure_fbp_time ...` to generate `fbp_time.json`, set `fbp_time_path: ./fbp_time.json` in the YAML, then evaluate with `--use-fbp-cache`.

3. Cache consistency: if parameters do not match, delete the old cache and rebuild with `--generate-fbp-cache`.

## Training Gets Killed: Causes and Mitigations (iradonmap)

### Explanation

* Disk expansion is not memory expansion: SSD affects storage only and does not increase container memory limits.
* Container memory has an upper bound (cgroup): when usage approaches `memory.max`, the process may be killed by SIGKILL.
* File cache counts as memory: when reading large HDF5 files from LoDoPaB, the OS page cache can accumulate and be counted against cgroup memory.
* GPU VRAM is unrelated here: CUDA OOM would show an error; this case is a CPU memory limit.

### Diagnostics

```bash
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

### Conservative Improvements (Without Reducing batch_size)

* Disable DataLoader `pin_memory` (already set to `false` for train/val in `train_iradonmap.yaml`).
* Explicitly set `impl: 'astra_cuda'` (already added in `train_iradonmap.yaml`).
* Drop page cache after reading HDF5 (`posix_fadvise(DONTNEED)`, implemented in `src/data/lodopab_dataset.py`, Linux-only; skipped on Windows).

Potential impact: `pin_memory: false` may slightly reduce host-to-GPU throughput, but is more stable than being killed and restarting.

## FedM2CT Integration Notes (Global SpatialNet Only)

* This repo integrates the official FedM2CT global SpatialNet branch (FBP-input) for fair comparison on LoDoPaB.
* It does not include the original federated/local sinogram branches or the mutual distillation/aggregation workflow.
* Config: `configs/train_fedm2ct.yaml`. FBP cache behavior is the same as `fbp_*` models.

Training command:

```bash
python trainer.py --config configs/train_fedm2ct.yaml
```

To support the full federated workflow (local sinograms + mutual distillation + aggregation), the Trainer must be extended to support multiple models/optimizers/clients. This is not implemented in the current version.

Reproduction clarification:

* This repo implements "FedM2CT-Global (SpatialNet only)": input is FBP images, and only the global SpatialNet branch is kept.
* LoDoPaB does not provide multi-center/protocol metadata or federation settings, so the cross-site sharing and meta-model aggregation of CPML/FMDL cannot be reproduced.
* This implementation is used only as an image-domain baseline and does not represent full federated FedM2CT performance.

## DUGAN Integration Notes (Full GAN Version)

* Config: `configs/train_dugan.yaml`. Default `trainer.gan_mode: true`. Uses LoDoPaB FBP input (cache can be used, similar to `fbp_*`).
* Model: generator + image/gradient discriminators, LS-GAN + CutMix CR, with an EMA generator used for validation/saving.

Training command:

```bash
python trainer.py --config configs/train_dugan.yaml
```

No impact on other models: the GAN branch is enabled only when `gan_mode` is true (either `trainer.gan_mode` or `recon.gan_mode`). Other models follow the original single-optimizer workflow.

Cache and evaluation: same as `fbp_unet`. Use the `generate_cache` / `use_fbp_cache` workflow, and evaluate via `tests/test_eval`.

## Evaluation Fairness Notes (Model Comparison)

* All models are evaluated via `tests/test_eval.py` on the LoDoPaB test split with the same metrics (PSNR/SSIM/RMSE/LPIPS/NPS, `data_range=1.0`) and identical sample indices.
* Input domain must match each model’s design for a fair comparison:

  * FBP-input models: `fbp_unet`, `fbp_edcnn`, `fbp_swinir`, `redcnn`, `fedm2ct`, `dugan`

    * Use FBP cache or on-the-fly FBP with the same `filter_type` / `frequency_scaling` across models.
  * Sinogram-input models: `sist`, `iradonmap`, `attrdn` (operate on raw sinograms; no FBP cache).
  * Image-domain model: `dnst` expects image-domain input (FBP images). If you evaluate it here, ensure the input is image-domain (e.g., FBP), otherwise results are not comparable.
* DUGAN differs from the official repo only in dataset adaptation: discriminator inputs are padded to handle LoDoPaB’s 362×362 size; generator/losses remain unchanged.
* AttrDN integration uses the attrdn-info AttRDN architecture:

  * The model denoises sinograms.
  * The loss is computed on sinograms (L1 + MS-SSIM) against forward-projected ground truth.
  * Evaluation metrics are computed on FBP-reconstructed images for comparability with other models.
  * To keep FBP output well-defined, training uses full sinograms (no patch cropping).
* AttrDN stability tweak: MS-SSIM in `src/models/networks/attrdn/losses.py` clamps intermediate values to `[1e-6, 1.0]` and applies `nan_to_num` on the final score. This prevents NaNs when sinogram-domain CS values become non-positive on LoDoPaB while preserving the original loss form.

## Trainer CLI Note

* The current `trainer.py` does not parse `--config` from the command line.
* It uses a hardcoded default at the bottom of the file: `configs/train_iradonmap.yaml`.
* If you run `python trainer.py --config ...`, it will still train `iradonmap` unless you update the script to read CLI arguments.

## Resume / Continue Training

Configure this in the YAML (if not set, it is disabled):

```yaml
train:
  resume_from: src/results/sist_stop18/final.pt
  resume_epoch: 19
```

Notes:

* `resume_from` loads model weights only; it does not restore optimizer/scheduler state.
* `resume_epoch` is the starting epoch for continued training (e.g., if `stop=18`, set it to `19`).

## Resume Training: Continue Curves and Early-Stopping State

If you want to continue the previous validation-loss curve and early-stopping counter, set the following in the YAML:

```yaml
train:
  resume_metrics_path: results/sist_stop18/metrics_history.json
```

Design rationale:

* `metrics_history.json` is the stable log source during training; reading it allows reuse of historical validation-loss curves.
* By replaying historical validation loss, it reconstructs `best_loss` and `es_wait`, making early stopping consistent with the state before resuming.
* This is optional; if not set, it does not affect the default training flow for other models.

## Plot Historical Loss Curves (No Retraining)

If training has finished but no plot was generated, you can draw it from `metrics_history.json`:

```bash
python -m tests.plot_metrics --metrics results/sist_stop18/metrics_history.json
```

## Test Image Saving (tests/test_eval.py)

* Only 5 test samples are saved by default: two from the first half, one middle, two from the second half.

  * If `--num_samples` is set, indices are picked within the first N samples.
* For FBP-input models (`fbp_*`, `redcnn`, `fedm2ct`, `dugan`):

  * Always save both `sinogram.png` (raw observation) and `fbp.png` (FBP reconstruction).
  * `observation.png` equals `fbp.png` when `--use-fbp-cache` is on; otherwise it equals `sinogram.png`.
* For non-FBP models: only `observation.png`, `reconstruction.png`, `ground_truth.png` are saved.

## References

```bibtex
@INPROCEEDINGS{edcnn,
  author={Liang, Tengfei and Jin, Yi and Li, Yidong and Wang, Tao},
  booktitle={2020 15th IEEE International Conference on Signal Processing (ICSP)},
  title={EDCNN: Edge enhancement-based Densely Connected Network with Compound Loss for Low-Dose CT Denoising},
  year={2020},
  volume={1},
  number={},
  pages={193-198},
  doi={10.1109/ICSP48669.2020.9320928}
}

@article{sist,
  title={Low-Dose CT Denoising via Sinogram Inner-Structure Transformer},
  author={Liutao Yang and Zhongnian Li and Rongjun Ge and Junyong Zhao and Haipeng Si and Daoqiang Zhang},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  volume={42},
  pages={910-921},
  url={https://api.semanticscholar.org/CorpusID:248006190}
}

@article{redcnn,
  author = {Chen, Hu and Zhang, Yi and Kalra, Mannudeep and Lin, Feng and Chen, Yang and Liao, Peixi and Zhou, Jiliu and Wang, Ge},
  year = {2017},
  month = {06},
  pages = {2524-2535},
  title = {Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network},
  volume = {36},
  journal = {IEEE Transactions on Medical Imaging},
  doi = {10.1109/TMI.2017.2715284}
}

@article{attrdn,
  title = {Sinogram denoising via attention residual dense convolutional neural network for low-dose computed tomography},
  volume = {32},
  issn = {2210-3147},
  url = {https://doi.org/10.1007/s41365-021-00874-2},
  doi = {10.1007/s41365-021-00874-2},
  number = {4},
  journal = {Nuclear Science and Techniques},
  author = {Ma, Yin-Jin and Ren, Yong and Feng, Peng and He, Peng and Guo, Xiao-Dong and Wei, Biao},
  month = apr,
  year = {2021},
  pages = {41}
}

@ARTICLE{fbpunet,
  author={Jin, Kyong Hwan and McCann, Michael T. and Froustey, Emmanuel and Unser, Michael},
  journal={IEEE Transactions on Image Processing},
  title={Deep Convolutional Neural Network for Inverse Problems in Imaging},
  year={2017},
  volume={26},
  number={9},
  pages={4509-4522},
  doi={10.1109/TIP.2017.2713099}
}

@ARTICLE{iradonmap,
  author={He, Ji and Wang, Yongbo and Ma, Jianhua},
  journal={IEEE Transactions on Medical Imaging},
  title={Radon Inversion via Deep Learning},
  year={2020},
  volume={39},
  number={6},
  pages={2076-2087},
  doi={10.1109/TMI.2020.2964266}
}

@ARTICLE{dugan,
  author={Huang, Zhizhong and Zhang, Junping and Zhang, Yi and Shan, Hongming},
  journal={IEEE Transactions on Instrumentation and Measurement},
  title={DU-GAN: Generative Adversarial Networks With Dual-Domain U-Net-Based Discriminators for Low-Dose CT Denoising},
  year={2022},
  volume={71},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2021.3128703}
}
```
