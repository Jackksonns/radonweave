# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import os
import argparse
import importlib
from json import JSONDecodeError

# 安全导入类
def import_by_path(path: str):
    mod, cls = path.rsplit('.', 1)
    return getattr(importlib.import_module(mod), cls)

# 统一 measure 名称大小写
def _normalize_measures(names):
    """
    【防坑】DIVαL 里 measure 的 short_name 通常是小写（'psnr','ssim','rmse','mse'）。
    用户命令行里常写大写（'PSNR'），这里统一转小写，避免找不到注册指标。
    """
    return [n.lower() for n in names]

# 兼容 get_data_pairs 的新旧签名
def _get_test_pairs_compat(ds, part: str, start: int, num: int):
    """
    【兼容不同 DIVαL 版本的 Dataset.get_data_pairs 参数签名】
    - 旧版: get_data_pairs(part, start, num)
    - 新版: get_data_pairs(part, num)  # 没有 start
    """
    try:
        return ds.get_data_pairs(part, start, num)
    except TypeError:
        pairs = ds.get_data_pairs(part, num)
        if start:
            # 既兼容可切片的序列，也兼容迭代器
            try:
                end = (start + num) if num is not None else None
                return pairs[start:end]
            except TypeError:
                from itertools import islice
                stop = None if num is None else start + num
                return list(islice(pairs, start, stop))
        else:
            return pairs

# 剥掉常见的权重 key 前缀
def _strip_prefixes(state_dict):
    """
    【关键】你训练脚本用了 Channelify(model) 包装并保存了 model.state_dict()，
    会导致权重 key 前缀多出 'inner.'；DataParallel 训练还可能有 'module.'。
    这里统一剥掉两种前缀，避免大量 missing/unexpected。
    """
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('inner.'):
            nk = nk[len('inner.'):]
        new_state[nk] = v
    return new_state