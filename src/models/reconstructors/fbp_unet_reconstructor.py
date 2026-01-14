# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
# fbp_unet_reconstructor.py
from warnings import warn
from copy import deepcopy
import torch, numpy as np, torch.nn as nn
from odl.tomo import fbp_op

from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.unet import UNet
from src.data.fbp_dataset import (
    FBPDataset,
    generate_fbp_cache_files,
    get_cached_fbp_dataset,
)
from src.utils.registry import register

# 可选复合损失（感知 + 像素）；如未装好 torchvision 权重，建议先用 L1
try:
    from src.utils.compound_loss import CompoundLoss
    _COMPOUND_OK = True
except Exception:
    _COMPOUND_OK = False


# 注册FBP UNet重建器
@register("reconstructor", "fbp_unet")
class FBPUNetReconstructor(StandardLearnedReconstructor):
    """
    CT 重建：FBP 后接 UNet 做去噪/去伪影（与 FBP+EDCNN/SwinIR 同构）
    
    References
    ----------
    .. [1] K. H. Jin, M. T. McCann, E. Froustey, et al., 2017,
           "Deep Convolutional Neural Network for Inverse Problems in Imaging".
           IEEE Transactions on Image Processing.
           `doi:10.1109/TIP.2017.2713099
           <https://doi.org/10.1109/TIP.2017.2713099>`_
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        # —— UNet 结构超参（保持分辨率不变，单通道进出）——
        'scales':            {'default': 5, 'retrain': True},
        'skip_channels':     {'default': 4, 'retrain': True},
        'channels':          {'default': (32, 32, 64, 64, 128, 128), 'retrain': True},

        # —— FBP 超参（与 FBPEDCNN / FBPSwinIR 一致）——
        'filter_type':       {'default': 'Hann',  'retrain': True},
        'frequency_scaling': {'default': 1.0,     'retrain': True},

        # —— 优化器/学习率（与现有脚本一致）——
        'lr':          {'default': 2e-4, 'retrain': True},
        'scheduler':   {'default': 'cosine', 'choices': ['base','cosine'], 'retrain': True},
        'lr_min':      {'default': 1e-4, 'retrain': True},

        # —— 损失相关 —— 
        'use_compound_loss': {'default': False, 'retrain': True},   # 是否使用 CompoundLoss
        'compound_mse_weight': {'default': 1.0,  'retrain': True},
        'compound_resnet_weight': {'default': 0.01, 'retrain': True},

        # 输出是否加 Sigmoid（CT 通常不需要）
        'use_sigmoid': {'default': False, 'retrain': True},
        
        # 初始化bias为零
        'init_bias_zero': {'default': True, 'retrain': True},
    })

    def __init__(self, ray_trafo,
                 allow_multiple_workers_without_random_access=False, **kwargs):
        self.allow_multiple_workers_without_random_access = (
            allow_multiple_workers_without_random_access)
        
        # 从 kwargs 中提取 in_ch 和 out_ch 参数，避免传递给父类
        in_ch = kwargs.pop('in_ch', 1)
        out_ch = kwargs.pop('out_ch', 1)
        
        # 提取FBPUNetReconstructor特有的参数，避免传递给父类
        unet_params = {}
        for param in ['scales', 'skip_channels', 'channels', 'filter_type', 
                      'frequency_scaling', 'lr', 'scheduler', 'lr_min',
                      'use_compound_loss', 'compound_mse_weight', 
                      'compound_resnet_weight', 'use_sigmoid', 'init_bias_zero']:
            if param in kwargs:
                unet_params[param] = kwargs.pop(param)
        
        # 处理 op / ray_trafo，避免重复传参导致 multiple values for 'op'
        op = kwargs.pop('op', ray_trafo)
        kwargs.pop('ray_trafo', None)

        # 调用父类初始化，确保hyper_params被正确初始化
        super().__init__(op, **kwargs)

        # 设置in_ch和out_ch属性
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # 将提取的参数添加到hyper_params中
        for param, value in unet_params.items():
            self.hyper_params[param] = value

    # 训练入口：尽量沿用 FBPEDCNN/FBPSwinIR 的做法（有缓存就用，没有就 on-the-fly 计算）
    def train(self, dataset):
        try:
            fbp_dataset = dataset.fbp_dataset
        except AttributeError:
            warn('No cached FBP dataset: computing FBPs on the fly. '
                 'Consider precomputing with `generate_fbp_cache_files(...)` '
                 'and attaching via `dataset.fbp_dataset = get_cached_fbp_dataset(...)`.')
            fbp_dataset = FBPDataset(
                dataset, self.non_normed_op,
                filter_type=self.filter_type,
                frequency_scaling=self.frequency_scaling
            )

        if not fbp_dataset.supports_random_access():
            if not self.allow_multiple_workers_without_random_access:
                if self.num_data_loader_workers > 1:
                    warn('Overriding data loader workers to 1 for non-random-access dataset. '
                         'Pass allow_multiple_workers_without_random_access=True to override.')
                self.num_data_loader_workers = min(self.num_data_loader_workers, 1)

        super().train(fbp_dataset)

    def init_model(self):
        # FBP 运算符（ODL）
        self.fbp_op = fbp_op(self.op, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)

        # —— 构建 UNet：同分辨率复原，使用配置中的通道数 —— 
        net = UNet(in_ch=int(self.in_ch), out_ch=int(self.out_ch),
                   channels=self.channels[:self.scales],
                   skip_channels=[self.skip_channels] * (self.scales),
                   use_sigmoid=self.use_sigmoid)
        
        # 初始化bias为零
        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            net.apply(weights_init)

        self.model = net

        # —— 损失函数 —— 
        if bool(self.use_compound_loss) and _COMPOUND_OK:
            self.loss_function = CompoundLoss(
                mse_weight=float(self.compound_mse_weight),
                resnet_weight=float(self.compound_resnet_weight)
            )
        else:
            # 默认 L1（稳定），与现有风格一致
            self.loss_function = nn.L1Loss()

        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    def init_optimizer(self, dataset_train):
        # Align with dival example: Adam without weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

    def init_scheduler(self, dataset_train):
        if str(self.scheduler).lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.lr_min)
        else:
            super().init_scheduler(dataset_train)

    # 推理：FBP -> tensor -> UNet -> numpy -> ODL 元素
    def _reconstruct(self, observation):
        self.model.eval()
        fbp = self.fbp_op(observation)  # ODL element
        fbp_tensor = torch.from_numpy(np.asarray(fbp)[None, None]).to(self.device).float()
        with torch.no_grad():
            reco_tensor = self.model(fbp_tensor)
        reconstruction = reco_tensor.detach().cpu().numpy()[0, 0]
        return self.reco_space.element(reconstruction)
    
    def build_dataloaders(self, data_cfg, dataset=None, ray_trafo=None):
        """
        构建训练和验证数据加载器
        
        Parameters
        ----------
        data_cfg : dict
            数据配置，包含batch_size等参数
            
        Returns
        -------
        train_loader, val_loader : tuple of DataLoader
            训练和验证数据加载器
        """
        from src.data import get_standard_dataset
        from torch.utils.data import DataLoader
        
        impl = data_cfg.get('impl', 'astra_cuda')
        # Reuse dataset / ray_trafo when provided (avoids duplicate ASTRA contexts)
        dataset = dataset or get_standard_dataset('lodopab', impl=impl)
        ray_trafo = ray_trafo or getattr(dataset, 'ray_trafo', None)
        if ray_trafo is None and hasattr(dataset, 'get_ray_trafo'):
            ray_trafo = dataset.get_ray_trafo(impl=impl)
        
        cache_files_cfg = data_cfg.get('cache_files') or {}
        # 默认仅缓存 FBP 分量，GT 不缓存，用 None 占位
        cache_files = {
            'train': cache_files_cfg.get('train', ('./cache_lodopab_train_fbp.npy', None)),
            'validation': cache_files_cfg.get('validation', ('./cache_lodopab_validation_fbp.npy', None)),
        }
        # 归一化 cache_files，防止传入单个字符串导致逐字符索引
        norm_cache_files = {}
        for part, files in cache_files.items():
            if isinstance(files, str):
                files = (files, None)
            elif isinstance(files, list):
                if len(files) == 1:
                    files = (files[0], None)
                else:
                    files = tuple(files[:2])
            norm_cache_files[part] = files
        cache_files = norm_cache_files
        cache_size = data_cfg.get('cache_params', {})
        if data_cfg.get('generate_cache'):
            generate_fbp_cache_files(
                dataset,
                ray_trafo,
                cache_files,
                size=cache_size,
                filter_type=self.filter_type,
                frequency_scaling=self.frequency_scaling,
            )
        # Use cached FBPs when available; fall back to on-the-fly if missing
        # 如果需要，先生成缓存（只需一次）
        if data_cfg.get('generate_cache'):
            generate_fbp_cache_files(
                dataset,
                ray_trafo,
                cache_files,
                size=cache_size,
                filter_type=self.filter_type,
                frequency_scaling=self.frequency_scaling,
            )

        # 加载缓存（若不存在且未请求生成，则报错，避免静默回退高显存模式）
        fbp_dataset = get_cached_fbp_dataset(
            dataset,
            ray_trafo,
            cache_files,
            size=cache_size,
            filter_type=self.filter_type,
            frequency_scaling=self.frequency_scaling,
        )
        if not fbp_dataset.supports_random_access():
            if not self.allow_multiple_workers_without_random_access:
                if self.num_data_loader_workers > 1:
                    warn('Overriding data loader workers to 1 for non-random-access dataset. '
                         'Pass allow_multiple_workers_without_random_access=True to override.')
                self.num_data_loader_workers = min(self.num_data_loader_workers, 1)
        
        # 创建训练和验证数据集
        train_dataset = fbp_dataset.create_torch_dataset(
            part='train', 
            reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape)
        )
        val_dataset = fbp_dataset.create_torch_dataset(
            part='validation', 
            reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape)
        )
        
        # 获取配置参数
        train_cfg = data_cfg.get('train', {})
        val_cfg = data_cfg.get('val', {})
        train_batch_size = train_cfg.get('batch_size', 2)
        val_batch_size = val_cfg.get('batch_size', 2)
        train_shuffle = train_cfg.get('shuffle', True)
        val_shuffle = val_cfg.get('shuffle', False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=0,  # 设置为0避免多进程问题
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size,
            shuffle=val_shuffle,
            num_workers=0,  # 设置为0避免多进程问题
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def build_model(self, cfg):
        """
        构建模型
        
        Parameters
        ----------
        cfg : dict
            配置参数
            
        Returns
        -------
        model : torch.nn.Module
            构建的模型
        """
        # 初始化模型
        self.init_model()
        return self.model
    
    def build_optim(self, cfg):
        """
        构建优化器
        
        Parameters
        ----------
        cfg : dict
            配置参数
            
        Returns
        -------
        optimizer : torch.optim.Optimizer
            构建的优化器
        """
        # 初始化优化器
        self.init_optimizer(None)
        return self.optimizer


# 导出类以便在 __init__.py 中使用
__all__ = ['FBPUNetReconstructor']
