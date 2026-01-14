# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import torch.nn as nn
from copy import deepcopy

from src.utils.registry import register
from src.models.reconstructors.standard_learned_reconstructor import StandardLearnedReconstructor
from dival.reconstructors.networks.iradonmap import get_iradonmap_model


@register("reconstructor", "iradonmap")
class IRadonMapReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor that learns a fully connected layer for filtering along
    the axis of the detector pixels s, followed by the backprojection
    (segment 1). After that, a residual CNN acts as a post-processing net
    (segment 2). We use the U-Net from the FBPUnet model.

    In the original paper [1]_, a learned version of the back-
    projection layer (sinusoidal layer) is used. This layer introduces a lot
    more parameters. Therefore, we added an option to directly use the operator
    in our implementation. Additionally, we drop the tanh activation after
    the first fully connected layer, due to bad performance.

    In any configuration, the iRadonMap has less parameters than an
    Automap network [2]_.

    References
    ----------
    .. [1] J. He and J. Ma, 2020,
           "Radon Inversion via Deep Learning".
           IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 2076-2087
           `doi:10.1109/TMI.2020.2964266
           <https://doi.org/10.1109/TMI.2020.2964266>`_
    .. [2] B. Zhu, J. Z. Liu, S. F. Cauly et al., 2018,
           "Image Reconstruction by Domain-Transform Manifold Learning".
           Nature 555, 487--492.
           `doi:10.1038/nature25988
           <https://doi.org/10.1038/nature25988>`_
    """

    def __init__(self, op, coord_mat=None, **kwargs):
        """
        Parameters
        ----------
        op : :class:`odl.tomo.RayTransform`
            Ray transform (the forward operator).
        coord_mat : array, optional
            Precomputed coordinate matrix for the `LearnedBackprojection`.
            This option is provided for performance optimization.
            If `None` is passed, the matrix is computed in :meth:`init_model`.

        Further keyword arguments are passed to ``super().__init__()``.
        """
        # Extract parameters from kwargs
        self.coord_mat = coord_mat
        self.fully_learned = kwargs.pop('fully_learned', False)
        self.scales = kwargs.pop('scales', 5)
        self.skip_channels = kwargs.pop('skip_channels', 4)
        self.use_sigmoid = kwargs.pop('use_sigmoid', False)

        # Remove extra kwargs passed by Trainer to avoid unexpected keyword errors
        kwargs.pop('ray_trafo', None)
        
        # Set default use_cuda if not provided
        if 'use_cuda' not in kwargs:
            kwargs['use_cuda'] = True
        
        # Call parent constructor
        super().__init__(op=op, **kwargs)
        
        # Add parameters to hyper_params
        self.hyper_params.update({
            'fully_learned': self.fully_learned,
            'scales': self.scales,
            'skip_channels': self.skip_channels,
            'use_sigmoid': self.use_sigmoid,
        })

    def init_model(self):
        """Initialize the IRadonMap model."""
        self.model = get_iradonmap_model(
                ray_trafo=self.op, fully_learned=self.fully_learned,
                scales=self.scales, skip=self.skip_channels,
                use_sigmoid=self.use_sigmoid, coord_mat=self.coord_mat)
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)
    
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
        
        # 获取数据集
        impl = data_cfg.get('impl', 'astra_cuda')
        dataset = dataset or get_standard_dataset('lodopab', impl=impl)
        
        # 创建训练和验证数据集
        train_dataset = dataset.create_torch_dataset(
            part='train', 
            reshape=((1,) + dataset.space[0].shape, (1,) + dataset.space[1].shape)
        )
        val_dataset = dataset.create_torch_dataset(
            part='validation', 
            reshape=((1,) + dataset.space[0].shape, (1,) + dataset.space[1].shape)
        )
        
        # 获取配置参数
        train_cfg = data_cfg.get('train', {})
        val_cfg = data_cfg.get('val', {})
        train_batch_size = train_cfg.get('batch_size', 2)
        val_batch_size = val_cfg.get('batch_size', 2)
        train_shuffle = train_cfg.get('shuffle', True)
        val_shuffle = val_cfg.get('shuffle', False)
        train_pin_memory = train_cfg.get('pin_memory', True)
        val_pin_memory = val_cfg.get('pin_memory', True)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=0,  # 设置为0避免多进程问题
            pin_memory=train_pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size,
            shuffle=val_shuffle,
            num_workers=0,  # 设置为0避免多进程问题
            pin_memory=val_pin_memory
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
    
    def _reconstruct(self, observation):
        """
        使用IRadonMap网络进行重建
        
        Parameters
        ----------
        observation : odl.discr.lp_discr.DiscreteLp
            观测数据（正弦图）
            
        Returns
        -------
        reconstruction : odl.discr.lp_discr.DiscreteLp
            重建结果
        """
        self.model.eval()
        with torch.no_grad():
            # 将观测数据转换为tensor
            obs_tensor = torch.from_numpy(observation.asarray()[None, None]).to(self.device).float()
            
            # 通过模型
            reco_tensor = self.model(obs_tensor)
            
            # 转换回numpy并返回ODL元素
            reconstruction = reco_tensor.detach().cpu().numpy()[0, 0]
            return self.reco_space.element(reconstruction)
