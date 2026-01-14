# -*- coding: utf-8 -*-
"""Provides standard datasets for benchmarking.
"""
from warnings import warn
from functools import partial
import numpy as np
import odl
from .lodopab_dataset import LoDoPaBDataset
from .angle_subset_dataset import get_angle_subset_dataset


STANDARD_DATASET_NAMES = ['ellipses', 'lodopab']


def get_standard_dataset(name, **kwargs):
    """
    Return a standard dataset by name.

    The standard datasets are (currently):

        ``'ellipses'``
            A typical synthetical CT dataset with ellipse phantoms.

            `EllipsesDataset` is used as ground truth dataset, a ray
            transform with parallel beam geometry using 30 angles is applied,
            and white gaussian noise with a standard deviation of 2.5% (i.e.
            ``0.025 * mean(abs(observation))``) is added.

            In order to avoid the inverse crime, the ground truth images of
            shape (128, 128) are upscaled by bilinear interpolation to a
            resolution of (400, 400) before the ray transform is applied (whose
            discretization is different from the one of :attr:`ray_trafo`).
            
            Attributes of the returned dataset:
                `ray_trafo` : :class:`odl.tomo.RayTransform`
                    Ray transform corresponding to the noiseless forward
                    operator.
                ``get_ray_trafo(**kwargs)`` : function
                    Function that returns a ray transform corresponding to the
                    noiseless forward operator. Keyword arguments (e.g. `impl`)
                    are forwarded to the :class:`RayTransform` constructor.

        ``'lodopab'``
            The LoDoPaB-CT dataset, which is documented in the Data Descriptor
            article `<https://www.nature.com/articles/s41597-021-00893-z>`_ and
            hosted on `<https://zenodo.org/record/3384092>`_.
            It is a simulated low dose CT dataset based on real reconstructions
            from the `LIDC-IDRI
            <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_
            dataset.

            The dataset contains 42895 pairs of images and projection data.
            For simulation, a ray transform with parallel beam geometry using
            1000 angles and 513 detector pixels is used. Poisson noise
            corresponding to 4096 incident photons per pixel before attenuation
            is applied to the projection data.
            
            Attributes of the returned dataset:
                `ray_trafo` : :class:`odl.tomo.RayTransform`
                    Ray transform corresponding to the noiseless forward
                    operator.
            Methods of the returned dataset:
                ``get_ray_trafo(**kwargs)``
                    Function that returns a ray transform corresponding to the
                    noiseless forward operator. Keyword arguments (e.g. `impl`)
                    are forwarded to the :class:`RayTransform` constructor.

    Parameters
    ----------
    name : str
        Name of the dataset.
    kwargs : dict
        Keyword arguments.
        Supported parameters for the datasets are:

            ``'ellipses'``
                impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                        optional
                    Implementation passed to :class:`odl.tomo.RayTransform`
                    Default: ``'astra_cuda'``.
                fixed_seeds : dict or bool, optional
                    Seeds to use for random ellipse generation, passed to
                    :meth:`.EllipsesDataset.__init__`.
                    Default: ``False``.
                fixed_noise_seeds : dict or bool, optional
                    Seeds to use for noise generation, passed as `noise_seeds`
                    to :meth:`.GroundTruthDataset.create_pair_dataset`.
                    If ``True`` is passed (the default), the seeds
                    ``{'train': 1, 'validation': 2, 'test': 3}`` are used.
            ``'lodopab'``
                num_angles : int, optional
                    Number of angles to use from the full 1000 angles.
                    Must be a divisor of 1000.
                observation_model : {``'post-log'``, ``'pre-log'``}, optional
                    The observation model to use. Default is ``'post-log'``.
                min_photon_count : float, optional
                    Replacement value for a simulated photon count of zero.
                    If ``observation_model == 'post-log'``, a value greater
                    than zero is required in order to avoid undefined values.
                    The default is 0.1, both for ``'post-log'`` and
                    ``'pre-log'`` model.
                sorted_by_patient : bool, optional
                    Whether to sort the samples by patient id.
                    Useful to resplit the dataset.
                    Default: ``False``.
                impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                        optional
                    Implementation passed to :class:`odl.tomo.RayTransform`
                    Default: ``'astra_cuda'``.

    Returns
    -------
    dataset : :class:`.Dataset`
        The standard dataset.
        It has an attribute `standard_dataset_name` that stores its name.
    """
    name = name.lower()
    

    if name == 'lodopab':

        num_angles = kwargs.pop('num_angles', None)
        lodopab_kwargs = {}
        for k in ['observation_model', 'min_photon_count', 'sorted_by_patient',
                  'impl']:
            if k in kwargs:
                lodopab_kwargs[k] = kwargs.pop(k)

        dataset = LoDoPaBDataset(**lodopab_kwargs)

        if num_angles is not None:
            dataset = get_angle_subset_dataset(
                dataset, num_angles, impl=kwargs.get('impl', 'astra_cuda'))

    else:
        raise ValueError("unknown dataset '{}'. Known standard datasets are {}"
                         .format(name, STANDARD_DATASET_NAMES))

    if kwargs:
        warn('unused keyword arguments: {}'
             .format(', '.join(kwargs.keys())))

    dataset.standard_dataset_name = name
    return dataset
