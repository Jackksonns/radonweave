# -*- coding: utf-8 -*-
"""Provides the DataPairs class."""

class DataPairs:
    """A collection of paired data (observations and ground truth).

    Attributes
    ----------
    observations : list
        List of observation samples.
    ground_truth : list
        List of ground truth samples.
    name : str, optional
        Name of the data pairs collection.
    """
    def __init__(self, observations, ground_truth, name=None):
        """
        Parameters
        ----------
        observations : list
            List of observation samples.
        ground_truth : list
            List of ground truth samples.
        name : str, optional
            Name of the data pairs collection.
        """
        self.observations = observations
        self.ground_truth = ground_truth
        self.name = name or ''

    def get_len(self):
        """Return the number of pairs."""
        return len(self.observations)