#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from . import measure
from . import plot
from .torch_utility import to_device
from .std_out_err_redirect_tqdm import std_out_err_redirect_tqdm
import pandas as pd
import os
import datetime
import numpy as np


class TaskTable:
    """
    A task table that holds a number of tasks and can run them.
    A task is a dictionary of the following form:
        {
            'reconstructor': a reconstructor object,
            'ground_truth': a list of ground truth images,
            'test_data': a list of test data,
            'measures': {
                'measure_a': callable_a,
                'measure_b': callable_b
            }
        }
    The table can be run, which will iterate through all tasks and test data
    and apply the reconstructor and the measures.
    The results are stored in a pandas DataFrame.
    """

    def __init__(self, name='default'):
        self.name = name
        self.tasks = []
        self.results = pd.DataFrame()

    def append(self, reconstructor, test_data, measures,
               options=None):

        task = {
            'reconstructor': reconstructor,
            'test_data': test_data,
            'measures': measures,
            'options': options
        }
        self.tasks.append(task)

    def run(self, save_reconstructions=False, show_progress='tqdm'):
        self.results = pd.DataFrame()
        if save_reconstructions:
            self.reconstructions = {}

        if show_progress == 'tqdm':
            outer_loop = tqdm(self.tasks, desc=self.name, position=0)
        else:
            outer_loop = self.tasks

        for i, task in enumerate(outer_loop):
            if show_progress == 'tqdm':
                outer_loop.set_description(
                    'task {}/{}'.format(i + 1, len(self.tasks)))
            elif show_progress == 'text':
                print('task {}/{}'.format(i + 1, len(self.tasks)))

            reconstructor = task['reconstructor']
            test_data = task['test_data']
            measures = task['measures']
            options = task['options'] if task['options'] is not None else {}

            if not options.get('skip_training', False):
                reconstructor.train(test_data.train)

            if save_reconstructions:
                self.reconstructions[i] = []

            if show_progress == 'tqdm':
                inner_loop = tqdm(test_data, desc=reconstructor.name,
                                  position=1, leave=False)
            else:
                inner_loop = test_data

            with std_out_err_redirect_tqdm() as orig_stdout:
                for j, (obs, gt) in enumerate(inner_loop):
                    if show_progress == 'tqdm':
                        inner_loop.set_description(
                            'sample {}/{}'.format(j + 1, len(test_data)))
                    elif show_progress == 'text':
                        print('sample {}/{}'.format(j + 1, len(test_data)))

                    x = to_device(reconstructor.device, obs)
                    reconstruction = reconstructor.reconstruct(
                        x, gt.space).asarray()

                    if save_reconstructions:
                        self.reconstructions[i].append(
                            (reconstruction, gt.asarray()))

                    row = {'task': i,
                           'reconstructor': reconstructor.name,
                           'sample': j}

                    for measure_name, measure_func in measures.items():
                        row[measure_name] = measure_func(reconstruction,
                                                         gt.asarray())

                    self.results = self.results.append(row, ignore_index=True)

        return self.results

    def save_results(self, folder=None):
        if folder is None:
            folder = 'results_{}'.format(self.name)
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            'results_{}.csv'.format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        self.results.to_csv(path)

    def plot_results(self, folder=None, x='reconstructor', y=None,
                     hue='sample', style=None, col=None, row=None,
                     kind='strip', options=None):

        if y is None:
            y = list(self.tasks[0]['measures'].keys())

        if folder is None:
            folder = 'results_{}'.format(self.name)
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            'results_{}.png'.format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        plot.plot_grid(self.results, path, x, y, hue, style, col, row, kind,
                       options)
