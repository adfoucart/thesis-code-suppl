import json

import numpy as np
from .statistics import friedman_test, nemenyi_multitest


class _GLAS_PARAMS:
    reference = {
        'F-Norm-4': ['Baseline', 'Original'],
        'F-R50':  ['Baseline', 'Noisy'],
        'F-BB':  ['Baseline', 'BB'],
        'F-R50-BB':  ['Baseline', 'NoisyBB'],
        'F-R50-HD':  ['Baseline', 'NoisyHD'],
        'GEN-anno':  ['OnlyP', 'Original'],
        'GEN-R50': ['OnlyP', 'Noisy'],
        'GEN-BB': ['OnlyP', 'BB'],
        'GEN-R50-BB': ['OnlyP', 'NoisyBB'],
        'GEN-R50-HD': ['OnlyP', 'NoisyHD'],
        'SS0-anno': ['SS', 'Original'],
        'SS0-R50': ['SS', 'Noisy'],
        'SS0-BB': ['SS', 'BB'],
        'SS0-R50-BB': ['SS', 'NoisyBB'],
        'SS0-R50-HD': ['SS', 'NoisyHD'],
        'SSP-anno': ['SSP', 'Original'],
        'SSP-r50': ['SSP', 'Noisy'],
        'SSP-BB':['SSP', 'BB'],
        'SSP-R50-BB': ['SSP', 'NoisyBB'],
        'SSP-R50-HD': ['SSP', 'NoisyHD'],
        'GSN-anno-datacentered': ['GA100', 'Original'],
        'GSN-r50-datacentered': ['GA100', 'Noisy'],
        'GSN-bb-datacentered': ['GA100', 'BB'],
        'GSN-r50-bb-datacentered': ['GA100', 'NoisyBB'],
        'GSN-r50-hd-datacentered': ['GA100', 'NoisyHD'],
        'GSN75-anno-datacentered': ['GA75', 'Original'],
        'GSN75-r50-datacentered': ['GA75', 'Noisy'],
        'GSN75-bb-datacentered': ['GA75', 'BB'],
        'GSN75-r50-bb-datacentered': ['GA75', 'NoisyBB'],
        'GSN75-r50-hd-datacentered': ['GA75', 'NoisyHD'],
        'tda-anno': ['LA', 'Original'],
        'tda-r50': ['LA', 'Noisy'],
        'tda-bb': ['LA', 'BB'],
        'tda-r50-bb': ['LA', 'NoisyBB'],
        'tda-r50-hd': ['LA', 'NoisyHD']
    }

    networks = ['Baseline', 'OnlyP', 'SS', 'SSP', 'GA100', 'GA75', 'LA']
    datasets = ['Original', 'Noisy', 'BB', 'NoisyBB', 'NoisyHD']
    net_short = ['BASE', 'OP', 'SS', 'SSP', 'GA100', 'GA75', 'LA']
    ds_short = ['ORI', 'NOI', 'BB', 'NBB', 'NHD']


class _EPITHELIUM_PARAMS:
    reference = {
        'Baseline-Full': ['Baseline', 'Original'],
        'Baseline-Noisy': ['Baseline', 'Noisy'],
        'Baseline-Deformed': ['Baseline', 'Deformed'],
        'OnlyP-Full': ['OnlyP', 'Original'],
        'OnlyP-Noisy': ['OnlyP', 'Noisy'],
        'OnlyP-Deformed': ['OnlyP', 'Deformed'],
        'GSN100-Full': ['GA100', 'Original'],
        'GSN100-Noisy': ['GA100', 'Noisy'],
        'GSN100-Deformed': ['GA100', 'Deformed'],
        'TDA-Full': ['LA', 'Original'],
        'TDA-Noisy': ['LA', 'Noisy'],
        'TDA-Deformed': ['LA', 'Deformed'],
        'SSL-OnlyP-Full': ['SSL-OnlyP', 'Original'],
        'SSL-OnlyP-Noisy': ['SSL-OnlyP', 'Noisy'],
        'SSL-OnlyP-Deformed': ['SSL-OnlyP', 'Deformed']
    }

    networks = ['Baseline', 'OnlyP', 'GA100', 'LA', 'SSL-OnlyP']
    datasets = ['Original', 'Noisy', 'Deformed']
    net_short = ['BASE', 'OP', 'GA100', 'LA', 'SSP']
    ds_short = ['ORI', 'NOI', 'DEF']


_DATASETS_PARAMS = {
    'epithelium': _EPITHELIUM_PARAMS,
    'glas': _GLAS_PARAMS
}


def compute_stat_score(results: np.array):
    differences = np.zeros((results.shape[0], results.shape[1], results.shape[1]))

    for d in range(results.shape[0]):
        means = results[d].mean(axis=1)
        for i in range(results.shape[1]):
            for j in range(results.shape[1]):
                differences[d, i, j] = means[i] - means[j]

    # Compute Friedman, Nemenyi & points
    points_summary = np.zeros((results.shape[1], results.shape[0]))
    for d in range(results.shape[0]):
        values = [results[d, i, :].T for i in range(results.shape[1])]
        fv, pv, ranks, pivots = friedman_test(*values)

        pvArray = nemenyi_multitest(pivots)
        isBetter = differences[d] > 0
        points = (isBetter * (0 + (pvArray < 0.05)) - (np.equal(isBetter, False) * (pvArray < 0.05)))
        points[np.eye(results.shape[1]) == 1] = 0

        points_s = points.sum(axis=1)
        points_summary[:, d] = points_s

    return points_summary


def print_result_table(results: np.array, title: str, dataset: str) -> None:
    print(title)
    net_short = _DATASETS_PARAMS[dataset].net_short
    ds_short = _DATASETS_PARAMS[dataset].ds_short
    print('\t' + '\t'.join(ds_short))
    for idn, n in enumerate(net_short):
        if results.dtype == float:
            print(n + '\t' + '\t'.join([f'{p:.3f}' for p in results[idn, :]]))
        else:
            print(n + '\t' + '\t'.join([f'{p}' for p in results[idn, :]]))


def get_glas_results_tables(json_file: str) -> np.array:
    with open(json_file, 'r') as fp:
        scores = json.load(fp)

    networks = _DATASETS_PARAMS['glas'].networks
    datasets = _DATASETS_PARAMS['glas'].datasets

    n_train = 85
    n_testA = 60
    n_testB = 20

    dsc_matrix = np.zeros((len(datasets), len(networks), n_testA + n_testB))
    mcc_matrix = np.zeros((len(datasets), len(networks), n_testA + n_testB))

    for clf, (net, ds) in _DATASETS_PARAMS['glas'].reference.items():
        dsc_matrix[datasets.index(ds), networks.index(net), :] = scores[clf]['f1'][n_train:]
        mcc_matrix[datasets.index(ds), networks.index(net), :] = scores[clf]['mcc'][n_train:]

    return dsc_matrix, mcc_matrix


def get_epithelium_results_tables(json_file: str) -> np.array:
    with open(json_file, 'r') as fp:
        scores = json.load(fp)

    macros = ['shortres', 'pan']
    networks = _DATASETS_PARAMS['epithelium'].networks
    datasets = _DATASETS_PARAMS['epithelium'].datasets

    n_train = 35
    n_test = 7

    dsc_matrix = np.zeros((len(macros), len(datasets), len(networks), n_test))
    mcc_matrix = np.zeros((len(macros), len(datasets), len(networks), n_test))

    for idm, macro in enumerate(macros):
        for clf, (net, ds) in _DATASETS_PARAMS['epithelium'].reference.items():
            dsc_matrix[idm, datasets.index(ds), networks.index(net), :] = scores[macro][clf]['dsc'][n_train:]
            mcc_matrix[idm, datasets.index(ds), networks.index(net), :] = scores[macro][clf]['mcc'][n_train:]

    return dsc_matrix, mcc_matrix

