import numpy as np
import torch


def rmwe_loss(g, t, reduction='mean'):
    """
    Relative mean weighted error
    :param g: array-like, guess
    :param t: array-like, target
    :param reduction: 'mean' or 'sum'
    :return: rmwe of guess from target
    """
    if reduction == 'mean':
        return torch.mean(torch.square(t - g) / torch.square(t))
    elif reduction == 'sum':
        return torch.sum(torch.square(t - g) / torch.square(t))


def mape_loss(g, t, reduction='mean'):
    """
    MAPE loss
    :param g: array-like, guess
    :param t: array-like, target
    :param reduction: str, only 'mean'
    :return: MAPE of guess from target
    """
    if reduction == 'mean':
        return torch.mean(torch.abs((g - t) / t)) * 100
    else:
        raise NotImplementedError('Only mean reduction is supported.')


def calc_mae(g, t, axis=None):
    """
    mae
    :param g: array-like, guess
    :param t: array-like, target
    :param axis: int, axis along which to calculate MAE, None for entire mean
    :return: mae of guess from target
    """
    return np.mean(np.abs(t - g), axis=axis)


def calc_mape(g, t, axis=None):
    """
    mape
    :param g: array-like, guess
    :param t: array-like, target
    :param axis: int, axis along which to calculate MAE, None for entire mean
    :return: mape of guess from target
    """
    return np.mean(np.abs((t - g) / t), axis=axis) * 100
