import torch


def zero_one_approximation_loss(guess, target, sigma):
    """
    Zero One Approximation loss, "well" shaped around target
    :param guess: torch.Tensor, guess
    :param target: torch.Tensor, target
    :param sigma: width of well around target, higher value means larger width
    :return: loss
    """
    diff_squared = (guess - target) ** 2
    loss = 1 - torch.exp(-diff_squared / (2 * sigma ** 2))
    return loss.mean()


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


def calc_mae(g: torch.Tensor, t: torch.Tensor, axis=None) -> torch.Tensor:
    """
    mae
    :param g: torch.Tensor, guess
    :param t: torch.Tensor, target
    :param axis: int, dimension along which to calculate MAE, None for entire mean
    :return: torch.Tensor, mae of guess from target
    """
    return torch.mean(torch.abs(t - g), dim=axis)


def calc_mape(g: torch.Tensor, t: torch.Tensor, axis=None) -> torch.Tensor:
    """
    mape
    :param g: torch.Tensor, guess
    :param t: torch.Tensor, target
    :param axis: int, dimension along which to calculate MAPE, None for entire mean
    :return: torch.Tensor, mape of guess from target
    """
    return torch.mean(torch.abs((t - g) / t), dim=axis) * 100
