import torch


def residual_pnl_variance(
    pnl: torch.Tensor,
) -> torch.Tensor:
    """
    Calcule la variance du PnL final.

    La loss mesure la dispersion des PnL
    autour de leur moyenne.

    loss
    =
    moyenne de (PnL - PnL moyen)²
    """

    if pnl.ndim != 1:
        raise ValueError(
            "Le PnL doit être un tenseur "
            "à une dimension."
        )

    if pnl.numel() < 2:
        raise ValueError(
            "Il faut au moins deux trajectoires."
        )

    mean_pnl = pnl.mean()

    centered_pnl = (
        pnl - mean_pnl
    )

    variance = torch.mean(
        centered_pnl**2
    )

    return variance