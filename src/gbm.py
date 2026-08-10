import math

import torch


def simulate_gbm_paths(
    initial_price: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    number_of_steps: int,
    number_of_paths: int,
) -> torch.Tensor:
    """
    Simule plusieurs trajectoires d'un mouvement
    brownien géométrique avec PyTorch.

    La forme du tenseur retourné est :

    nombre de trajectoires
    ×
    nombre de dates

    Le nombre de dates vaut number_of_steps + 1,
    car le prix initial est également conservé.
    """

    if initial_price <= 0:
        raise ValueError(
            "Le prix initial doit être positif."
        )

    if volatility < 0:
        raise ValueError(
            "La volatilité ne peut pas être négative."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être positive."
        )

    if number_of_steps <= 0:
        raise ValueError(
            "Le nombre d'étapes doit être positif."
        )

    if number_of_paths <= 0:
        raise ValueError(
            "Le nombre de trajectoires doit être positif."
        )

    time_step = (
        maturity / number_of_steps
    )

    random_shocks = torch.randn(
        number_of_paths,
        number_of_steps,
        dtype=torch.float32,
    )

    prices = torch.empty(
        number_of_paths,
        number_of_steps + 1,
        dtype=torch.float32,
    )

    prices[:, 0] = initial_price

    drift = (
        risk_free_rate
        - 0.5 * volatility**2
    ) * time_step

    diffusion_scale = (
        volatility
        * math.sqrt(time_step)
    )

    for step in range(number_of_steps):

        current_prices = prices[:, step]

        current_shocks = (
            random_shocks[:, step]
        )

        next_prices = (
            current_prices
            * torch.exp(
                drift
                + diffusion_scale
                * current_shocks
            )
        )

        prices[:, step + 1] = (
            next_prices
        )

    return prices