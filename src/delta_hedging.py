import math

import torch

from src.hedging import (
    european_call_payoff,
)


def standard_normal_cdf_tensor(
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Fonction de répartition de la loi normale
    standard appliquée à un tenseur.
    """

    return 0.5 * (
        1.0
        + torch.erf(
            values / math.sqrt(2.0)
        )
    )


def black_scholes_delta_tensor(
    stock_prices: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> torch.Tensor:
    """
    Calcule simultanément le delta Black-Scholes
    pour plusieurs prix d'action.
    """

    if strike <= 0:
        raise ValueError(
            "Le strike doit être positif."
        )

    if volatility <= 0:
        raise ValueError(
            "La volatilité doit être positive."
        )

    if time_to_maturity <= 0:
        return (
            stock_prices > strike
        ).to(
            dtype=stock_prices.dtype
        )

    d1 = (
        torch.log(
            stock_prices / strike
        )
        + (
            risk_free_rate
            + 0.5 * volatility**2
        )
        * time_to_maturity
    ) / (
        volatility
        * math.sqrt(
            time_to_maturity
        )
    )

    delta = standard_normal_cdf_tensor(
        d1
    )

    return delta


def simulate_delta_hedging_pnl(
    paths: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    option_premium: float,
) -> dict[str, torch.Tensor]:
    """
    Simule un delta-hedging Black-Scholes
    sur toutes les trajectoires.
    """

    if paths.ndim != 2:
        raise ValueError(
            "paths doit avoir deux dimensions."
        )

    number_of_paths = paths.shape[0]

    number_of_steps = (
        paths.shape[1] - 1
    )

    time_step = (
        maturity / number_of_steps
    )

    interest_growth = math.exp(
        risk_free_rate * time_step
    )

    # La banque reçoit la prime.
    cash_account = torch.full(
        size=(number_of_paths,),
        fill_value=option_premium,
        dtype=paths.dtype,
        device=paths.device,
    )

    # Aucune action au départ.
    current_position = torch.zeros(
        number_of_paths,
        dtype=paths.dtype,
        device=paths.device,
    )

    positions_over_time = []

    for step in range(
        number_of_steps
    ):

        current_prices = (
            paths[:, step]
        )

        time_to_maturity = (
            maturity
            - step * time_step
        )

        # Black-Scholes nous donne directement
        # la nouvelle quantité d'actions à détenir.
        new_position = (
            black_scholes_delta_tensor(
                stock_prices=current_prices,
                strike=strike,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_maturity=(
                    time_to_maturity
                ),
            )
        )

        shares_bought = (
            new_position
            - current_position
        )

        rebalancing_cost = (
            shares_bought
            * current_prices
        )

        cash_account = (
            cash_account
            - rebalancing_cost
        )

        current_position = (
            new_position
        )

        positions_over_time.append(
            current_position
        )

        cash_account = (
            cash_account
            * interest_growth
        )

    terminal_prices = (
        paths[:, -1]
    )

    payoff = european_call_payoff(
        terminal_prices=terminal_prices,
        strike=strike,
    )

    final_hedging_wealth = (
        cash_account
        + current_position
        * terminal_prices
    )

    pnl = (
        final_hedging_wealth
        - payoff
    )

    hedge_positions = torch.stack(
        positions_over_time,
        dim=1,
    )

    return {
        "pnl": pnl,
        "payoff": payoff,
        "final_hedging_wealth": (
            final_hedging_wealth
        ),
        "hedge_positions": (
            hedge_positions
        ),
    }