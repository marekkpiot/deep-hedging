import math

import torch

from src.hedging_network import (
    HedgingNetwork,
)


def european_call_payoff(
    terminal_prices: torch.Tensor,
    strike: float,
) -> torch.Tensor:
    """
    Calcule le payoff d'un call européen.

    payoff = max(prix terminal - strike, 0)
    """

    if strike <= 0:
        raise ValueError(
            "Le strike doit être positif."
        )

    return torch.relu(
        terminal_prices - strike
    )


def simulate_hedging_pnl(
    network: HedgingNetwork,
    paths: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    maturity: float,
    option_premium: float,
) -> dict[str, torch.Tensor]:
    """
    Simule la couverture d'un call vendu.

    Le réseau choisit une position en actions
    avant chaque mouvement du prix.

    La fonction renvoie notamment :

    - le payoff du call ;
    - les positions de couverture ;
    - la valeur finale du portefeuille ;
    - le PnL final.
    """

    if paths.ndim != 2:
        raise ValueError(
            "Le tenseur paths doit avoir "
            "deux dimensions."
        )

    if paths.shape[1] < 2:
        raise ValueError(
            "Chaque trajectoire doit contenir "
            "au moins deux prix."
        )

    if strike <= 0:
        raise ValueError(
            "Le strike doit être positif."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être positive."
        )

    if option_premium < 0:
        raise ValueError(
            "La prime ne peut pas être négative."
        )

    if (paths <= 0).any():
        raise ValueError(
            "Tous les prix doivent être positifs."
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

    # La banque reçoit la prime du call au départ.
    cash_account = torch.full(
        size=(number_of_paths,),
        fill_value=option_premium,
        dtype=paths.dtype,
        device=paths.device,
    )

    # Au départ, aucune action n'est détenue.
    current_position = torch.zeros(
        number_of_paths,
        dtype=paths.dtype,
        device=paths.device,
    )

    positions_over_time = []

    for step in range(number_of_steps):

        current_prices = paths[:, step]

        proportion_of_time_remaining = (
            number_of_steps - step
        ) / number_of_steps

        time_remaining = torch.full_like(
            current_prices,
            fill_value=(
                proportion_of_time_remaining
            ),
        )

        moneyness = (
            current_prices / strike
        )

        state = torch.stack(
            [
                moneyness,
                time_remaining,
                current_position,
            ],
            dim=1,
        )

        new_position = network(state)

        number_of_shares_bought = (
            new_position
            - current_position
        )

        cost_of_rebalancing = (
            number_of_shares_bought
            * current_prices
        )

        cash_account = (
            cash_account
            - cost_of_rebalancing
        )

        current_position = new_position

        positions_over_time.append(
            current_position
        )

        # Le compte en espèces produit des intérêts
        # ou, s'il est négatif, génère un coût d'emprunt.
        cash_account = (
            cash_account
            * interest_growth
        )

    terminal_prices = paths[:, -1]

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