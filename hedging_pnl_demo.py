from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.black_scholes import (
    black_scholes_call_price,
)

from src.gbm import (
    simulate_gbm_paths,
)

from src.hedging import (
    european_call_payoff,
    simulate_hedging_pnl,
)

from src.hedging_network import (
    HedgingNetwork,
)


INITIAL_PRICE = 100.0
STRIKE = 100.0

RISK_FREE_RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

NUMBER_OF_STEPS = 30
NUMBER_OF_PATHS = 10_000

RANDOM_SEED = 42


def display_pnl_statistics(
    name: str,
    pnl: torch.Tensor,
) -> None:
    """
    Affiche quelques statistiques descriptives
    d'une distribution de PnL.
    """

    mean_pnl = pnl.mean().item()

    standard_deviation = (
        pnl.std().item()
    )

    root_mean_squared_pnl = (
        torch.mean(pnl**2)
        .sqrt()
        .item()
    )

    minimum_pnl = pnl.min().item()
    maximum_pnl = pnl.max().item()

    print(name)
    print()

    print(
        f"PnL moyen : "
        f"{mean_pnl:.4f} €"
    )

    print(
        f"Écart-type du PnL : "
        f"{standard_deviation:.4f} €"
    )

    print(
        f"Racine de la moyenne des PnL² : "
        f"{root_mean_squared_pnl:.4f} €"
    )

    print(
        f"PnL minimal : "
        f"{minimum_pnl:.4f} €"
    )

    print(
        f"PnL maximal : "
        f"{maximum_pnl:.4f} €"
    )

    print()


def plot_pnl_distributions(
    unhedged_pnl: torch.Tensor,
    network_pnl: torch.Tensor,
) -> None:
    """
    Compare les distributions de PnL sans couverture
    et avec le réseau non entraîné.
    """

    Path("figures").mkdir(
        exist_ok=True
    )

    unhedged_as_array = (
        unhedged_pnl
        .detach()
        .cpu()
        .numpy()
    )

    network_as_array = (
        network_pnl
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        unhedged_as_array,
        bins=80,
        alpha=0.5,
        label="Sans couverture",
    )

    plt.hist(
        network_as_array,
        bins=80,
        alpha=0.5,
        label="Réseau non entraîné",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        label="PnL nul",
    )

    plt.title(
        "PnL avant entraînement du réseau"
    )

    plt.xlabel("PnL final (€)")
    plt.ylabel("Nombre de trajectoires")
    plt.grid()
    plt.legend()

    plt.savefig(
        "figures/untrained_network_pnl.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    torch.manual_seed(
        RANDOM_SEED
    )

    option_premium = (
        black_scholes_call_price(
            initial_price=INITIAL_PRICE,
            strike=STRIKE,
            risk_free_rate=RISK_FREE_RATE,
            volatility=VOLATILITY,
            maturity=MATURITY,
        )
    )

    print(
        f"Prime Black-Scholes du call : "
        f"{option_premium:.4f} €"
    )

    print()

    paths = simulate_gbm_paths(
        initial_price=INITIAL_PRICE,
        risk_free_rate=RISK_FREE_RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        number_of_steps=NUMBER_OF_STEPS,
        number_of_paths=NUMBER_OF_PATHS,
    )

    network = HedgingNetwork(
        hidden_size=16
    )

    network.eval()

    terminal_prices = paths[:, -1]

    payoff = european_call_payoff(
        terminal_prices=terminal_prices,
        strike=STRIKE,
    )

    unhedged_pnl = (
        option_premium
        * math.exp(
            RISK_FREE_RATE * MATURITY
        )
        - payoff
    )

    # Démonstration uniquement :
    # nous ne voulons pas encore calculer de gradients.
    with torch.no_grad():

        hedging_results = (
            simulate_hedging_pnl(
                network=network,
                paths=paths,
                strike=STRIKE,
                risk_free_rate=(
                    RISK_FREE_RATE
                ),
                maturity=MATURITY,
                option_premium=(
                    option_premium
                ),
            )
        )

    network_pnl = (
        hedging_results["pnl"]
    )

    hedge_positions = (
        hedging_results[
            "hedge_positions"
        ]
    )

    print(
        "Forme des positions de couverture :"
    )

    print(hedge_positions.shape)

    print()

    print(
        "Premières positions de la "
        "première trajectoire :"
    )

    print(
        hedge_positions[0, :5]
    )

    print()

    display_pnl_statistics(
        name="Sans couverture",
        pnl=unhedged_pnl,
    )

    display_pnl_statistics(
        name="Réseau non entraîné",
        pnl=network_pnl,
    )

    loss = torch.mean(
        network_pnl**2
    )

    print(
        "Loss du réseau non entraîné :"
    )

    print(
        f"{loss.item():.6f}"
    )

    plot_pnl_distributions(
        unhedged_pnl=unhedged_pnl,
        network_pnl=network_pnl,
    )

    print()
    print(
        "Graphique enregistré dans "
        "figures/untrained_network_pnl.png"
    )


if __name__ == "__main__":
    main()