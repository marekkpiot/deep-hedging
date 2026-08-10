from pathlib import Path
import math
import statistics

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.black_scholes import (
    black_scholes_call_price,
)

from src.delta_hedging import (
    simulate_delta_hedging_pnl,
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


MODEL_FILE = (
    "models/hedging_network.pt"
)

INITIAL_PRICE = 100.0
STRIKE = 100.0

RISK_FREE_RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

NUMBER_OF_STEPS = 30
NUMBER_OF_TEST_PATHS = 20_000

TEST_SEED = 2026


def load_trained_network() -> HedgingNetwork:
    """
    Charge le réseau précédemment entraîné.
    """

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )

    hidden_size = checkpoint[
        "hidden_size"
    ]

    network = HedgingNetwork(
        hidden_size=hidden_size
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.eval()

    return network

def compute_left_tail_cvar(
    pnl: torch.Tensor,
    tail_probability: float = 0.05,
) -> tuple[float, float]:
    """
    Calcule la VaR et la CVaR dans la queue gauche
    de la distribution du PnL.

    tail_probability = 0.05
    signifie que l'on regarde les 5 % pires PnL.

    Returns
    -------
    var_pnl:
        seuil séparant approximativement les
        5 % pires scénarios.

    cvar_pnl:
        PnL moyen parmi ces scénarios.
    """

    if pnl.ndim != 1:
        raise ValueError(
            "Le PnL doit avoir une seule dimension."
        )

    if not 0 < tail_probability < 1:
        raise ValueError(
            "La probabilité doit être comprise "
            "entre 0 et 1."
        )

    var_pnl = torch.quantile(
        pnl,
        tail_probability,
    )

    tail_pnl = pnl[
        pnl <= var_pnl
    ]

    cvar_pnl = tail_pnl.mean()

    return (
        var_pnl.item(),
        cvar_pnl.item(),
    )


def compute_statistics(
    pnl: torch.Tensor,
) -> dict[str, float]:
    """
    Calcule les principales statistiques
    de la distribution du PnL.
    """

    var_95, cvar_95 = (
        compute_left_tail_cvar(
            pnl=pnl,
            tail_probability=0.05,
        )
    )

    return {
        "mean": pnl.mean().item(),

        "std": pnl.std(
            unbiased=False
        ).item(),

        "variance": pnl.var(
            unbiased=False
        ).item(),

        "rmse": torch.mean(
            pnl**2
        ).sqrt().item(),

        "minimum": pnl.min().item(),

        "maximum": pnl.max().item(),

        "var_95": var_95,

        "cvar_95": cvar_95,
    }


def display_statistics(
    name: str,
    statistics: dict[str, float],
) -> None:
    """
    Affiche les statistiques d'une stratégie.
    """

    print(name)
    print("-" * len(name))

    print(
        f"PnL moyen       : "
        f"{statistics['mean']:.4f} €"
    )

    print(
        f"Écart-type      : "
        f"{statistics['std']:.4f} €"
    )

    print(
        f"Variance        : "
        f"{statistics['variance']:.4f}"
    )

    print(
        f"RMSE du PnL     : "
        f"{statistics['rmse']:.4f} €"
    )

    print(
        f"VaR 95 %         : "
        f"{statistics['var_95']:.4f} €"
    )

    print(
        f"CVaR 95 %        : "
        f"{statistics['cvar_95']:.4f} €"
    )

    print(
        f"PnL minimum     : "
        f"{statistics['minimum']:.4f} €"
    )

    print(
        f"PnL maximum     : "
        f"{statistics['maximum']:.4f} €"
    )

    print()


def plot_pnl_comparison(
    unhedged_pnl: torch.Tensor,
    delta_pnl: torch.Tensor,
    network_pnl: torch.Tensor,
) -> None:
    """
    Compare graphiquement les distributions
    du PnL final.
    """

    Path("figures").mkdir(
        exist_ok=True
    )

    unhedged_array = (
        unhedged_pnl
        .detach()
        .cpu()
        .numpy()
    )

    delta_array = (
        delta_pnl
        .detach()
        .cpu()
        .numpy()
    )

    network_array = (
        network_pnl
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(11, 6))

    plt.hist(
        unhedged_array,
        bins=100,
        alpha=0.35,
        density=True,
        label="Sans couverture",
    )

    plt.hist(
        delta_array,
        bins=100,
        alpha=0.50,
        density=True,
        label="Delta Black-Scholes",
    )

    plt.hist(
        network_array,
        bins=100,
        alpha=0.50,
        density=True,
        label="Deep hedging",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        label="PnL nul",
    )

    plt.title(
        "Comparaison des distributions "
        "du PnL final"
    )

    plt.xlabel("PnL final (€)")
    plt.ylabel("Densité")
    plt.grid()
    plt.legend()

    plt.savefig(
        "figures/pnl_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:

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
        f"Prime Black-Scholes : "
        f"{option_premium:.4f} €"
    )

    print()

    # Les trajectoires de test n'ont jamais
    # été utilisées pour entraîner le réseau.
    torch.manual_seed(
        TEST_SEED
    )

    test_paths = simulate_gbm_paths(
        initial_price=INITIAL_PRICE,
        risk_free_rate=RISK_FREE_RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        number_of_steps=NUMBER_OF_STEPS,
        number_of_paths=(
            NUMBER_OF_TEST_PATHS
        ),
    )

    terminal_prices = (
        test_paths[:, -1]
    )

    payoff = european_call_payoff(
        terminal_prices=terminal_prices,
        strike=STRIKE,
    )

    # -----------------------------
    # 1. Aucune couverture
    # -----------------------------

    unhedged_pnl = (
        option_premium
        * math.exp(
            RISK_FREE_RATE
            * MATURITY
        )
        - payoff
    )

    # -----------------------------
    # 2. Delta Black-Scholes
    # -----------------------------

    delta_results = (
        simulate_delta_hedging_pnl(
            paths=test_paths,
            strike=STRIKE,
            risk_free_rate=RISK_FREE_RATE,
            volatility=VOLATILITY,
            maturity=MATURITY,
            option_premium=option_premium,
        )
    )

    delta_pnl = (
        delta_results["pnl"]
    )

    # -----------------------------
    # 3. Deep hedging
    # -----------------------------

    network = (
        load_trained_network()
    )

    with torch.no_grad():

        network_results = (
            simulate_hedging_pnl(
                network=network,
                paths=test_paths,
                strike=STRIKE,
                risk_free_rate=RISK_FREE_RATE,
                maturity=MATURITY,
                option_premium=option_premium,
            )
        )

    network_pnl = (
        network_results["pnl"]
    )

    # -----------------------------
    # Statistiques
    # -----------------------------

    unhedged_statistics = (
        compute_statistics(
            unhedged_pnl
        )
    )

    delta_statistics = (
        compute_statistics(
            delta_pnl
        )
    )

    network_statistics = (
        compute_statistics(
            network_pnl
        )
    )

    display_statistics(
        "Sans couverture",
        unhedged_statistics,
    )

    display_statistics(
        "Delta Black-Scholes",
        delta_statistics,
    )

    display_statistics(
        "Deep hedging",
        network_statistics,
    )

    plot_pnl_comparison(
        unhedged_pnl=unhedged_pnl,
        delta_pnl=delta_pnl,
        network_pnl=network_pnl,
    )

    print(
        "Graphique enregistré dans "
        "figures/pnl_comparison.png"
    )


if __name__ == "__main__":
    main()