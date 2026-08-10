from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.gbm import simulate_gbm_paths


INITIAL_PRICE = 100.0
RISK_FREE_RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

NUMBER_OF_STEPS = 30
NUMBER_OF_PATHS = 5000

RANDOM_SEED = 42


def display_information(
    paths: torch.Tensor,
) -> None:
    """
    Affiche les caractéristiques principales
    des trajectoires simulées.
    """

    terminal_prices = paths[:, -1]

    empirical_terminal_mean = (
        terminal_prices.mean().item()
    )

    theoretical_terminal_mean = (
        INITIAL_PRICE
        * math.exp(
            RISK_FREE_RATE * MATURITY
        )
    )

    print("Forme du tenseur :")
    print(paths.shape)

    print()
    print("Première trajectoire :")
    print(paths[0])

    print()
    print("Cinq premiers prix initiaux :")
    print(paths[:5, 0])

    print()
    print("Cinq premiers prix terminaux :")
    print(paths[:5, -1])

    print()
    print(
        "Moyenne empirique des prix terminaux :"
    )

    print(
        f"{empirical_terminal_mean:.4f}"
    )

    print()
    print(
        "Moyenne théorique des prix terminaux :"
    )

    print(
        f"{theoretical_terminal_mean:.4f}"
    )


def plot_paths(
    paths: torch.Tensor,
) -> None:
    """
    Trace les premières trajectoires simulées.
    """

    Path("figures").mkdir(exist_ok=True)

    paths_as_array = (
        paths
        .detach()
        .cpu()
        .numpy()
    )

    number_of_plotted_paths = 20

    plt.figure(figsize=(10, 6))

    for path_index in range(
        number_of_plotted_paths
    ):
        plt.plot(
            paths_as_array[path_index]
        )

    plt.title(
        "Trajectoires simulées du prix de l'action"
    )

    plt.xlabel("Étape temporelle")
    plt.ylabel("Prix de l'action")
    plt.grid()

    plt.savefig(
        "figures/gbm_paths.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    torch.manual_seed(
        RANDOM_SEED
    )

    paths = simulate_gbm_paths(
        initial_price=INITIAL_PRICE,
        risk_free_rate=RISK_FREE_RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        number_of_steps=NUMBER_OF_STEPS,
        number_of_paths=NUMBER_OF_PATHS,
    )

    display_information(paths)

    plot_paths(paths)

    print()
    print(
        "Graphique enregistré dans "
        "figures/gbm_paths.png"
    )


if __name__ == "__main__":
    main()