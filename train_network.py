from pathlib import Path

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
    simulate_hedging_pnl,
)

from src.hedging_network import (
    HedgingNetwork,
)

from src.losses import (
    residual_pnl_variance,
)


INITIAL_PRICE = 100.0
STRIKE = 100.0

RISK_FREE_RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

NUMBER_OF_STEPS = 30

HIDDEN_SIZE = 16

BATCH_SIZE = 2048
NUMBER_OF_ITERATIONS = 3000

LEARNING_RATE = 0.001

PRINT_EVERY = 100

TRAINING_SEED = 42
VALIDATION_SEED = 123

NUMBER_OF_VALIDATION_PATHS = 10_000


def evaluate_network(
    network: HedgingNetwork,
    validation_paths: torch.Tensor,
    option_premium: float,
) -> dict[str, float]:
    """
    Évalue le réseau sur des trajectoires fixes
    qui ne sont pas utilisées pour modifier ses poids.
    """

    network.eval()

    with torch.no_grad():

        results = simulate_hedging_pnl(
            network=network,
            paths=validation_paths,
            strike=STRIKE,
            risk_free_rate=RISK_FREE_RATE,
            maturity=MATURITY,
            option_premium=option_premium,
        )

        pnl = results["pnl"]

        loss = residual_pnl_variance(
            pnl
        )

    return {
        "loss": loss.item(),
        "mean_pnl": pnl.mean().item(),
        "pnl_std": pnl.std(
            unbiased=False
        ).item(),
    }


def plot_training_history(
    training_losses: list[float],
    validation_iterations: list[int],
    validation_losses: list[float],
) -> None:
    """
    Trace l'évolution de la loss pendant
    l'entraînement.
    """

    Path("figures").mkdir(
        exist_ok=True
    )

    training_iterations = range(
        1,
        len(training_losses) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        training_iterations,
        training_losses,
        alpha=0.5,
        label="Loss sur le batch d'entraînement",
    )

    plt.plot(
        validation_iterations,
        validation_losses,
        marker="o",
        label="Loss sur les trajectoires de validation",
    )

    plt.title(
        "Convergence de la loss de deep hedging"
    )

    plt.xlabel(
        "Étape d'entraînement"
    )

    plt.ylabel(
        "Variance du PnL final"
    )

    plt.grid()
    plt.legend()

    plt.savefig(
        "figures/training_loss.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def save_model(
    network: HedgingNetwork,
    final_validation_statistics: dict[str, float],
) -> None:
    """
    Sauvegarde les poids et les principaux paramètres
    nécessaires pour reconstruire le réseau.
    """

    Path("models").mkdir(
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": (
            network.state_dict()
        ),
        "hidden_size": HIDDEN_SIZE,
        "initial_price": INITIAL_PRICE,
        "strike": STRIKE,
        "risk_free_rate": RISK_FREE_RATE,
        "volatility": VOLATILITY,
        "maturity": MATURITY,
        "number_of_steps": NUMBER_OF_STEPS,
        "final_validation_statistics": (
            final_validation_statistics
        ),
    }

    torch.save(
        checkpoint,
        "models/hedging_network.pt",
    )


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

    # Création d'un ensemble fixe de validation.
    #
    # Ces trajectoires servent uniquement à mesurer
    # les progrès du réseau. Elles ne modifient
    # jamais ses paramètres.
    torch.manual_seed(
        VALIDATION_SEED
    )

    validation_paths = simulate_gbm_paths(
        initial_price=INITIAL_PRICE,
        risk_free_rate=RISK_FREE_RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        number_of_steps=NUMBER_OF_STEPS,
        number_of_paths=(
            NUMBER_OF_VALIDATION_PATHS
        ),
    )

    # Nouvelle graine pour les batches
    # utilisés pendant l'entraînement.
    torch.manual_seed(
        TRAINING_SEED
    )

    network = HedgingNetwork(
        hidden_size=HIDDEN_SIZE
    )

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE,
    )

    initial_statistics = evaluate_network(
        network=network,
        validation_paths=validation_paths,
        option_premium=option_premium,
    )

    print()
    print("Avant entraînement :")

    print(
        f"Loss de validation : "
        f"{initial_statistics['loss']:.6f}"
    )

    print(
        f"PnL moyen : "
        f"{initial_statistics['mean_pnl']:.6f} €"
    )

    print(
        f"Écart-type du PnL : "
        f"{initial_statistics['pnl_std']:.6f} €"
    )

    training_losses: list[float] = []

    validation_iterations = [0]

    validation_losses = [
        initial_statistics["loss"]
    ]

    print()
    print("Début de l'entraînement")
    print()

    for iteration in range(
        1,
        NUMBER_OF_ITERATIONS + 1,
    ):
        network.train()

        # Un nouveau batch de trajectoires
        # est simulé à chaque étape.
        training_paths = simulate_gbm_paths(
            initial_price=INITIAL_PRICE,
            risk_free_rate=RISK_FREE_RATE,
            volatility=VOLATILITY,
            maturity=MATURITY,
            number_of_steps=NUMBER_OF_STEPS,
            number_of_paths=BATCH_SIZE,
        )

        # Efface les gradients calculés
        # à l'étape précédente.
        optimizer.zero_grad()

        # Forward pass :
        # le réseau choisit toutes les positions
        # et on calcule les PnL finaux.
        hedging_results = simulate_hedging_pnl(
            network=network,
            paths=training_paths,
            strike=STRIKE,
            risk_free_rate=RISK_FREE_RATE,
            maturity=MATURITY,
            option_premium=option_premium,
        )

        pnl = hedging_results["pnl"]

        loss = residual_pnl_variance(
            pnl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "La loss est devenue non finie."
            )

        # Backpropagation :
        # calcul des gradients de tous les poids.
        loss.backward()

        # Adam modifie les poids à partir
        # des gradients calculés.
        optimizer.step()

        training_losses.append(
            loss.item()
        )

        if (
            iteration == 1
            or iteration % PRINT_EVERY == 0
            or iteration == NUMBER_OF_ITERATIONS
        ):
            validation_statistics = (
                evaluate_network(
                    network=network,
                    validation_paths=(
                        validation_paths
                    ),
                    option_premium=(
                        option_premium
                    ),
                )
            )

            validation_iterations.append(
                iteration
            )

            validation_losses.append(
                validation_statistics["loss"]
            )

            print(
                f"Étape "
                f"{iteration:4d} / "
                f"{NUMBER_OF_ITERATIONS}"
            )

            print(
                f"  Loss entraînement : "
                f"{loss.item():.6f}"
            )

            print(
                f"  Loss validation   : "
                f"{validation_statistics['loss']:.6f}"
            )

            print(
                f"  PnL moyen validation : "
                f"{validation_statistics['mean_pnl']:.6f} €"
            )

            print(
                f"  Écart-type validation : "
                f"{validation_statistics['pnl_std']:.6f} €"
            )

            print()

    final_statistics = evaluate_network(
        network=network,
        validation_paths=validation_paths,
        option_premium=option_premium,
    )

    save_model(
        network=network,
        final_validation_statistics=(
            final_statistics
        ),
    )

    plot_training_history(
        training_losses=training_losses,
        validation_iterations=(
            validation_iterations
        ),
        validation_losses=(
            validation_losses
        ),
    )

    print("Entraînement terminé.")
    print()

    print(
        f"Loss initiale de validation : "
        f"{initial_statistics['loss']:.6f}"
    )

    print(
        f"Loss finale de validation : "
        f"{final_statistics['loss']:.6f}"
    )

    print(
        f"Écart-type final du PnL : "
        f"{final_statistics['pnl_std']:.6f} €"
    )

    print()

    print(
        "Modèle enregistré dans "
        "models/hedging_network.pt"
    )

    print(
        "Graphique enregistré dans "
        "figures/training_loss.png"
    )


if __name__ == "__main__":
    main()