import torch

from src.hedging_network import (
    HedgingNetwork,
)


RANDOM_SEED = 42


def count_parameters(
    network: HedgingNetwork,
) -> int:
    """
    Compte le nombre total de paramètres
    ajustables du réseau.
    """

    return sum(
        parameter.numel()
        for parameter in network.parameters()
    )


def main() -> None:
    torch.manual_seed(
        RANDOM_SEED
    )

    network = HedgingNetwork(
        hidden_size=16
    )

    print("Architecture du réseau :")
    print(network)

    print()
    print(
        "Nombre de paramètres :",
        count_parameters(network),
    )

    states = torch.tensor(
        [
            # Prix au strike,
            # un an restant,
            # aucune position actuelle.
            [1.00, 1.00, 0.00],

            # Prix 10 % au-dessus du strike,
            # la moitié du temps restant,
            # position actuelle de 0,50.
            [1.10, 0.50, 0.50],

            # Prix 10 % sous le strike,
            # peu de temps restant,
            # position actuelle de 0,20.
            [0.90, 0.10, 0.20],
        ],
        dtype=torch.float32,
    )

    print()
    print("Situations données au réseau :")
    print(states)

    print()
    print("Forme des entrées :")
    print(states.shape)

    # Pour cette simple démonstration,
    # nous ne calculons pas de gradients.
    with torch.no_grad():
        hedge_positions = network(
            states
        )

    print()
    print("Positions produites :")
    print(hedge_positions)

    print()
    print("Forme des sorties :")
    print(hedge_positions.shape)

    print()
    print(
        "Attention : le réseau n'est pas entraîné."
    )

    print(
        "Les positions affichées sont donc "
        "aléatoires et ne doivent pas être "
        "interprétées financièrement."
    )


if __name__ == "__main__":
    main()