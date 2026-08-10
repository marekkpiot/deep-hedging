import torch
from torch import nn


class HedgingNetwork(nn.Module):
    """
    Réseau de neurones qui transforme l'état actuel
    du marché en une position de couverture.

    Les trois entrées sont :

    1. prix de l'action divisé par le strike ;
    2. proportion de temps restant ;
    3. position de couverture actuelle.

    La sortie est comprise entre 0 et 1.
    """

    def __init__(
        self,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                "Le nombre de neurones cachés "
                "doit être positif."
            )

        self.network = nn.Sequential(
            nn.Linear(
                in_features=3,
                out_features=hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=hidden_size,
                out_features=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Effectue le passage des entrées dans le réseau.

        La forme attendue de state est :

        nombre de situations
        ×
        3 variables
        """

        if state.ndim != 2:
            raise ValueError(
                "Le tenseur state doit avoir "
                "deux dimensions."
            )

        if state.shape[1] != 3:
            raise ValueError(
                "Chaque situation doit contenir "
                "exactement trois variables."
            )

        hedge_position = self.network(
            state
        )

        # Avant squeeze :
        # [nombre de situations, 1]
        #
        # Après squeeze :
        # [nombre de situations]
        return hedge_position.squeeze(-1)