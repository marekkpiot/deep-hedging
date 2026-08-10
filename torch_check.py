import torch


def main() -> None:
    """
    Vérifie que PyTorch est correctement installé
    et présente les premières notions de tenseur.
    """

    print("Version de PyTorch :")
    print(torch.__version__)

    print()
    print("CUDA disponible :")
    print(torch.cuda.is_available())

    numbers = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    print()
    print("Premier tenseur :")
    print(numbers)

    print()
    print("Type du tenseur :")
    print(numbers.dtype)

    print()
    print("Forme du tenseur :")
    print(numbers.shape)

    doubled_numbers = 2.0 * numbers

    print()
    print("Tenseur multiplié par 2 :")
    print(doubled_numbers)

    matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    print()
    print("Matrice :")
    print(matrix)

    print()
    print("Forme de la matrice :")
    print(matrix.shape)


if __name__ == "__main__":
    main()