import math


def standard_normal_cdf(
    value: float,
) -> float:
    """
    Fonction de répartition de la loi normale standard.

    Elle renvoie la probabilité qu'une variable normale
    standard soit inférieure à value.
    """

    return 0.5 * (
        1.0
        + math.erf(
            value / math.sqrt(2.0)
        )
    )


def black_scholes_call_price(
    initial_price: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """
    Calcule le prix Black-Scholes d'un call européen
    sans dividende.
    """

    if initial_price <= 0:
        raise ValueError(
            "Le prix initial doit être positif."
        )

    if strike <= 0:
        raise ValueError(
            "Le strike doit être positif."
        )

    if volatility <= 0:
        raise ValueError(
            "La volatilité doit être positive."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être positive."
        )

    volatility_over_period = (
        volatility * math.sqrt(maturity)
    )

    d1 = (
        math.log(initial_price / strike)
        + (
            risk_free_rate
            + 0.5 * volatility**2
        )
        * maturity
    ) / volatility_over_period

    d2 = (
        d1 - volatility_over_period
    )

    call_price = (
        initial_price
        * standard_normal_cdf(d1)
        - strike
        * math.exp(
            -risk_free_rate * maturity
        )
        * standard_normal_cdf(d2)
    )

    return call_price

def black_scholes_call_delta(
    stock_price: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float:
    """
    Calcule le delta Black-Scholes
    d'un call européen sans dividende.
    """

    if stock_price <= 0:
        raise ValueError(
            "Le prix de l'action doit être positif."
        )

    if strike <= 0:
        raise ValueError(
            "Le strike doit être positif."
        )

    if volatility <= 0:
        raise ValueError(
            "La volatilité doit être positive."
        )

    if time_to_maturity <= 0:
        # À l'échéance, le delta devient
        # essentiellement 0 ou 1.
        if stock_price > strike:
            return 1.0

        if stock_price < strike:
            return 0.0

        return 0.5

    d1 = (
        math.log(stock_price / strike)
        + (
            risk_free_rate
            + 0.5 * volatility**2
        )
        * time_to_maturity
    ) / (
        volatility
        * math.sqrt(time_to_maturity)
    )

    delta = standard_normal_cdf(d1)

    return delta