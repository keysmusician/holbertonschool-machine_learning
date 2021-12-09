"""Defines `poly_derivative`."""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial from a list of its coefficients.
    """
    if not poly:
        return None

    derivative = [i * n for i, n in enumerate(poly)][1:]
    if not derivative:
        return [0]
