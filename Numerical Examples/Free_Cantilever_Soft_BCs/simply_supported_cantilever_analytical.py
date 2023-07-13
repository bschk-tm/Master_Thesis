def analytical_solution(x, q, E, I, L):
    """
    Calculates the analytical solution of the beam bending PDE.
    
    Args:
        x (numpy.ndarray): Spatial coordinates.
        q (float): Line load.
        E (float): Young's modulus.
        I (float): Moment of inertia.
        L (float): System Length
    
    Returns:
        numpy.ndarray: Analytical solution of the beam bending PDE.
    """
    w = q / (24*E*I) * (- x**4 + 5/2 * L * x**3 - 3/2 * x**2 )
    return w


