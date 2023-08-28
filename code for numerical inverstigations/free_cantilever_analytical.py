#import matplotlib.pyplot as plt
def free_cantilever_analytical(x, q, E, I, L):
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
    w = ((q*L**4)/(24*E*I)) * (-(x/L)**4 + 4*(x/L)**3 - 6*(x/L)**2)
    
    return w


# plt.figure()
# y_ana = free_cantilever_analytical(X_test, p, E, I, L)
# plt.plot(X_test, y_ana)
# plt.grid()
# plt.legend()
# plt.show()

