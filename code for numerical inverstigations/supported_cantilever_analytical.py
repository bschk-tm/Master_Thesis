import matplotlib.pyplot as plt
#from Dataset import X_test, p, E, I, L
import numpy as np

def supported_cantilever_analytical(x, q, E, I, L):
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
    w = ((q*L**4)/(24*E*I))*(-(x/L)**4 + (5/2)*(x/L)**3 - (3/2)*(x/L)**2)
    return w


# plt.figure()
# y_ana = supported_cantilever_analytical(X_test, p, E, I, L)
# plt.plot(X_test, y_ana)
# plt.grid()
# plt.legend()
# plt.show()
