import tensorflow as tf
from dataset import Dataset


# building an ANN with subclassing 

class CantileverPINN(tf.keras.Model):
    """ PINN for 1D free cantilever """
    
    def __init__(self,E = 1.0, I = 1.0, p = 1.0 ):
        """ Input: 
            - E: Young's Modulus for the 1D elasto-static problem [N/m²]
            - I: Area moment of inertia [m⁴]
            - p: Line Load on the cantilever [kN/m] """
        
        super(CantileverPINN, self).__init__()
        self.E = E
        self.I = I
        self.p = p

       

