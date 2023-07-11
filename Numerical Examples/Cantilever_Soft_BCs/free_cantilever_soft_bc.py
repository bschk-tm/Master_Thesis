import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import Dataset


# building a PINN Model using subclassing 
# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

class CantileverPINN(tf.keras.Model):
    """ PINN for 1D free cantilever bending for primary unknown w(x) """
    
    def __init__(self,E = 1.0, I = 1.0, p = 1.0):
        """ Input: 
            - E: Young's Modulus for the 1D elasto-static problem [N/m²]
            - I: Area moment of inertia [m⁴]
            - p: Line Load on the cantilever [kN/m] """
        
        super(CantileverPINN, self).__init__()
        self.E = E
        self.I = I
        self.p = p
        
        # builds a network's layers while instanciation
        self.build(input_shape=1) 

    def build(self,input_shape):
        """Building the Network Layers using the Keras Functional API
           https://www.tensorflow.org/guide/keras/functional_api#when_to_use_the_functional_api

           Topology according to Katsikis et al. without hyperparameter tuning 
           Layer's Output is directly given as the next Layer's Input
        """
        input_layer    = tf.keras.layers.Input(shape=input_shape, name='Input_Layer')
        hidden_layer_1 = tf.keras.layers.Dense(units=15,activation='tanh',name='Hidden_Layer_1')(input_layer)
        hidden_layer_2 = tf.keras.layers.Dense(units=30,activation='tanh',name='Hidden_Layer_2')(hidden_layer_1)
        hidden_layer_3 = tf.keras.layers.Dense(units=60,activation='tanh',name='Hidden_Layer_2')(hidden_layer_2)
        output_layer   = tf.keras.layers.Dense(units=1,name='Output_Layer')(hidden_layer_3) # 1 primary unknown -> 1 output neuron

        self.ann = tf.keras.Model(inputs=input_layer,outputs=output_layer, name='ann')

    def compute_loss(self, inputs):
        """ Calculates the PDE Residuals and BC Residuals
            - pde_residual = w_xxxx - p(x_i)/EI
            - bc_residual_x0_w = w
            - bc_residual_x0_w_x = w_x 
            - bc_residual_xL_M = - w_xx * EI 
            - bc_residual_xL_V = - w_xxx * EI
        """
        gradients = self.compute_gradients(inputs)
        w      = gradients[0]
        w_x    = gradients[1]
        w_xx   = gradients[2]
        w_xxx  = gradients[3]
        w_xxxx = gradients[4]

        pde_residual = w_xxxx - (self.p/(self.E * self.I))
        bc_residual_x0_w = w
        bc_residual_x0_w_x = w_x
        bc_residual_xL_M = - w_xx  * self.E * self.I
        bc_residual_xL_V = - w_xxx * self.E * self.I 

        


    def call(self, inputs):
        """Forward step:
           calculation of the PDE and BC Residuals at every collocation point of the beam
           Propagation of the layer outputs through the following layers
           Input:
           - inputs: x-coordinates of the 1D mechanical system
        """
        pass
       

       
    
    def compute_gradients(self,inputs):
        """Computes the first derivative of a variable w.r.t. the input using Automatic Differentiation

        Input:
        - inputs:

        Output:
        - list with all gradients, including w
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
        w = self.ann(inputs)
        w_x = tape.gradient(w, inputs)
        w_xx = tape.gradient(w_x, inputs)
        w_xxx = tape.gradient(w_xx, inputs)
        w_xxxx = tape.gradient(w_xxx, inputs)

        return [w,w_x,w_xx,w_xxx,w_xxxx]

    
   
     

        

    
        








        

