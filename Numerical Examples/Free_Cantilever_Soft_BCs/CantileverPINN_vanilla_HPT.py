import tensorflow as tf
import numpy as np

# building PINN Model using subclassing 
# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class
# https://www.tensorflow.org/api_docs/python/tf/keras/Mod

class CantileverPINN_vanilla_HPT(tf.keras.Model):
    """ PINN for 1D free cantilever bending, solving for primary unknown w(x) """
    
    def __init__(self,E = 1.0, I = 1.0, p = 1.0, L=1.0):
        """ Input: 
            - E: Young's Modulus for the 1D elasto-static problem [N/m²]
            - I: Area moment of inertia [m⁴]
            - p: Line Load on the cantilever [N/m]
            - L: Model Length [m]
        """
        
        super(CantileverPINN_vanilla_HPT, self).__init__()
        self.E = E
        self.I = I
        self.p = p
        self.Length = L
        
        # builds a network's layers while instanciation
        self.build(input_shape=1) 

    def build(self,input_shape):
        """Building the Network Layers using the Keras Functional API
           https://www.tensorflow.org/guide/keras/functional_api#when_to_use_the_functional_api

           Topology according to Katsikis et al. without hyperparameter tuning 
           Layer's Output is directly given as the next Layer's Input
        """
  
        self.hidden_layer_1 = tf.keras.layers.Dense(units=15,activation='tanh',name='Hidden_Layer_1')
        self.hidden_layer_2 = tf.keras.layers.Dense(units=30,activation='tanh',name='Hidden_Layer_2')
        self.hidden_layer_3 = tf.keras.layers.Dense(units=60,activation='tanh',name='Hidden_Layer_3')
        self.output_layer   = tf.keras.layers.Dense(units=1,name='Output_Layer') # 1 primary unknown -> 1 output neuron

        # ensure correct initialization of weights and biases based on provided input_shape 
        super(CantileverPINN_vanilla_HPT, self).build([input_shape])


    
    def compute_gradients(self,inputs):
        """Computes up to the 4th derivative of "w(x)" w.r.t. the inputs "x" using Automatic Differentiation
        4th Derivative -> 4-times nested Gradient Tape
        https://www.tensorflow.org/api_docs/python/tf/GradientTape
        """
        x = tf.reshape(inputs,(-1,1)) # considering input shape for the layers has to be min_dim = 2


        with tf.GradientTape(persistent=True) as tape: # fourth derivative
            tape.watch(x)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x)
                    with tf.GradientTape(persistent=True) as tape4: # first derivative
                        tape4.watch(x)

                        # calculation of network output w(x) via propagating the input through the layers
                        h1 = self.hidden_layer_1(x)
                        h2 = self.hidden_layer_2(h1)
                        h3 = self.hidden_layer_3(h2)
                        w = self.output_layer(h3)

                    w_x = tape4.gradient(w, x) # gradient of w(x) w.r.t x
                w_xx = tape3.gradient(w_x, x)
            w_xxx = tape2.gradient(w_xx,x)
        w_xxxx = tape.gradient(w_xxx, x)

        return [w,w_x,w_xx,w_xxx,w_xxxx]
       

    def call(self, inputs):
        """Forward step:
           calculation of the PDE and BC Residuals at every collocation point of the beam  

            - pde_residual = w_xxxx - p(x_i)/EI
            - bc_residual_x0_w = w
            - bc_residual_x0_w_x = w_x 
            - bc_residual_xL_M = - w_xx * EI 
            - bc_residual_xL_V = - w_xxx * EI

           Input:
           - inputs: x-coordinates of the 1D mechanical system
        """

        # input datasets for compliance of the bc's
        x_left_bc = tf.zeros_like(inputs)
        x_right_bc = tf.ones_like(inputs) * self.Length

        # compute gradients for different input datasets
        gradients_field = self.compute_gradients(inputs)
        gradients_left_bc = self.compute_gradients(x_left_bc)
        gradients_right_bc = self.compute_gradients(x_right_bc)
        
        # needed for residuals on the left boundary
        w_0  = gradients_left_bc[0]
        w_0_x = gradients_left_bc[1]

        # needed for residuals on the right boundary
        m_L  = gradients_right_bc[2]
        v_L = gradients_right_bc[3]

        # needed for the pde residual in the whole field
        w_xxxx = gradients_field[4]     

        pde_residual = w_xxxx + self.p/(self.E * self.I)
        bc_residual_w_0 = w_0 
        bc_residual_w_0_x = w_0_x
        bc_residual_M_L = m_L  
        bc_residual_V_L = v_L 

        pde_loss = tf.reduce_mean(tf.square(pde_residual))
        left_bc_w_loss = tf.square(bc_residual_w_0)
        left_bc_w_x_loss = tf.square(bc_residual_w_0_x)
        right_bc_M_loss = tf.square(bc_residual_M_L)
        right_bc_V_loss = tf.square(bc_residual_V_L)
       
        loss = pde_loss + left_bc_w_loss + left_bc_w_x_loss + right_bc_M_loss + right_bc_V_loss
   
        return loss, pde_loss, left_bc_w_loss, left_bc_w_x_loss, right_bc_M_loss, right_bc_V_loss       
    
    def predict(self, inputs):
        """Makes predictions based on the trained model parameters"""
        x = tf.reshape(inputs,(-1,1))
        h1 = self.hidden_layer_1(x)
        h2 = self.hidden_layer_2(h1)
        h3 = self.hidden_layer_3(h2)
        self.prediction = self.output_layer(h3)

        return self.prediction 
    
    def evaluation(self, x_test, y_test):
        """ Evaluates Model Performance on unseen test data in comparison to the analytical solution
            Returns the Loss Value for the Model in Test Mode according to MAE and the abosulte error on each collocation point
        Input:
        - x_test: spatial coordinates of the collocation points, where the system should be evaluated
        - y_test: targets related to the spatial coordinates according to the analytical solution
        """

        # prediction on the test dataset
        y_pred = tf.squeeze(self.predict(x_test))
        
        abs_error = np.abs((y_pred - y_test))
        # comparison to the targets (here: analytical solution)
        mse_loss = np.mean(np.square(abs_error)) # scalar loss value acc. to MSE

        return [abs_error, mse_loss] 