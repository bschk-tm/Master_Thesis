import tensorflow as tf
import numpy as np

# building PINN Model using subclassing 
# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

class Supported_CantileverPINN_tuned(tf.keras.Model):
    """ PINN for 1D supported cantilever bending, solving for primary unknown w(x) """
    
    def __init__(self,units_per_layer = [15,15,15],activation = 'tanh', dropout_rate= 0.2, E = 1.0, I = 1.0, p = 1.0, L=1.0):
        """ Input: 
            - E: Young's Modulus for the 1D elasto-static problem [N/m²]
            - I: Area moment of inertia [m⁴]
            - p: Line Load on the cantilever [N/m]
            - L: Model Length [m]
        """
        
        super(Supported_CantileverPINN_tuned, self).__init__()
        self.E = E
        self.I = I
        self.p = p
        self.Length = L
        
        self.units_per_layer = units_per_layer
        self.activation = activation
        self.droput_rate = dropout_rate
        self.regularization = True
        
        # builds a network's layers while instanciation
        self.build(input_shape=1) 

    def build(self,input_shape):
        """Building the Network Layers using the Keras Functional API
           https://www.tensorflow.org/guide/keras/functional_api#when_to_use_the_functional_api

           Topology according to Katsikis et al. without hyperparameter tuning 
           Layer's Output is directly given as the next Layer's Input
        """
        self.hidden_layers = []
        for units in self.units_per_layer:
            self.hidden_layers.append(tf.keras.layers.Dense(units=units,activation=self.activation))
            self.hidden_layers.append(tf.keras.layers.Dropout(self.droput_rate)) # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
   
        self.output_layer   = tf.keras.layers.Dense(units=1,name='Output_Layer') # 1 primary unknown -> 1 output neuron

        # ensure correct initialization of weights and biases based on provided input_shape 
        super(Supported_CantileverPINN_tuned, self).build([input_shape])


    
    def compute_gradients(self,inputs, training):
        """Computes up to the 4th derivative of "w(x)" w.r.t. the inputs "x" using Automatic Differentiation
        4th Derivative -> 4-times nested Gradient Tape
        https://www.tensorflow.org/api_docs/python/tf/GradientTape
        """
        x = tf.reshape(inputs,(-1,1)) # considering input shape for the layers has to be min_dim = 2
        input = x

        with tf.GradientTape(persistent=True) as tape: # fourth derivative
            tape.watch(input)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(input)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(input)
                    with tf.GradientTape(persistent=True) as tape4: # first derivative
                        tape4.watch(input)

                        # calculation of network output w(x) via propagating the input through the layers
                        for layer in self.hidden_layers:
                            x = layer(x, training = training)
                        w = self.output_layer(x)

                    w_x = tape4.gradient(w, input) # gradient of w(x) w.r.t x
                w_xx = tape3.gradient(w_x, input)
            w_xxx = tape2.gradient(w_xx,input)
        w_xxxx = tape.gradient(w_xxx, input)

        return [w,w_x,w_xx,w_xxx,w_xxxx]
       
    # Setting training = False per default enables using dropout only for training purposes
    # During training, the fit-method sets training variable appropriately to True automatically
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

    def call(self, inputs, training = False):
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
        gradients_field = self.compute_gradients(inputs, training)
        gradients_left_bc = self.compute_gradients(x_left_bc, training)
        gradients_right_bc = self.compute_gradients(x_right_bc, training)
        
        # needed for residuals on the left boundary
        w_0  = gradients_left_bc[0]
        w_0_x = gradients_left_bc[1]

        # needed for residuals on the right boundary
        m_L  = gradients_right_bc[2]
        w_L = gradients_right_bc[0]

        # needed for the pde residual in the whole field
        w_xxxx = gradients_field[4]     

        pde_residual = w_xxxx + self.p/(self.E * self.I)
        bc_residual_w_0 = w_0 
        bc_residual_w_0_x = w_0_x
        bc_residual_M_L = m_L  
        bc_residual_w_L = w_L 

        pde_loss = tf.reduce_mean(tf.square(pde_residual))
        left_bc_w_loss = tf.square(bc_residual_w_0)
        left_bc_w_x_loss = tf.square(bc_residual_w_0_x)
        right_bc_M_loss = tf.square(bc_residual_M_L)
        right_bc_w_loss = tf.square(bc_residual_w_L)
       
        loss = pde_loss + left_bc_w_loss + left_bc_w_x_loss + right_bc_M_loss + right_bc_w_loss
   
        return loss, pde_loss, left_bc_w_loss, left_bc_w_x_loss, right_bc_M_loss, right_bc_w_loss       
    
    def predict(self, inputs, training = False):
        """Makes predictions based on the trained model parameters"""
        x = tf.reshape(inputs,(-1,1))
        for layer in self.hidden_layers:
            x = layer(x,training = training)
        self.prediction = self.output_layer(x)

        return self.prediction 
    
    def evaluation(self, X_test, y_test):
        """ Evaluates Model Performance on unseen test data compared to the analytical solution
            Returns the Absolute Error between Prediction and Analytical Solution as well as
            Loss Value for the Model on Test Data according to customized Loss Function
        Input:
        - x_test: spatial coordinates of the collocation points, where the system should be evaluated
        - y_test: targets related to the spatial coordinates according to the analytical solution
        """

        # prediction on the test dataset
        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        y_pred = tf.squeeze(self.predict(X_test))
        #print([y_test.shape,y_pred.shape])

        
        
        abs_error = np.abs((y_pred - y_test[0]))
        #comparison to the targets (here: analytical solution)
        composite_loss_on_test_data = self.evaluate(x=X_test,y=y_test, verbose=0)[0] # scalar test loss value acc. to custom loss function
        #print([abs_error.shape, abs_error[0]])


        return [abs_error, composite_loss_on_test_data] 