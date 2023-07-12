import tensorflow as tf
import matplotlib.pyplot as plt


# building PINN Model using subclassing 
# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

class CantileverPINN(tf.keras.Model):
    """ PINN for 1D free cantilever bending, solving for primary unknown w(x) """
    
    def __init__(self,E = 1.0, I = 1.0, p = 1.0, L=1.0):
        """ Input: 
            - E: Young's Modulus for the 1D elasto-static problem [N/m²]
            - I: Area moment of inertia [m⁴]
            - p: Line Load on the cantilever [N/m]
            - L: Model Length [m]
        """
        
        super(CantileverPINN, self).__init__()
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
        self.hidden_layer_3 = tf.keras.layers.Dense(units=60,activation='tanh',name='Hidden_Layer_2')
        self.output_layer   = tf.keras.layers.Dense(units=1,name='Output_Layer') # 1 primary unknown -> 1 output neuron

        # ensure correct initialization of weights and biases based on provided input_shape 
        super(CantileverPINN, self).build([input_shape])

    
    # def compute_gradients(self,x):
    #     """Computes up to the 4th derivative of "w(x)" w.r.t. the inputs "x" using Automatic Differentiation
    #        4th Derivative -> 4-times nested Gradient Tape
    #        https://www.tensorflow.org/api_docs/python/tf/GradientTape
    #     """
    #     x = tf.reshape(x,(-1,1)) # considering input shape for the layers has to be min_dim = 2
    #     with tf.GradientTape(persistent=True) as tape: # fourth derivative
    #         tape.watch(x)
    #         with tf.GradientTape(persistent=True) as tape2:
    #             tape2.watch(x)
    #             with tf.GradientTape(persistent=True) as tape3:
    #                 tape3.watch(x)
    #                 with tf.GradientTape(persistent=True) as tape4: # first derivative
    #                     tape4.watch(x)

    #                     # calculation of network output w(x)
    #                     h_1 = self.hidden_layer_1(x)
    #                     h_2 = self.hidden_layer_2(h_1)
    #                     h_3 = self.hidden_layer_3(h_2)
    #                     w = self.output_layer(h_3)

    #                 w_x = tape4.gradient(w, x) # gradient of w(x) w.r.t x
    #             w_xx = tape3.gradient(w_x, x)
    #         w_xxx = tape2.gradient(w_xx,x)
    #     w_xxxx = tape.gradient(w_xxx, x)

    #     return [w,w_x,w_xx,w_xxx,w_xxxx]
       
       


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
        # gradients = self.compute_gradients(inputs)
        # w      = gradients[0]
        # w_x    = gradients[1]
        # w_xx   = gradients[2]
        # w_xxx  = gradients[3]
        # w_xxxx = gradients[4]

        x = tf.reshape(inputs,(-1,1)) # considering input shape for the layers has to be min_dim = 2
        with tf.GradientTape(persistent=True) as tape: # fourth derivative
            tape.watch(x)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x)
                    with tf.GradientTape(persistent=True) as tape4: # first derivative
                        tape4.watch(x)

                        # calculation of network output w(x)
                        h_1 = self.hidden_layer_1(x)
                        h_2 = self.hidden_layer_2(h_1)
                        h_3 = self.hidden_layer_3(h_2)
                        w = self.output_layer(h_3)

                    w_x = tape4.gradient(w, x) # gradient of w(x) w.r.t x
                w_xx = tape3.gradient(w_x, x)
            w_xxx = tape2.gradient(w_xx,x)
        w_xxxx = tape.gradient(w_xxx, x)

        pde_residual = w_xxxx - self.p/(self.E * self.I)
        bc_residual_x0_w = w
        bc_residual_x0_w_x = w_x
        bc_residual_xL_M = - w_xx  * self.E * self.I
        bc_residual_xL_V = - w_xxx * self.E * self.I 

        print(w)

        #loss = tf.reduce_mean(tf.square(pde_residual) + tf.square(bc_residual_x0_w) + tf.square(bc_residual_x0_w_x) + tf.square(bc_residual_xL_M) + tf.square(bc_residual_xL_V))
        loss = tf.reduce_mean(tf.square(pde_residual))
        
        self.add_loss(loss)
        return loss
       
    
        

if __name__ == "__main__":

    # instanciate Model
    pinn = CantileverPINN(E=200000000000,I=0.000038929334,p=60000,L=2.7)
    pinn.compile(optimizer=tf.keras.optimizers.Nadam())
    pinn.summary()

    # set up input data, 10000 datapoints
    coords = tf.linspace(0.0,pinn.Length, 100)
    history = pinn.fit(coords,epochs=100)

    # plotting results
    plt.figure(0)
    plt.title("Loss Curve over Epochs")
    plt.semilogy(history.history["loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid()

   

    




        

    
   
     

        

    
        








        

