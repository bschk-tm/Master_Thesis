import tensorflow as tf
import matplotlib.pyplot as plt

class PINN(tf.keras.Model):
    """PINN for a exemplary 1D linear elasticity mechanical system with hard boundary conditions

    Input:
          - youngs_modulus: Young's modulus for linear-elastic behavior
          - force: The force acting on the rod to derive Neumann bc
          - area: The cross-section area of the construction element
    """

    def __init__(self, youngs_modulus=1.0, force=1.0, area=1.0):
        super(PINN, self).__init__()
        self.E = youngs_modulus
        self.Force = force
        self.area = area
        self.sig = self.Force / self.area # value for the Neumann bc

        self.build(input_shape=1) # since we are in 1D the input feature dimension is 1

    def build(self, input_shape):
        """Builds the artificial neural network for hard bc's (Dirichlet + Neumann) so 2 Output Neurons has to be provided   """
       
        input_layer = tf.keras.layers.Input(shape=input_shape, name="input_layer")
        hidden_layer = tf.keras.layers.Dense(units=32, activation="tanh", name="hidden_layer_0")(input_layer)
        hidden_layer = tf.keras.layers.Dense(units=32, activation="tanh", name="hidden_layer_1")(hidden_layer)
        output_layer = tf.keras.layers.Dense(units=2, name="output_layer")(hidden_layer)

        G_u = 0.0 # to fullfil Dirichlet BC
        D_u = input_layer[:, 0] # choose D_u so, that u_PINN vanishes where Dirichlet bc is located -> D_u(x=0) != 0
        u = G_u + D_u * output_layer[:, 0] # displacement is the first neuron of the output layer
        u = tf.reshape(u, (-1, 1))

        G_sig = 1.0 # to fullfil Neumann BC
        D_sig = 3.0 - input_layer[:, 0] # choose D_sig so, that sig_PINN vanishes where Neumann bc is located -> D_sig(x=3) != 0
        sig = G_sig + D_sig * output_layer[:, 1] # stress is the second neuron of the output layer
        sig = tf.reshape(sig, (-1, 1))

        self.ann = tf.keras.Model(inputs=[input_layer], outputs=[u, sig], name="ANN")
        self.built = True

    def residual(self, inputs):
        """Calculate residual of governing equations. Since hard bc's are applied, the Loss consists of 2 parts, namely the material and the balance residual
          which arise from the fundamental laws of continuum mechanics:

            - material residual: sig - E * u_x = residual_material
            - balance residual: sig_x = residual_balance

          This two residuals will be minimized, using the following loss term:
          Loss = Loss_balance + Loss_constitutive, where each part follows MSE of the corresponding residual

        Input:
                - inputs: spatial coordinates as input for the ANN
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            u = self.ann(inputs)[0]
            sig = self.ann(inputs)[1]

        u_x = tape.gradient(u, inputs)
        sig_x = tape.gradient(sig, inputs)

        residual_material = self.E * u_x - sig
        residual_balance = sig_x

        return {"residual_material": residual_material, "residual_balance": residual_balance}

    def call(self, inputs):
        """ Forward pass of the PINN to calculate the residuals at every spatial coordinate point

        Input:
                - inputs: spatial coordinates as input for the ANN
        Output:
                - loss: information on the loss
        """
        inputs = tf.reshape(inputs, (-1, 1))
        residual = self.residual(inputs=inputs)
        r_mat = residual["residual_material"]
        r_bal = residual["residual_balance"]
        loss = tf.reduce_mean(tf.square(r_mat) + tf.square(r_bal))
        self.add_loss(loss)
        return loss

    def predict(self, inputs):
        """Prediction of displacement and stress.

        Input:
                - inputs: spatial coordinates as input for the ANN
        Output:
                - returns a dictionary with predicted displacement and stress
        """
        u = self.ann(inputs)[0] # displacement is the output of first output-neuron
        sig = self.ann(inputs)[1] # stress is the output of second output-neuron
        return {"disp": u, "sig": sig}


if __name__ == "__main__":
    pinn = PINN()
    pinn.compile(optimizer=tf.keras.optimizers.Nadam(), run_eagerly=False)
    pinn.summary()

    # Input Data = spatial coordinates
    coords = tf.linspace(0.0, 3.0, 100)
    history = pinn.fit(coords, epochs=100, verbose=2)

    plt.figure(0)
    plt.title("Loss Curve over Epochs")
    plt.semilogy(history.history["loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid()

    prediction = pinn.predict(coords)
    displacement = prediction["disp"]
    stress = prediction["sig"]

    plt.figure(1)
    plt.plot(coords,displacement, color="green")
    plt.title("PINN displacement prediction")
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("PINN solution for displacement")
    plt.grid()

    plt.figure(2)
    plt.plot(coords, stress, color="blue")
    plt.title("PINN stress prediction")
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("PINN solution for stress")
#    plt.gca().set_xlim([2.9,3.0])
    plt.gca().set_ylim([0.50,1.50])
    plt.grid()

    plt.show()
