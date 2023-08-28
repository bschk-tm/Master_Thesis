import tensorflow as tf
import matplotlib.pyplot as plt
from Dataset_Parametric import X_train, X_val, y_train, y_val, X_test_free, y_test_free, X_test_supp, y_test_supp, E, I, L, p
from write_network_summary_free import write_network_summary_free
from write_network_summary_supp import write_network_summary_supp
import csv
import time


class ParametricPINN(tf.keras.Model):
  def __init__(self, E=1.0, I=1.0,p = 1.0, L=1.0):
      super(ParametricPINN, self).__init__()
      self.E = E
      self.I = I
      self.p = p
      self.L = L

      self.build(input_shape=(2,))

  def build(self, input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape,name="input_layer")

    hidden_layer = tf.keras.layers.Dense(units=416,activation="tanh",name="hidden_layer_0")(input_layer)

    hidden_layer = tf.keras.layers.Dense(units=96,activation="tanh",name="hidden_layer_1")(hidden_layer)

    output_layer = tf.keras.layers.Dense(units=1,name="output_layer")(hidden_layer)

    self.ann = tf.keras.Model(inputs=[input_layer],outputs=[output_layer],name="ANN")
    
    self.built = True

  def compute_gradients(self,inputs, training=False):
      """Computes up to the 4th derivative of "w(x)" w.r.t. the inputs "x" using Automatic Differentiation
      4th Derivative -> 4-times nested Gradient Tape
      https://www.tensorflow.org/api_docs/python/tf/GradientTape

      - inputs: is a 1D Tensor containing 2 elements [x,a], shape=(2,)
      """

      with tf.GradientTape(persistent=True) as tape: # fourth derivative
          tape.watch(inputs)
          with tf.GradientTape(persistent=True) as tape2:
              tape2.watch(inputs)
              with tf.GradientTape(persistent=True) as tape3:
                  tape3.watch(inputs)
                  with tf.GradientTape(persistent=True) as tape4:
                      tape4.watch(inputs)

                      # calculation of network output w(x) via propagating the input through the layers
                      w = self.ann(inputs=inputs)
                  
                  w_x = tape.gradient(w,inputs)     
              w_xx = tape.gradient(w_x,inputs)
          w_xxx = tape.gradient(w_xx,inputs)
      w_xxxx = tape.gradient(w_xxx,inputs)

      return [w,w_x,w_xx,w_xxx,w_xxxx]

  def residual(self, inputs):
    # Input Shape = (None,2) which means a 2D-Tensor (Matrix) with unspecified batchsize in the first dimension(None), this can be any positive integer value (variable batch size)
    # Second dimension shows that each sample has 2 elements = features
    x = inputs[:,0]
    a = inputs[:,1] # 1D Tensor (Vector) with shape = (None,)

    ones = tf.ones_like(a) # 1D Tensor (Vector) with shape = (None,)
    zeros = tf.zeros_like(x) 
 
    input_left_bc = tf.stack([zeros,a], axis=1) 
    input_right_bc = tf.stack([ones * self.L,a], axis=1)

    field_gradients = self.compute_gradients(inputs)
    left_bc_gradients = self.compute_gradients(input_left_bc)
    right_bc_gradients = self.compute_gradients(input_right_bc)

    w_xxxx = field_gradients[4]
    w_0, w_0_x = left_bc_gradients[0], left_bc_gradients[1]
    w_L, w_L_xx, w_L_xxx = right_bc_gradients[0], right_bc_gradients[2], right_bc_gradients[3]

    pde_residual = w_xxxx + (self.p/(self.E * self.I))
    bc_residual_w_0 = w_0 
    bc_residual_w_L  = w_L 
    bc_residual_w_0_x = w_0_x
    bc_residual_w_L_xx = w_L_xx  
    bc_residual_w_L_xxx = w_L_xxx

    return pde_residual, bc_residual_w_0, bc_residual_w_L, bc_residual_w_0_x, bc_residual_w_L_xx, bc_residual_w_L_xxx
     
 
  def call(self, inputs):
 
    a = tf.reshape(inputs[:,1],(-1,1)) # 1D Tensor (Vector) with shape = (None,), containing allocation values   
    ones = tf.ones_like(a)

    pde_residual, bc_residual_w_0, bc_residual_w_L, bc_residual_w_0_x, bc_residual_w_L_xx, bc_residual_w_L_xxx = self.residual(inputs=inputs)

    # usage of tf.mean_reduce for the bc-residuals leads to a scalar loss output
    pde_loss = tf.reduce_mean(tf.square(pde_residual)) # 0D Tensor -> scalar value 
    left_bc_w_0_loss = tf.reduce_mean(tf.square(bc_residual_w_0))
    left_bc_w_0_x_loss = tf.reduce_mean(tf.square(bc_residual_w_0_x))
    right_bc_w_L_loss =  tf.reduce_mean(tf.square(a * bc_residual_w_L)) # 0D Tensor > scalar value
    right_bc_w_L_xx_loss = tf.reduce_mean(tf.square(bc_residual_w_L_xx))
    right_bc_w_L_xxx_loss =  tf.reduce_mean(tf.square((ones-a) * bc_residual_w_L_xxx))
      
   
    loss =  pde_loss + left_bc_w_0_loss + left_bc_w_0_x_loss + right_bc_w_L_loss + right_bc_w_L_xx_loss  + right_bc_w_L_xxx_loss # 0D Tensor -> scalar value
    

    return loss, pde_loss, left_bc_w_0_loss, left_bc_w_0_x_loss, right_bc_w_L_loss, right_bc_w_L_xx_loss, right_bc_w_L_xxx_loss
  
  def predict(self, inputs):
  
    return self.ann(inputs)
  

if __name__ == "__main__":
  
  tf.config.threading.set_inter_op_parallelism_threads(10)
  batch_sizes = [1]
  for i,batch_size in enumerate(batch_sizes):

    start_time = time.time()

    epochs = 100
    batch_size = batch_size

    X_train = tf.convert_to_tensor(X_train) # 2D Tensor shape = (no. training points,2)
    y_train = tf.convert_to_tensor(y_train)

    X_test_free = tf.convert_to_tensor(X_test_free)
    y_test_free = tf.convert_to_tensor(y_test_free)

    X_test_supp = tf.convert_to_tensor(X_test_supp)
    y_test_supp = tf.convert_to_tensor(y_test_supp)

    pinn = ParametricPINN(E,I,p,L)
    pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lambda y_true, y_pred: y_pred)
    pinn.summary() 

    history = pinn.fit(X_train,y_train,epochs=epochs,batch_size=batch_size, verbose=2, shuffle=False)

    prediction_free = pinn.predict(X_test_free)
    prediction_supp = pinn.predict(X_test_supp)  

    tf.saved_model.save(pinn, f'Results/saved_models/{pinn.__class__.__name__}_iter{i}_batchsize{batch_size}')


    loss_file_path = f'Results/csv/parametric_loss_data_iter{i}_batchsize{batch_size}.csv'
    prediction_path = f'Results/csv/parametric_prediction_iter{i}_batchsize{batch_size}.csv'

 
    combined_prediction_data = zip(range(1, len(prediction_free) + 1), X_test_supp[:,0], prediction_free[:, 0], y_test_free[:, 0], prediction_supp[:,0], y_test_supp[:,0]) 

    with open(prediction_path, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(['X_test', 'Prediction Free Canti', 'y_test_free', 'Prediction Supp Canti', 'y_test_supp'])
      for data_row in combined_prediction_data:
          csv_writer.writerow(data_row)

    composite_loss_data = history.history['loss']
    pde_loss_data =   history.history['output_1_loss']
    w_0_loss_data = history.history['output_2_loss']
    w_x_0_loss_data =  history.history['output_3_loss']
    w_L_loss_data = history.history['output_4_loss']
    w_xx_L_loss_data = history.history['output_5_loss']
    w_xxx_L_loss_data = history.history['output_6_loss']

    combined_loss_data = zip(range(1, len(composite_loss_data) + 1),composite_loss_data,pde_loss_data, w_0_loss_data, w_x_0_loss_data, w_L_loss_data, w_xx_L_loss_data, w_xxx_L_loss_data) 

    with open(loss_file_path, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(['Epoch', 'Composite Loss', 'PDE Loss', 'w0 Loss', 'wx0 Loss', 'wL Loss', 'wxxL Loss', 'wxxxL Loss'])
      for data_row in combined_loss_data:
          csv_writer.writerow(data_row)
    
    calculation_time = time.time() - start_time

    with open(f'Results/Protocols/output_{pinn.__class__.__name__}_iter{i}.txt', 'w') as file:
      file.write(f""" Layer: 2
                      Units: [416,96]
                      Activation: tanh
                      Learning Rate: 0.001
                      Batch Size: {batch_size}
                      Model Name: {pinn.__class__.__name__}
                      Calculation Time: {calculation_time} seconds """)


  

    # plt.figure(1)
    # plt.semilogy(history.history["loss"])
    # plt.title("Loss Curve over Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.grid()


    # plt.figure(2)
    # plt.plot(X_test_free[:,0],prediction_free[:,0], label='Prediction')
    # plt.plot(X_test_free[:,0], y_test_free, label='Analytical Solution Free Cantilever')
    # plt.title("Prediction on Free Cantilever Test Dataset")
    # plt.xlabel("Spatial Coordinate")
    # plt.ylabel("Bending")
    # plt.legend()
    # plt.grid()


    # plt.figure(3)
    # plt.plot(X_test_supp[:,0],prediction_supp[:,0], label='Prediction')
    # plt.plot(X_test_supp[:,0], y_test_supp, label='Analytical Solution Supported Cantilever')
    # plt.title("Prediction on Supported Cantilever Test Dataset")
    # plt.xlabel("Spatial Coordinate")
    # plt.ylabel("Bending")
    # plt.legend()
    # plt.grid()

    # plt.show()
