import tensorflow as tf
import csv
import random
import time
import matplotlib.pyplot as plt
from write_network_summary_free import write_network_summary_free
from write_network_summary_supp import write_network_summary_supp
from keras.callbacks import EarlyStopping
from Dataset_Parametric import X_train, X_val, y_train, y_val, X_test_free, y_test_free, X_test_supp, y_test_supp, E, I, L, p

class ParametricPINN_tuned(tf.keras.Model):
  """ PINN for 1D supported cantilever bending, solving for primary unknown w(x) """
  
  def __init__(self,units_per_layer = [512,128,64],activation = 'tanh', dropout_rate= 0.0, E = 1.0, I = 1.0, p = 1.0, L=1.0):
      """ Input: 
          - E: Young's Modulus for the 1D elasto-static problem [N/m²]
          - I: Area moment of inertia [m⁴]
          - p: Line Load on the cantilever [N/m]
          - L: Model Length [m]
      """
      
      super(ParametricPINN_tuned, self).__init__()
      self.E = E
      self.I = I
      self.p = p
      self.L = L
      
      self.units_per_layer = units_per_layer
      self.activation = activation
      self.droput_rate = dropout_rate
      self.regularization = True
      
      # builds a network's layers while instanciation
      self.build(input_shape=(2,))

  def build(self, input_shape):
    
    input_layer = tf.keras.layers.Input(shape=input_shape)
    hidden_layer = input_layer

    for units in self.units_per_layer:
       hidden_layer = tf.keras.layers.Dense(units=units,activation=self.activation)(hidden_layer)
    
    output_layer = tf.keras.layers.Dense(units=1,name="output_layer")(hidden_layer)

    self.ann = tf.keras.Model(inputs=[input_layer],outputs=[output_layer],name="ANN")
    self.built = True

  def compute_gradients(self,inputs, training=False):
    """Computes up to the 4th derivative of "w(x)" w.r.t. the inputs "x" using Automatic Differentiation
    4th Derivative -> 4-times nested Gradient Tape
    https://www.tensorflow.org/api_docs/python/tf/GradientTape

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
    bc_residual_w_L  = w_L  # 2d Tensor shape=(None,1), where None is a flexible number depending on batch size
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
    """ Output is a 2D Tensor with shape = (no. testing points, 1) """
    loss = self.ann(inputs)
    return loss

  def evaluation(self, X_test, y_test):
      """X_test is a 2D Tensor with (no. testing points, 2)
         y_test is a 1D Tensor with (no. testing points, )
      """
      # bring y_pred in same dimension like y_test      
      y_pred = tf.squeeze(self.predict(X_test)) # (no. testing points,)

      abs_error = tf.abs(y_pred - y_test)

      # comparison to the targets (here: analytical solution)
      composite_loss_on_test_data = self.evaluate(x=X_test,y=y_test, verbose=0)[0] # scalar test loss value acc. to custom loss function
      return [abs_error.numpy(), composite_loss_on_test_data] 

if __name__ == "__main__":

  tf.config.threading.set_inter_op_parallelism_threads(10)
  start_time = time.time()

  # Setup
  num_search_iterations = 500
  epochs=1000
  batch_size=1

  # Input Data
  X_train = tf.convert_to_tensor(X_train) # 2D Tensor shape = (no. training points,2)
  X_test_free = tf.convert_to_tensor(X_test_free)
  X_test_supp = tf.convert_to_tensor(X_test_supp)
  y_train = tf.convert_to_tensor(y_train)
  y_val = tf.convert_to_tensor(y_val)
  y_test_free = tf.convert_to_tensor(y_test_free, dtype=tf.float32)   # 1D Tensor shape = (no. test points,)
  y_test_supp = tf.convert_to_tensor(y_test_supp, dtype=tf.float32) 
  
  # initialize hyperparameter + ranges
  num_layers = [*range(2,6,1)]
  num_neurons = [*range(32,513,32)]
  activation_functions = ['relu', 'sigmoid', 'tanh']
  learning_rates = [0.001, 0.0001, 0.00001]
  #batch_sizes = [*range(8,65,8)]
  patience_values = [*range(20,41,5)]
  dropout_rates = [0.0,0.2,0.4]

  # initialize best hyperparameters
  best_loss = float('inf')
  best_layers = None
  best_units = None
  best_activation = None
  best_patience = None
  best_dropout_rate = None

  for _ in range(num_search_iterations):
    layers = random.choice(num_layers)
    units = [random.choice(num_neurons) for _ in range(layers)]
    activation_func = random.choice(activation_functions)
    learning_rate = random.choice(learning_rates)
    #batch_size = random.choice(batch_sizes)
    patience = random.choice(patience_values)
    dropout_rate = random.choice(dropout_rates)

    pinn = ParametricPINN_tuned(units_per_layer=units,activation=activation_func,dropout_rate=dropout_rate,E=E, I=I, p=p, L=L)
    pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: y_pred)
    pinn.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], validation_data=(X_val, y_val), shuffle=False)
  
    loss = pinn.evaluate(x=X_val, y=y_val, verbose=0)[0] # first entry = composite loss

    # Check if this model has the best loss so far
    if loss < best_loss:
      best_layers = layers
      best_units = units
      best_activation = activation_func
      best_learning_rate = learning_rate
      #best_batch_size = batch_size
      best_loss = loss
      best_pinn= pinn
      best_patience = patience
      best_dropout_rate = dropout_rate

  # Re-Instantiate with best parameters and Re-Compile
  best_pinn = ParametricPINN_tuned(units_per_layer=best_units, activation=best_activation,dropout_rate=best_dropout_rate,E=E, I=I, p=p, L=L)
  best_pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate), loss=lambda y_true, y_pred: y_pred)
  best_pinn.summary()

  # Re-Train the best model
  early_stopping_tuned_model = EarlyStopping(monitor='val_loss', patience=best_patience, restore_best_weights=True) # cancles training after defined no. of epochs without increasing loss
  history_parametric_tuned_model = best_pinn.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping_tuned_model],validation_data=(X_val, y_val), shuffle=False)

  # write summary into *.txt-file
  tf.saved_model.save(best_pinn, f'Results/saved_models/{best_pinn.__class__.__name__}')
  parametric_tuned_model_time = time.time() - start_time 
  write_network_summary_free(best_pinn,X_test_free, y_test_free,parametric_tuned_model_time,best_layers,best_units, best_activation, best_learning_rate,batch_size,best_patience,best_dropout_rate,num_search_iterations)
  write_network_summary_supp(best_pinn,X_test_supp, y_test_supp,parametric_tuned_model_time,best_layers,best_units, best_activation, best_learning_rate,batch_size,best_patience,best_dropout_rate,num_search_iterations)

  prediction_free = best_pinn.predict(X_test_free)
  prediction_supp = best_pinn.predict(X_test_supp)



  loss_file_path = 'Results/csv/parametric_tuned_loss_data.csv'
  prediction_path = 'Results/csv/parametric_tuned_prediction.csv'

  composite_loss_data = history_parametric_tuned_model.history['loss']
  pde_loss_data =   history_parametric_tuned_model.history['output_1_loss']
  w_0_loss_data = history_parametric_tuned_model.history['output_2_loss']
  w_x_0_loss_data =  history_parametric_tuned_model.history['output_3_loss']
  w_L_loss_data = history_parametric_tuned_model.history['output_4_loss']
  w_xx_L_loss_data = history_parametric_tuned_model.history['output_5_loss']
  w_xxx_L_loss_data = history_parametric_tuned_model.history['output_6_loss']

  combined_loss_data = zip(range(1, len(composite_loss_data) + 1),composite_loss_data,pde_loss_data, w_0_loss_data, w_x_0_loss_data, w_L_loss_data, w_xx_L_loss_data, w_xxx_L_loss_data) 

  combined_prediction_data = zip(range(1, len(prediction_free) + 1), X_test_supp[:,0], prediction_free[:, 0], y_test_free[:, 0], prediction_supp[:,0], y_test_supp[:,0]) 

  with open(prediction_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X_test', 'Prediction Free Canti', 'y_test_free', 'Prediction Supp Canti', 'y_test_supp'])
    for data_row in combined_prediction_data:
        csv_writer.writerow(data_row)

  with open(loss_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Composite Loss', 'PDE Loss', 'w0 Loss', 'wx0 Loss', 'wL Loss', 'wxxL Loss', 'wxxxL Loss'])
    for data_row in combined_loss_data:
        csv_writer.writerow(data_row)

    






  
  # plt.figure("Training Loss Curve",  figsize=(12, 6))
  # plt.semilogy(history_parametric_tuned_model.history["loss"], label=f'training_composite_loss_{best_pinn.__class__.__name__}')
  # plt.title("Training Loss Curve over Epochs")
  # plt.xlabel("Epochs")
  # plt.ylabel("Loss")
  # plt.legend()
  # plt.grid()
  # filename = f'Results/Loss Curve/Train_Composite_Loss_parametric.png'
  # plt.savefig(filename, dpi=300)

  # # plt.figure("Train and Test Loss Curves",figsize=(12, 6))
  # # plt.semilogy(history_parametric_tuned_model.history["loss"], label=f'training_loss_{best_pinn.__class__.__name__}')
  # # #plt.axhline(best_pinn.evaluate(X_test_free)[0],label=f'test_loss_free_cantilever')
  # # #plt.axhline(best_pinn.evaluate(X_test_supp)[0],label=f'test_loss_supp_cantilever')
  # # plt.title("Loss Curves over Epochs")
  # # plt.xlabel("Epochs")
  # # plt.ylabel("Loss")
  # # plt.legend()
  # # plt.grid()
  # # filename = f'Results/Loss Curve/Train_Test_Losses_parametric.png'
  # # plt.savefig(filename, dpi=300)

  # plt.figure("Training Loss Curve all Terms",  figsize=(12, 6))
  # plt.semilogy(history_parametric_tuned_model.history["loss"], label='composite loss')
  # plt.semilogy(history_parametric_tuned_model.history['output_1_loss'], label='pde loss')     #pde loss
  # plt.semilogy(history_parametric_tuned_model.history['output_2_loss'], label = 'w(0)')       #left_bc_w_0
  # plt.semilogy(history_parametric_tuned_model.history['output_3_loss'], label='w_x(0)')       #left_bc_w_0_x
  # plt.semilogy(history_parametric_tuned_model.history['output_4_loss'], label='w(L)')         #right_bc_w_L  
  # plt.semilogy(history_parametric_tuned_model.history['output_5_loss'], label='w_xx(L)')      #right_bc_w_L_xx
  # plt.semilogy(history_parametric_tuned_model.history['output_6_loss'], label='w_xxx(L)')     #right_bc_w_L_xxx

  # plt.title("Training Losses over Epochs of the tuned parametric PINN")
  # plt.xlabel("Epochs")
  # plt.ylabel("Loss")
  # plt.legend()
  # plt.grid()
  # filename = f'Results/Loss Curve/Train_All_Loss_Terms_parametric.png'
  # plt.savefig(filename, dpi=300)

  # prediction_free = best_pinn.predict(X_test_free)
  # prediction_supp = best_pinn.predict(X_test_supp)

  # plt.figure("Prediction vs. Analytical Solution on Free Cantilever Dataset",  figsize=(12, 6))
  # plt.scatter(X_test_free[:,0],prediction_free[:,0], label='Prediction', marker='x', s=20)
  # plt.plot(X_test_free[:,0], y_test_free, label='Analytical Solution Free Cantilever', c='orange')
  # plt.title("Prediction on Free Cantilever Test Dataset")
  # plt.xlabel("Spatial Coordinate x [m]")
  # plt.ylabel("Bending w(x) [m]")
  # plt.legend()
  # plt.grid()
  # filename = f'Results/Prediction/prediction_parametric_free.png'
  # plt.savefig(filename, dpi=300)


  # plt.figure("Prediction vs. Analytical Solution on Supported Cantilever Dataset",  figsize=(12, 6))
  # plt.scatter(X_test_supp[:,0],prediction_supp[:,0], label='Prediction', marker='x', s=20)
  # plt.plot(X_test_supp[:,0], y_test_supp, label='Analytical Solution Supported Cantilever', c='orange')
  # plt.title("Prediction on Supported Cantilever Test Dataset")
  # plt.xlabel("Spatial Coordinate x [m]")
  # plt.ylabel("Bending w(x) [m]")
  # plt.legend()
  # plt.grid()
  # filename = f'Results/Prediction/prediction_parametric_supported.png'
  # plt.savefig(filename, dpi=300)

  # plt.show()