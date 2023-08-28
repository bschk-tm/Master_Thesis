import time
import random
import tensorflow as tf
from keras.callbacks import EarlyStopping
#from callback_class import LossCurveCallback
#from ParametricPINN_tuned import ParametricPINN_tuned
from CantileverPINN_tuned import CantileverPINN_tuned
from Supported_CantileverPINN_tuned import Supported_CantileverPINN_tuned
from write_network_summary import write_network_summary
from Dataset import X_train, X_val, y_train, y_val, E, L, p, I 
#from Dataset_Parametric import X_train as X_train_param,y_train as y_train_param, X_val as X_val_param,y_val as y_val_param, X_test as X_test_param, y_test as y_test_param 



def hyperparameter_tuning(X_test, y_test, start_time, num_search_iterations, epochs, is_free_canti_tuned_model,is_supported_canti_tuned_model,is_parametric_tuned_model):
  """ hyperparameter tuning for each network given in the argument
      returns two lists containing first the tuned models and second the training histories of the tuned models """
  
  # tuned models lists
  tuned_models = []
  histories_tuned_models = []
  

  # initialize hyperparameter + ranges
  num_layers = [*range(2,6,1)]
  num_neurons = [*range(32,512,16)]
  activation_functions = ['relu', 'sigmoid', 'tanh', 'selu']
  learning_rates = [0.001, 0.0001, 0.00001]
  batch_sizes = [*range(8,65,8)]
  patience_values = [*range(20,41,10)]
  dropout_rates = [0.0,0.2,0.4]

  # initialize best hyperparameters
  best_loss = float('inf')
  best_layers = None
  best_units = None
  best_activation = None
  best_patience = None
  best_dropout_rate = None
    
# 1. MECHANICAL MODEL
   
  if is_free_canti_tuned_model:


    # random search for tuning
    for _ in range(num_search_iterations):
      layers = random.choice(num_layers)
      units = [random.choice(num_neurons) for _ in range(layers)]
      activation_func = random.choice(activation_functions)
      learning_rate = random.choice(learning_rates)
      batch_size = random.choice(batch_sizes)
      patience = random.choice(patience_values)
      dropout_rate = random.choice(dropout_rates)

      # instanciation and compilation
      free_canti_tuned_model = CantileverPINN_tuned(units_per_layer=units,activation=activation_func,dropout_rate=dropout_rate,E=E, I=I, p=p, L=L)
      free_canti_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: y_pred)

      # training including early stopping
      free_canti_tuned_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], validation_data=(X_val, y_val))
      
      # evaluate model on validation set
      loss = free_canti_tuned_model.evaluate(x=X_val, y=y_val, verbose=0)[0] # first entry = composite loss
      
      # Check if this model has the best loss so far
      if loss < best_loss:
        best_layers = layers
        best_units = units
        best_activation = activation_func
        best_learning_rate = learning_rate
        best_batch_size = batch_size
        best_loss = loss
        best_free_canti_tuned_model = free_canti_tuned_model
        best_patience = patience
        best_dropout_rate = dropout_rate

    # Re-Instantiate with best parameters and Re-Compile
    best_free_canti_tuned_model = CantileverPINN_tuned(units_per_layer=best_units, activation=best_activation,dropout_rate=best_dropout_rate,E=E, I=I, p=p, L=L)
    best_free_canti_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate), loss=lambda y_true, y_pred: y_pred)
    best_free_canti_tuned_model.summary()

    # Re-Train the best model
    early_stopping_tuned_model = EarlyStopping(monitor='val_loss', patience=best_patience, restore_best_weights=True) # cancles training after defined no. of epochs without increasing loss
    history_free_canti_tuned_model = best_free_canti_tuned_model.fit(x=X_train, y=y_train, batch_size=best_batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping_tuned_model],validation_data=(X_val, y_val))

    # write summary into *.txt-file
    free_canti_tuned_time = time.time() - start_time 
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    write_network_summary(best_free_canti_tuned_model,X_test, y_test,free_canti_tuned_time,best_layers,best_units, best_activation, best_learning_rate,best_batch_size,best_patience,best_dropout_rate,num_search_iterations)

    tuned_models.append(best_free_canti_tuned_model)      
    histories_tuned_models.append(history_free_canti_tuned_model)

# 2. MECHANICAL MODEL

  if is_supported_canti_tuned_model:
    


    # random search for tuning
    for _ in range(num_search_iterations):
      layers = random.choice(num_layers)
      units = [random.choice(num_neurons) for _ in range(layers)]
      activation_func = random.choice(activation_functions)
      learning_rate = random.choice(learning_rates)
      batch_size = random.choice(batch_sizes)
      patience = random.choice(patience_values)
      dropout_rate = random.choice(dropout_rates)

      # instanciation and compilation
      supported_canti_tuned_model = Supported_CantileverPINN_tuned(units_per_layer=units,activation=activation_func,dropout_rate=dropout_rate,E=E, I=I, p=p, L=L)
      supported_canti_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: y_pred)

      # training including early stopping
      supported_canti_tuned_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], validation_data=(X_val, y_val))
      
      # evaluate model on validation set
      loss = supported_canti_tuned_model.evaluate(x=X_val, y=y_val, verbose=0)[0] # first entry = composite loss
      
      # Check if this model has the best loss so far
      if loss < best_loss:
        best_layers = layers
        best_units = units
        best_activation = activation_func
        best_learning_rate = learning_rate
        best_batch_size = batch_size
        best_loss = loss
        best_supported_canti_tuned_model = supported_canti_tuned_model
        best_patience = patience
        best_dropout_rate = dropout_rate

    # Re-Instantiate with best parameters and Re-Compile
    best_supported_canti_tuned_model = Supported_CantileverPINN_tuned(units_per_layer=best_units, activation=best_activation,dropout_rate=best_dropout_rate,E=E, I=I, p=p, L=L)
    best_supported_canti_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate), loss=lambda y_true, y_pred: y_pred)
    best_supported_canti_tuned_model.summary()

    # Re-Train the best model
    early_stopping_tuned_model = EarlyStopping(monitor='val_loss', patience=best_patience, restore_best_weights=True) # cancles training after defined no. of epochs without increasing loss
    history_supported_canti_tuned_model = best_supported_canti_tuned_model.fit(x=X_train, y=y_train, batch_size=best_batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping_tuned_model],validation_data=(X_val, y_val))

    # write summary into *.txt-file
    supp_canti_tuned_time = time.time() - start_time 
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    write_network_summary(best_supported_canti_tuned_model,X_test, y_test,supp_canti_tuned_time,best_layers,best_units, best_activation, best_learning_rate,best_batch_size,best_patience,best_dropout_rate,num_search_iterations)

    tuned_models.append(best_supported_canti_tuned_model)      
    histories_tuned_models.append(history_supported_canti_tuned_model)
    

# 3. MECHANICAL MODEL  
  if is_parametric_tuned_model:
    pass 
    # # Input Data
    # X_train = tf.convert_to_tensor(X_train_param) # 2D Tensor shape = (no. training points,2)
    # X_test = tf.convert_to_tensor(X_test_param)
    # X_val = tf.convert_to_tensor(X_val_param)
    # y_train = tf.convert_to_tensor(y_train_param)
    # y_val = tf.convert_to_tensor(y_val_param)
    # y_test = tf.convert_to_tensor(y_test_param, dtype=tf.float32)   # 1D Tensor shape = (no. test points,)

    # # random search for tuning
    # for _ in range(num_search_iterations):
    #   layers = random.choice(num_layers)
    #   units = [random.choice(num_neurons) for _ in range(layers)]
    #   activation_func = random.choice(activation_functions)
    #   learning_rate = random.choice(learning_rates)
    #   batch_size = random.choice(batch_sizes)
    #   patience = random.choice(patience_values)
    #   dropout_rate = random.choice(dropout_rates)

    #   pinn = ParametricPINN_tuned(units_per_layer=units,activation=activation_func,dropout_rate=dropout_rate,E=E, I=I, p=p, L=L)
    #   pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: y_pred)
    #   pinn.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], validation_data=(X_val, y_val))
    
    #   loss = pinn.evaluate(x=X_val, y=y_val, verbose=0)[0] # first entry = composite loss

    #   # Check if this model has the best loss so far
    #   if loss < best_loss:
    #     best_layers = layers
    #     best_units = units
    #     best_activation = activation_func
    #     best_learning_rate = learning_rate
    #     best_batch_size = batch_size
    #     best_loss = loss
    #     best_pinn= pinn
    #     best_patience = patience
    #     best_dropout_rate = dropout_rate

    # # Re-Instantiate with best parameters and Re-Compile
    # best_pinn = ParametricPINN_tuned(units_per_layer=best_units, activation=best_activation,dropout_rate=best_dropout_rate,E=E, I=I, p=p, L=L)
    # best_pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate), loss=lambda y_true, y_pred: y_pred)
    # best_pinn.summary()

    # # Re-Train the best model
    # early_stopping_tuned_model = EarlyStopping(monitor='val_loss', patience=best_patience, restore_best_weights=True) # cancles training after defined no. of epochs without increasing loss
    # history_parametric_tuned_model = best_pinn.fit(x=X_train, y=y_train, batch_size=best_batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping_tuned_model],validation_data=(X_val, y_val))

    # # write summary into *.txt-file
    # parametric_tuned_model_time = time.time() - start_time 
    # write_network_summary(best_pinn,X_test, y_test,parametric_tuned_model_time,best_layers,best_units, best_activation, best_learning_rate,best_batch_size,best_patience,best_dropout_rate,num_search_iterations)
    
    # tuned_models.append(best_pinn)      
    # histories_tuned_models.append(history_parametric_tuned_model)

  return tuned_models, histories_tuned_models      
