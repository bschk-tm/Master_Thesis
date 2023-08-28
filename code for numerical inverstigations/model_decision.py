import time
import tensorflow as tf
from CantileverPINN import CantileverPINN
from Supported_CantileverPINN import Supported_CantileverPINN
from hyperparameter_tuning import hyperparameter_tuning
from write_network_summary import write_network_summary
from Dataset import X_train, X_test, y_train, y_test, E, L, p, I 





def model_decision(start_time, batch_size, epochs, is_free_canti_model=False, is_free_canti_tuned_model=False, is_supported_canti_model=False, is_supported_canti_tuned_model=False,is_parametric_tuned_model=False, num_search_iterations=1):

  untuned_models, tuned_models = [],[]
  histories_untuned_models, histories_tuned_models = [],[]

# Models without HPT  
 
  if is_free_canti_model:
  

    # instanciation and compilation
    free_canti_model = CantileverPINN(E,I,p,L)
    free_canti_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lambda y_true, y_pred: y_pred)

    # training
    history_free_canti_model = free_canti_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    # saving + writing
    free_canti_time = time.time() - start_time
    write_network_summary(free_canti_model, X_test, y_test,free_canti_time)
    untuned_models.append(free_canti_model)
    histories_untuned_models.append(history_free_canti_model)

  if is_supported_canti_model:

   
    
    # instanciation and compilation
    supported_canti_model = Supported_CantileverPINN(E,I,p,L)
    supported_canti_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lambda y_true, y_pred: y_pred)

    # training
    history_supported_canti_model = supported_canti_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    # saving + writing
    supp_canti_time = time.time() - start_time
    write_network_summary(supported_canti_model, X_test, y_test, supp_canti_time)
    untuned_models.append(supported_canti_model)
    histories_untuned_models.append(history_supported_canti_model)


# Models with HPT  

  if is_free_canti_tuned_model or is_supported_canti_tuned_model or is_parametric_tuned_model:
    
    tuned_models, histories_tuned_models = hyperparameter_tuning(X_test, y_test, start_time,num_search_iterations,epochs,is_free_canti_tuned_model,is_supported_canti_tuned_model,is_parametric_tuned_model)

  return tuned_models, untuned_models, histories_tuned_models, histories_untuned_models