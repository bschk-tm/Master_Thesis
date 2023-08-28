import time
import random 
random.seed(48)
import tensorflow as tf
from Dataset import X_test,y_test
#from Dataset_Parametric import X_test, y_test
from visualization import visualize
from model_decision import model_decision


########### setup ###########
tf.config.threading.set_inter_op_parallelism_threads(10)
epochs = 1000
num_search_iterations = 500 
#############################
is_free_canti_model = False
is_free_canti_tuned_model = False
is_supported_canti_model = True #  y_test anpassen
is_supported_canti_tuned_model = True
#############################
batch_size = 32
start_time = time.time()
predictions = []
evaluations = []
#period = 10 # needed for ModelCheckpoint
#############################

# model instanciation
tuned_models, untuned_models, histories_tuned_models, histories_untuned_models = model_decision(start_time,batch_size = batch_size, epochs= epochs, is_free_canti_model=is_free_canti_model, is_free_canti_tuned_model=is_free_canti_tuned_model, is_supported_canti_model=is_supported_canti_model,is_supported_canti_tuned_model = is_supported_canti_tuned_model,num_search_iterations=num_search_iterations)


# predict + evaluate + save the different models
if is_free_canti_model or is_supported_canti_model:
    for untuned_model in untuned_models:
        tf.saved_model.save(untuned_model, f'Results/saved_models/{untuned_model.__class__.__name__}')
        predictions.append(untuned_model.predict(X_test))                            # prediction of w(x)
        evaluations.append(untuned_model.evaluation(X_test=X_test,y_test=y_test))    # loss and absolute error on Test Dataset   

if is_free_canti_tuned_model or is_supported_canti_tuned_model:
    for tuned_model in tuned_models:
        tf.saved_model.save(tuned_model, f'Results/saved_models/{tuned_model.__class__.__name__}')
        predictions.append(tuned_model.predict(X_test))                        
        evaluations.append(tuned_model.evaluation(X_test=X_test,y_test=y_test)) 

# Stop tracking the calculation time
end_time = time.time()
calculation_time = end_time - start_time

# plotting and saving the results
visualize(X_test, y_test, models= untuned_models + tuned_models, histories=histories_untuned_models + histories_tuned_models, evaluations=evaluations, predictions=predictions,composite_loss = True, loss_terms = True)
#plotly_visualize(X_test, y_test, models= untuned_models + tuned_models, histories=histories_untuned_models + histories_tuned_models, evaluations=evaluations, predictions=predictions,composite_loss = True, loss_terms = True)