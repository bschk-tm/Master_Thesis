import tensorflow as tf
# All the possible Models
from CantileverPINN_vanilla import CantileverPINN_vanilla
from CantileverPINN_vanilla_HPT import CantileverPINN_vanilla_HPT
from CantileverPINN_Regularization import CantileverPINN_Regularization
from CantileverPINN_Regularization_HPT import CantileverPINN_Regularization_HPT
# Dataset and methods
from visualization import visualize
from keras.callbacks import EarlyStopping
from callback_class import LossCurveCallback
from Dataset import X_train, X_val, X_test, y_train, y_val, y_test, E, L, p, I 


# List of Hyperparameters:
# - Layer
# - Neurons
# - Activation Function
# - Regularization: None, L1, L2, L1L2, Dropout
# - Dropout Rate, Norm-Strength
# - Early Stopping Patience
# - Optimizer Adam/Nadam etc.
# - Learning Rate

# Setup
learning_rate = 0.001
patience = 20
period = 100
vanilla  = True
regularization = True
hyperparameter_tuning = True 

   

# INSTANCIATION
#model = CantileverPINN_vanilla_HPT(E,I,p,L)
#model = CantileverPINN_Regularization_HPT(E,I,p,L)
#model = CantileverPINN_Regularization(E,I,p,L)
model = CantileverPINN_vanilla(E,I,p,L)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: y_pred, metrics=['mse'])
model.summary()

# TRAINING (TRAINING AND VALIDATION DATASET)
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) # cancles training after 10 epochs without further improvement
loss_curve_callback = LossCurveCallback('Weights/weights.h5', period=period, x_test=X_test, y_test=y_test, model=model) # using ModelCheckpoint to evaluate @ distinct epochs without disturbing the training process
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=1000, verbose=2, callbacks=[loss_curve_callback, early_stopping], validation_data=(X_val, y_val))

# EVALUATION (TEST DATASET)
prediction = model.predict(X_test)
evaluation = model.evaluation(x_test=X_test,y_test=y_test)
   
# VISUALIZATION
visualize(history,model, evaluation, X_test, prediction, y_test,composite_loss = True, val_loss = True, loss_terms = False)