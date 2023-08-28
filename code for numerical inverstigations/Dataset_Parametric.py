from sklearn.model_selection import train_test_split
from free_cantilever_analytical import free_cantilever_analytical
from supported_cantilever_analytical import supported_cantilever_analytical
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import random
random.seed(48)
import csv

# Setup
p = 60000                                   # Line load
E = 200000000000                            # Young's modulus
I = 0.000038929334                          # Moment of inertia
L = 2.5                                     # System Lengths
no_collocation_points = 12500        # number of collocation points


# TRAINING & VALIDATION DATA

# samples 
region = qmc.LatinHypercube(d=1)
samples = region.random(n=no_collocation_points)
test_samples = region.random(n=(no_collocation_points//100))

# feature 01 = x-coordinate
feature_01 = np.reshape(qmc.scale(samples, 0.0, L), (-1,))
doubled_feature_01 = np.empty(feature_01.shape[0] * 2)
doubled_feature_01[::2] = feature_01
doubled_feature_01[1::2] = feature_01
feature_01 = doubled_feature_01.reshape(-1,1)

# feature 02 = allocation value
feature_02 = np.reshape(np.tile([0, 1], feature_01.shape[0] // 2), (-1,1))

# Stacking together to dataset
dataset = np.hstack((feature_01, feature_02)) # dim = (total points,2)

# target tensor for residuals
target_tensor = np.reshape((np.zeros_like(dataset[:,1])), (-1,1)) # dim = (total points, 1)

# splitting dataset
X_train, X_val, y_train, y_val = train_test_split(dataset,target_tensor,test_size=0.2, random_state=48,shuffle=False) 

# TEST DATA
f_01_test = np.sort(qmc.scale(test_samples, 0.0, L),axis=0)
f_02_test_free = np.zeros_like(f_01_test)
f_02_test_supp = np.ones_like(f_01_test)

X_test_free = np.concatenate((f_01_test, f_02_test_free), axis = 1) # dim = (no. testing points, 2)
y_test_free = free_cantilever_analytical(X_test_free[:,0],p,E,I,L) # test data for free Cantilever, dim = (no. testing points,)

X_test_supp = np.concatenate((f_01_test, f_02_test_supp), axis = 1) # dim = (no. testing points, 2)
y_test_supp = supported_cantilever_analytical(X_test_supp[:,0],p,E,I,L) # test data for supported Cantilever, dim = (no. testing points,)

# CSV Export
train_path = 'Results/csv/parametric_structured_train_data.csv'
val_path = 'Results/csv/parametric_structured_val_test.csv'
test_path = 'Results/csv/parametric_structured_test_data.csv'

combined_training_data = zip(range(1, len(X_train) + 1), X_train[:, 0], X_train[:, 1], y_train[:,0]) 
combined_val_data = zip(range(1, len(X_val) + 1),X_val[:, 0], X_val[:, 1], y_val[:,0]) 
combined_test_data = zip(range(1, len(X_test_free) + 1),X_test_free[:,0], X_test_free[:,1], y_test_free, X_test_supp[:,0],X_test_supp[:,1], y_test_supp) 

with open(train_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Sample', 'X_train_1', 'X_train_2', 'y_train'])
    for data_row in combined_training_data:
        csv_writer.writerow(data_row)

with open(val_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Sample', 'X_val_1','X_val_2', 'y_val'])
    for data_row in combined_val_data:
        csv_writer.writerow(data_row)

with open(test_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Sample', 'X_test_free_1','X_test_free_2', 'y_test_free', 'X_test_supp_1','X_test_supp_2', 'y_test_supp'])
    for data_row in combined_test_data:
        csv_writer.writerow(data_row)