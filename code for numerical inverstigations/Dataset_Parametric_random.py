from sklearn.model_selection import train_test_split
from free_cantilever_analytical import free_cantilever_analytical
from supported_cantilever_analytical import supported_cantilever_analytical
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import csv

# Setup
p = 60000                       # Line load
E = 200000000000                # Young's modulus
I = 0.000038929334              # Moment of inertia
L = 2.5                         # System Lengths
no_collocation_points = 12500   # number of collocation points
# is_free = True                  # Test Data for a free cantilever
# is_supported = False            # Test Data for a supported cantilever

# TRAINING DATA
# samples 
region = qmc.LatinHypercube(d=1)
dataset_01 = region.random(n=no_collocation_points)
dataset_02 = region.random(n=no_collocation_points)
testing_dataset = region.random(n=(no_collocation_points//100))

# feature 01 = x-coordinate
feature_01_set_01 = np.sort(qmc.scale(dataset_01, 0.0, L),axis=0)
feature_01_set_02 = np.sort(qmc.scale(dataset_02, 0.0, L), axis=0)
feature_01 = np.concatenate((feature_01_set_01,feature_01_set_02), axis=0)

# feature 02 = parametric value a_2
feature_02_set_01 = np.zeros_like(dataset_01)
feature_02_set_02 = np.ones_like(dataset_02)
feature_02 = np.concatenate((feature_02_set_01,feature_02_set_02), axis=0)

# putting together
dataset = np.concatenate((feature_01,feature_02), axis=1)

# target tensor for residuals
target_tensor = np.reshape((np.zeros_like(dataset[:,1])), (-1,1)) # dim = (2 * no. collocation points, 1)

# samples dim = (no. training points,2), targets dim = (no. training points,1)
X_train, X_val, y_train, y_val = train_test_split(dataset,target_tensor,test_size=0.2, random_state=42) 

#print(X_train[:,0].shape)

# TEST DATA
feature_01_test = np.sort(qmc.scale(testing_dataset, 0.0, L), axis=0) # dim = (no. testing points,1)


feature_02_free = np.zeros_like(feature_01_test) # Free Cantilever
X_test_free = np.concatenate((feature_01_test, feature_02_free), axis = 1) # dim = (no. testing points, 2)
y_test_free = free_cantilever_analytical(X_test_free[:,0],p,E,I,L) # test data for free Cantilever, dim = (no. testing points,)

feature_02_supported = np.ones_like(feature_01_test) # supported Cantilever
X_test_supp = np.concatenate((feature_01_test, feature_02_supported), axis = 1) # dim = (no. testing points, 2)
y_test_supp = supported_cantilever_analytical(X_test_supp[:,0],p,E,I,L) # test data for supported Cantilever, dim = (no. testing points,)




# CSV Export
train_path = 'Results/csv/parametric_unstructured_train_data.csv'
val_path = 'Results/csv/parametric_unstructured_val_test.csv'
test_path = 'Results/csv/parametric_unstructured_test_data.csv'

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