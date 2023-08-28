import tensorflow as tf
import numpy as np
import csv
from scipy.stats import qmc
import matplotlib.pyplot as plt
from free_cantilever_analytical import free_cantilever_analytical
from supported_cantilever_analytical import supported_cantilever_analytical
from sklearn.model_selection import train_test_split
np.random.seed(48)




# Setup
p = 60000                       # Line load
E = 200000000000                # Young's modulus
I = 0.000038929334              # Moment of inertia
L = 2.5                         # System Lengths
no_collocation_points = 12500   # number of collocation points


# INPUT DATASET
region = qmc.LatinHypercube(d=1)
training_data = region.random(n=no_collocation_points)
coords = qmc.scale(training_data, 0.0, L)
target_tensor = np.zeros_like(coords)

X_train, X_val, y_train, y_val = train_test_split(coords,target_tensor,test_size=0.2, random_state=42) # Training and Validation Dataset dim = (no. training points,1)

testing_data = region.random(n=(X_train.shape[0]//100))
X_test = qmc.scale(testing_data, 0.0, L) # dim = (no. testing points,1)
X_test = np.sort(X_test, axis=None) # dim = (no. testing points,)
#y_test = free_cantilever_analytical(X_test,p,E,I,L) # dim = (no. testing points,)
y_test = supported_cantilever_analytical(X_test,p,E,I,L) 

train_path = 'Results/csv/train_data.csv'
val_path = 'Results/csv/val_test.csv'
test_path = 'Results/csv/test_data.csv'

with open(train_path,'w', newline='') as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow(['Sample', 'X_train'])
  for epoch, loss in enumerate(X_train, start=1):
    csv_writer.writerow([epoch, loss])
  csv_writer.writerow(['Sample', 'y_train'])
  for epoch, loss in enumerate(y_train, start=1):
    csv_writer.writerow([epoch, loss])

with open(val_path,'w', newline='') as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow(['Sample', 'X_val'])
  for epoch, loss in enumerate(X_val, start=1):
    csv_writer.writerow([epoch, loss])
  csv_writer.writerow(['Sample', 'y_val'])
  for epoch, loss in enumerate(y_val, start=1):
    csv_writer.writerow([epoch, loss])

with open(test_path,'w', newline='') as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow(['Sample', 'X_test'])
  for epoch, loss in enumerate(X_test, start=1):
    csv_writer.writerow([epoch, loss])
  csv_writer.writerow(['Sample', 'y_test'])
  for epoch, loss in enumerate(y_test, start=1):
    csv_writer.writerow([epoch, loss])