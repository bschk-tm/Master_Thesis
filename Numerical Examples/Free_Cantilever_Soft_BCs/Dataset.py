import tensorflow as tf
from free_cantilever_analytical import analytical_solution
from sklearn.model_selection import train_test_split


# Setup
p = 1.0                         # Line load
E = 1.0                         # Young's modulus
I = 1.0                         # Moment of inertia
L = 1.0                         # System Lengths
no_collocation_points = 10000   # number of collocation points

# INPUT DATASET
coords = tf.linspace(0.0,L, no_collocation_points)
coords_tensor = tf.reshape(coords, (-1,1)) 
target_tensor = tf.zeros_like(coords_tensor)

X_train, X_val, y_train, y_val = train_test_split(coords_tensor.numpy(),target_tensor.numpy(),test_size=0.3, random_state=42) # Training and Validation Dataset
X_test = tf.sort(tf.random.uniform(shape=(no_collocation_points,), minval=0.0, maxval=L))
y_test = analytical_solution(X_test,E,I,p,L)   