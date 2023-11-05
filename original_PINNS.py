import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Want to review the layers and also activation function


# Defining layer
def diffusion_PINN():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,))
    ])

    # Number of hidden layers
    for i in range(4):
        model.add(tf.keras.layers.Dense(64, activation='tanh'))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))

    return model


# Create an instance of model
model = diffusion_PINN()
normal_neural_net = diffusion_PINN()



# Loss function
def pinns_loss(y_true, y_pred):
    # Coefficients/constants (change if necessary) - maybe should be defined globally
    D = 1.0  # Diffusion coefficient
    sigma_a = 0.5  # Absorption cross-section
    nu_Sigma_f = 0.2  # Fission-related term
    S = 2.0  # Source term - started with no net neutron source
    L = D/sigma_a

    x = tf.constant(np.linspace(0, a, 100), dtype=tf.float32)

    # Compute the first and the second derivatives of phi_pred with respect to x
    with tf.GradientTape() as t1:
        t1.watch(x)
        with tf.GradientTape() as t2:
            t2.watch(x)
            phi_pred = model(x)
        dphi_dx = t2.gradient(phi_pred, x)
    d2phi_dx2 = t1.gradient(dphi_dx, x)


    # Residual loss (from book)
    L_residual = (d2phi_dx2 - (1/L**2)* phi_pred) # rest of eq:  - nu_Sigma_f * phi_pred - S
    # Boundary condition loss

    # Boundary condition J(a) = 0
    L_residual += dphi_dx[-1]

    L_residual += phi_pred[-1]

    # J = derivative of phi (current)
    # This is boundary condition J+=1/2*S
    #L_residual += (dphi_dx[0] - 0.5 * S)

    # MSE of residual
    mse_residual = tf.reduce_mean(tf.square(L_residual))

    return mse_residual

def traditional_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

def combined_loss(y_true, y_pred):
    trad_loss = traditional_loss(y_true, y_pred)
    pinn_loss_val = pinns_loss(y_true, y_pred)  # Make sure to pass the necessary inputs here
    total_loss = trad_loss + pinn_loss_val
    return total_loss

# Compile model
model.compile(loss=combined_loss, optimizer='adam', run_eagerly=True)
normal_neural_net.compile(loss=['MeanAbsoluteError'], optimizer='adam')


# Define parameters for the problem
D = 1.0           # Diffusion coefficient
sigma_a = 0.5     # Absorption cross-section
nu_Sigma_f = 0.2  # Fission-related term
S = 2.0           # Source term - set as 2 for now
a = 10.0          # Length of the domain
L = math.sqrt(D/sigma_a) # L^2 = D/sigma_a

# x is number of points for data (10 evenly spaced 1m apart for now)
num_points = 2
x = np.linspace(0, 2, num_points)
x_val_points = np.linspace(0, a, 100)


# Generate training data
def generate_training_data(x):
    # Calculate the true neutron flux based on the diffusion equation pg 51 in Stacey
    #phi_true = 4*S*((np.sinh(a-x)/L)+(2*D/L)*np.cosh(a-x)/L)/((((2*(D/L)+1)**2)*np.exp(a/L))-((2*(D/L)-1)**2)*np.exp(-a/L))
    phi_true = ((L*S)/(2*D))*np.exp(-x/L)
    return x, phi_true


x_train, phi_train_true = generate_training_data(x)
x_test, phi_test = generate_training_data(x_val_points)



# Train the model
num_epochs = 200  # YChange if necessary ~ did 1000 took roughly 1.5 hrs set as 10 for now
batch_size = 32

history = model.fit(x_train, phi_train_true, epochs=num_epochs, batch_size=batch_size)
history_nn = normal_neural_net.fit(x_train, phi_train_true, epochs=num_epochs, batch_size=batch_size)


# Plot loss - MAE of ~ 3x10^-5
# Want to look at r2 value next

def loss_plotter(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.show()


# Evaluate with test data
plt.plot(x_test, phi_test, label='Analytical solution')
plt.plot(x_test, model.predict(x_test), label='PINN')
plt.plot(x_test, normal_neural_net.predict(x_test), label='Traditional NN')
plt.legend()
plt.show()




